"""Tests for mltk.training.checkpoint — checkpoint completeness and resume validation."""

import json
from pathlib import Path

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.checkpoint import (
    assert_checkpoint_complete,
    assert_resume_loss_continuous,
)


class TestCheckpointComplete:
    """Assert checkpoint file exists and contains required keys."""

    def test_checkpoint_complete_valid(self, tmp_path: Path) -> None:
        # SCENARIO: Valid JSON checkpoint with both default required keys present.
        # WHY: The happy path — a properly saved checkpoint should pass immediately.
        # EXPECTED: Passes — all required keys found, missing_keys=[].
        ckpt = tmp_path / "ckpt_epoch1.json"
        ckpt.write_text(json.dumps({"epoch": 1, "model_state": {"w": 0.5}, "loss": 0.42}))
        result = assert_checkpoint_complete(str(ckpt))
        assert result.passed is True
        assert result.details["missing_keys"] == []

    def test_checkpoint_missing_file(self, tmp_path: Path) -> None:
        # SCENARIO: Checkpoint file does not exist at the provided path.
        # WHY: Training pipelines that crash before saving produce missing files;
        #      the assertion should fail with a clear error rather than an OSError.
        # EXPECTED: Fails with MltkAssertionError mentioning the file path.
        missing = tmp_path / "nonexistent_ckpt.json"
        with pytest.raises(MltkAssertionError) as exc:
            assert_checkpoint_complete(str(missing))
        assert "not found" in str(exc.value).lower()

    def test_checkpoint_missing_keys(self, tmp_path: Path) -> None:
        # SCENARIO: Checkpoint JSON exists but only has "epoch" — missing "model_state".
        # WHY: Some frameworks save a minimal checkpoint during interrupted training;
        #      a checkpoint without model weights cannot be used for resumption.
        # EXPECTED: Fails — missing_keys=["model_state"].
        ckpt = tmp_path / "ckpt_partial.json"
        ckpt.write_text(json.dumps({"epoch": 3}))
        with pytest.raises(MltkAssertionError) as exc:
            assert_checkpoint_complete(str(ckpt))
        assert "missing" in str(exc.value).lower()

    def test_checkpoint_custom_required_keys(self, tmp_path: Path) -> None:
        # SCENARIO: Caller requires ["epoch", "optimizer_state", "scaler_state"].
        # WHY: Mixed-precision training needs the GradScaler state; callers should
        #      be able to declare custom required keys.
        # EXPECTED: Passes when all three custom keys are present.
        ckpt = tmp_path / "ckpt_full.json"
        ckpt.write_text(json.dumps({
            "epoch": 10,
            "optimizer_state": {"lr": 1e-4},
            "scaler_state": {"scale": 65536},
        }))
        result = assert_checkpoint_complete(
            str(ckpt),
            required_keys=["epoch", "optimizer_state", "scaler_state"],
        )
        assert result.passed is True

    def test_checkpoint_invalid_json(self, tmp_path: Path) -> None:
        # SCENARIO: File exists but contains malformed JSON (truncated write).
        # WHY: Disk-full or SIGKILL during save produces corrupt checkpoints;
        #      the assertion must catch the parse error gracefully.
        # EXPECTED: Fails with MltkAssertionError mentioning JSON validity.
        ckpt = tmp_path / "ckpt_corrupt.json"
        ckpt.write_text('{"epoch": 5, "model_state": {BROKEN')
        with pytest.raises(MltkAssertionError) as exc:
            assert_checkpoint_complete(str(ckpt))
        assert "json" in str(exc.value).lower()

    def test_checkpoint_accepts_path_object(self, tmp_path: Path) -> None:
        # SCENARIO: Path passed as pathlib.Path object instead of str.
        # WHY: The function signature accepts str | Path; both should work identically.
        # EXPECTED: Passes — Path object resolved correctly.
        ckpt = tmp_path / "ckpt_path.json"
        ckpt.write_text(json.dumps({"epoch": 2, "model_state": {}}))
        result = assert_checkpoint_complete(ckpt)   # Pass Path, not str
        assert result.passed is True


class TestResumeLossContinuous:
    """Assert loss does not spike when training resumes from a checkpoint."""

    def test_resume_loss_continuous(self) -> None:
        # SCENARIO: Pre-checkpoint losses decrease smoothly; post-resume losses
        #           continue from approximately the same level.
        # WHY: A correctly restored checkpoint should resume without a loss spike.
        # EXPECTED: Passes — gap between last pre and first post is within 0.5.
        pre = [2.0, 1.8, 1.6, 1.4, 1.2]
        post = [1.25, 1.1, 1.0, 0.9]
        result = assert_resume_loss_continuous(pre, post, max_gap=0.5)
        assert result.passed is True
        assert result.details["last_pre_loss"] == pytest.approx(1.2)
        assert result.details["first_post_loss"] == pytest.approx(1.25)

    def test_resume_loss_discontinuous(self) -> None:
        # SCENARIO: Post-resume loss jumps from 1.0 to 3.5 — optimizer state not restored.
        # WHY: When only model weights are saved (not optimizer/scheduler state),
        #      learning rate and momentum restart from scratch causing a visible spike.
        # EXPECTED: Fails — gap of 2.5 exceeds max_gap=0.5 threshold.
        pre = [1.5, 1.2, 1.0]
        post = [3.5, 3.0, 2.5]
        with pytest.raises(MltkAssertionError) as exc:
            assert_resume_loss_continuous(pre, post, max_gap=0.5)
        assert "discontinuity" in str(exc.value).lower()

    def test_resume_loss_exact_match(self) -> None:
        # SCENARIO: First post-resume loss equals last pre-checkpoint loss exactly.
        # WHY: Perfect continuity edge case — gap=0.0 must always pass.
        # EXPECTED: Passes — gap is 0.0.
        pre = [0.8, 0.75, 0.70]
        post = [0.70, 0.65, 0.60]
        result = assert_resume_loss_continuous(pre, post, max_gap=0.5)
        assert result.passed is True
        assert result.details["gap"] == pytest.approx(0.0, abs=1e-9)

    def test_resume_loss_custom_gap(self) -> None:
        # SCENARIO: Gap of 0.8 is acceptable when max_gap=1.0.
        # WHY: Some schedulers produce a controlled warm-restart spike that is
        #      expected behavior; callers can raise the allowed gap accordingly.
        # EXPECTED: Passes under max_gap=1.0.
        pre = [1.0]
        post = [1.8]
        result = assert_resume_loss_continuous(pre, post, max_gap=1.0)
        assert result.passed is True

    def test_resume_loss_empty_pre(self) -> None:
        # SCENARIO: pre_losses is an empty list.
        # WHY: Edge case — caller provides no pre-checkpoint data; assertion must
        #      fail cleanly rather than raising IndexError.
        # EXPECTED: Fails with MltkAssertionError mentioning empty pre_losses.
        with pytest.raises(MltkAssertionError) as exc:
            assert_resume_loss_continuous([], [1.0, 0.9], max_gap=0.5)
        assert "pre_losses" in str(exc.value).lower()

    def test_resume_loss_empty_post(self) -> None:
        # SCENARIO: post_losses is an empty list.
        # WHY: Edge case — caller provides no post-resume data; must fail cleanly.
        # EXPECTED: Fails with MltkAssertionError mentioning empty post_losses.
        with pytest.raises(MltkAssertionError) as exc:
            assert_resume_loss_continuous([1.0, 0.9], [], max_gap=0.5)
        assert "post_losses" in str(exc.value).lower()
