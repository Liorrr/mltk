"""Tests for mltk.domains.llm.retrieval -- retrieval ranking metrics."""

import math

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.retrieval import (
    assert_map_at_k,
    assert_mrr,
    assert_ndcg,
    assert_recall_at_k,
)


# ------------------------------------------------------------------
# nDCG@K
# ------------------------------------------------------------------

class TestNDCG:
    """nDCG@K -- normalized discounted cumulative gain."""

    def test_perfect_ranking_passes(self) -> None:
        # SCENARIO: Documents are already sorted by descending relevance.
        # WHY: The predicted ranking matches the ideal ranking exactly.
        # EXPECTED: nDCG = 1.0, passes any threshold <= 1.0.
        y_true = [[3, 2, 1, 0]]
        y_scores = [[1.0, 0.8, 0.5, 0.1]]
        result = assert_ndcg(y_true, y_scores, k=4, min_ndcg=0.99)
        assert result.passed is True
        assert abs(result.details["ndcg"] - 1.0) < 1e-9

    def test_reversed_ranking_fails(self) -> None:
        # SCENARIO: Least relevant doc ranked first, most relevant last.
        # WHY: This is the worst possible ranking -- nDCG should be low.
        # EXPECTED: nDCG << 0.8, raises MltkAssertionError.
        y_true = [[3, 2, 1, 0]]
        y_scores = [[0.1, 0.3, 0.7, 1.0]]  # reversed
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_ndcg(y_true, y_scores, k=4, min_ndcg=0.8)
        assert exc_info.value.result.details["ndcg"] < 0.8

    def test_k_truncation(self) -> None:
        # SCENARIO: k=2 but there are 4 documents.
        # WHY: Only the top-2 ranked docs should contribute to DCG.
        # EXPECTED: Perfect ranking in top-2 gives nDCG@2 = 1.0.
        y_true = [[3, 2, 0, 0]]
        y_scores = [[1.0, 0.9, 0.2, 0.1]]
        result = assert_ndcg(y_true, y_scores, k=2, min_ndcg=0.99)
        assert result.passed is True
        assert abs(result.details["ndcg"] - 1.0) < 1e-9
        assert result.details["k"] == 2

    def test_multiple_queries_averaged(self) -> None:
        # SCENARIO: Two queries -- one perfect, one imperfect.
        # WHY: Mean nDCG averages across queries.
        # EXPECTED: Average is between the two per-query scores.
        y_true = [[3, 0], [0, 3]]
        y_scores = [[1.0, 0.0], [1.0, 0.0]]
        # Query 0: perfect (nDCG=1.0).  Query 1: worst (score
        # puts irrelevant doc first).
        result = assert_ndcg(
            y_true, y_scores, k=2, min_ndcg=0.1,
        )
        assert result.passed is True
        per_q = result.details["per_query_ndcg"]
        assert abs(per_q[0] - 1.0) < 1e-9
        assert per_q[1] < 1.0
        assert result.details["num_queries"] == 2

    def test_empty_queries_trivially_passes(self) -> None:
        # SCENARIO: No queries at all.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Trivially passes with ndcg=1.0.
        result = assert_ndcg([], [], k=5, min_ndcg=0.8)
        assert result.passed is True
        assert result.details["ndcg"] == 1.0
        assert result.details["num_queries"] == 0

    def test_all_zero_relevance(self) -> None:
        # SCENARIO: All documents have relevance 0.
        # WHY: IDCG is 0, so nDCG is defined as 1.0.
        # EXPECTED: Passes.
        y_true = [[0, 0, 0]]
        y_scores = [[0.5, 0.3, 0.1]]
        result = assert_ndcg(y_true, y_scores, k=3, min_ndcg=0.5)
        assert result.passed is True
        assert abs(result.details["ndcg"] - 1.0) < 1e-9

    def test_result_has_timing(self) -> None:
        # SCENARIO: Every assertion should populate duration_ms.
        # WHY: @timed_assertion decorator must be active.
        # EXPECTED: duration_ms > 0.
        result = assert_ndcg(
            [[1, 0]], [[0.9, 0.1]], k=2, min_ndcg=0.1,
        )
        assert result.duration_ms > 0


# ------------------------------------------------------------------
# MRR
# ------------------------------------------------------------------

class TestMRR:
    """MRR -- mean reciprocal rank."""

    def test_first_result_relevant(self) -> None:
        # SCENARIO: Every query has its first result marked relevant.
        # WHY: Reciprocal rank = 1/1 = 1.0 for each query.
        # EXPECTED: MRR = 1.0, passes.
        results = [[True, False], [True, True]]
        result = assert_mrr(results, min_mrr=1.0)
        assert result.passed is True
        assert abs(result.details["mrr"] - 1.0) < 1e-9

    def test_no_relevant_results(self) -> None:
        # SCENARIO: No query has any relevant result.
        # WHY: All reciprocal ranks are 0.
        # EXPECTED: MRR = 0.0, fails when min_mrr > 0.
        results = [[False, False], [False, False, False]]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_mrr(results, min_mrr=0.1)
        assert exc_info.value.result.details["mrr"] == 0.0

    def test_mixed_ranks(self) -> None:
        # SCENARIO: Query 0 has relevant at rank 2, query 1 at rank 1.
        # WHY: MRR = (1/2 + 1/1) / 2 = 0.75.
        # EXPECTED: Passes with min_mrr=0.7.
        results = [[False, True, False], [True, False]]
        result = assert_mrr(results, min_mrr=0.7)
        assert result.passed is True
        assert abs(result.details["mrr"] - 0.75) < 1e-9

    def test_empty_queries_trivially_passes(self) -> None:
        # SCENARIO: No queries at all.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Trivially passes with mrr=1.0.
        result = assert_mrr([], min_mrr=0.5)
        assert result.passed is True
        assert result.details["mrr"] == 1.0

    def test_single_query_relevant_at_rank_3(self) -> None:
        # SCENARIO: One query, first relevant result at position 3.
        # WHY: RR = 1/3 ≈ 0.333.
        # EXPECTED: Fails when min_mrr=0.5.
        results = [[False, False, True]]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_mrr(results, min_mrr=0.5)
        rr = exc_info.value.result.details["mrr"]
        assert abs(rr - 1.0 / 3.0) < 1e-9

    def test_per_query_rr_in_details(self) -> None:
        # SCENARIO: Verify per-query reciprocal ranks are stored.
        # WHY: Useful for debugging which queries hurt MRR.
        # EXPECTED: per_query_rr list has correct values.
        results = [[True], [False, True]]
        result = assert_mrr(results, min_mrr=0.5)
        per_rr = result.details["per_query_rr"]
        assert abs(per_rr[0] - 1.0) < 1e-9
        assert abs(per_rr[1] - 0.5) < 1e-9


# ------------------------------------------------------------------
# Recall@K
# ------------------------------------------------------------------

class TestRecallAtK:
    """Recall@K -- fraction of relevant docs in top K."""

    def test_all_relevant_retrieved(self) -> None:
        # SCENARIO: All relevant docs appear in the top K.
        # WHY: Recall@K should be 1.0.
        # EXPECTED: Passes.
        relevant = [{"d1", "d2", "d3"}]
        retrieved = [["d1", "d2", "d3", "d4", "d5"]]
        result = assert_recall_at_k(
            relevant, retrieved, k=5, min_recall=1.0,
        )
        assert result.passed is True
        assert abs(result.details["recall"] - 1.0) < 1e-9

    def test_none_retrieved(self) -> None:
        # SCENARIO: Top K has zero relevant documents.
        # WHY: Recall@K = 0.
        # EXPECTED: Fails when min_recall > 0.
        relevant = [{"d1", "d2"}]
        retrieved = [["d5", "d6", "d7"]]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_recall_at_k(
                relevant, retrieved, k=3, min_recall=0.5,
            )
        assert exc_info.value.result.details["recall"] == 0.0

    def test_partial_recall(self) -> None:
        # SCENARIO: 2 out of 4 relevant docs in top 3.
        # WHY: Recall@3 = 2/4 = 0.5.
        # EXPECTED: Passes with min_recall=0.5.
        relevant = [{"d1", "d2", "d3", "d4"}]
        retrieved = [["d1", "d3", "d7", "d2", "d4"]]
        result = assert_recall_at_k(
            relevant, retrieved, k=3, min_recall=0.5,
        )
        assert result.passed is True
        assert abs(result.details["recall"] - 0.5) < 1e-9

    def test_k_larger_than_results(self) -> None:
        # SCENARIO: k=100 but only 3 results returned.
        # WHY: Should consider all 3 results without error.
        # EXPECTED: Recall@100 = 2/2 = 1.0 (all relevant found).
        relevant = [{"d1", "d2"}]
        retrieved = [["d1", "d2", "d3"]]
        result = assert_recall_at_k(
            relevant, retrieved, k=100, min_recall=1.0,
        )
        assert result.passed is True
        assert abs(result.details["recall"] - 1.0) < 1e-9

    def test_empty_queries_trivially_passes(self) -> None:
        # SCENARIO: No queries at all.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Trivially passes with recall=1.0.
        result = assert_recall_at_k([], [], k=5, min_recall=0.8)
        assert result.passed is True
        assert result.details["num_queries"] == 0

    def test_empty_relevant_set_for_query(self) -> None:
        # SCENARIO: A query has no relevant documents defined.
        # WHY: Recall is undefined; treated as 1.0 (trivial).
        # EXPECTED: That query contributes 1.0 to the mean.
        relevant = [set(), {"d1"}]
        retrieved = [["d5"], ["d1"]]
        result = assert_recall_at_k(
            relevant, retrieved, k=5, min_recall=1.0,
        )
        assert result.passed is True
        per_q = result.details["per_query_recall"]
        assert abs(per_q[0] - 1.0) < 1e-9
        assert abs(per_q[1] - 1.0) < 1e-9


# ------------------------------------------------------------------
# MAP@K
# ------------------------------------------------------------------

class TestMAPAtK:
    """MAP@K -- mean average precision at K."""

    def test_perfect_ranking(self) -> None:
        # SCENARIO: Relevant docs ranked at positions 1 and 2.
        # WHY: AP = (1/1 + 2/2) / 2 = 1.0.
        # EXPECTED: MAP = 1.0, passes.
        relevant = [{"d1", "d2"}]
        retrieved = [["d1", "d2", "d3"]]
        result = assert_map_at_k(
            relevant, retrieved, k=3, min_map=1.0,
        )
        assert result.passed is True
        assert abs(result.details["map_score"] - 1.0) < 1e-9

    def test_imperfect_ranking(self) -> None:
        # SCENARIO: Two relevant docs at positions 1 and 3.
        # WHY: AP = (1/1 + 2/3) / 2 = 0.8333.
        # EXPECTED: Passes with min_map=0.8.
        relevant = [{"d1", "d3"}]
        retrieved = [["d1", "d2", "d3"]]
        result = assert_map_at_k(
            relevant, retrieved, k=3, min_map=0.8,
        )
        assert result.passed is True
        expected_ap = (1.0 / 1 + 2.0 / 3) / 2
        assert abs(
            result.details["map_score"] - expected_ap
        ) < 1e-9

    def test_no_relevant_in_top_k(self) -> None:
        # SCENARIO: No relevant documents in the top K.
        # WHY: AP = 0.0.
        # EXPECTED: Fails when min_map > 0.
        relevant = [{"d1", "d2"}]
        retrieved = [["d5", "d6", "d7"]]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_map_at_k(
                relevant, retrieved, k=3, min_map=0.5,
            )
        assert exc_info.value.result.details["map_score"] == 0.0

    def test_multiple_queries(self) -> None:
        # SCENARIO: Two queries with different AP scores.
        # WHY: MAP averages AP across queries.
        # EXPECTED: MAP is the mean of per-query APs.
        relevant = [{"d1"}, {"d2"}]
        retrieved = [["d1", "d3"], ["d3", "d2"]]
        # Query 0: AP = (1/1) / 1 = 1.0
        # Query 1: AP = (1/2) / 1 = 0.5
        # MAP = (1.0 + 0.5) / 2 = 0.75
        result = assert_map_at_k(
            relevant, retrieved, k=3, min_map=0.5,
        )
        assert result.passed is True
        assert abs(result.details["map_score"] - 0.75) < 1e-9

    def test_empty_queries_trivially_passes(self) -> None:
        # SCENARIO: No queries at all.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Trivially passes with map=1.0.
        result = assert_map_at_k([], [], k=5, min_map=0.5)
        assert result.passed is True
        assert result.details["map_score"] == 1.0

    def test_k_truncation(self) -> None:
        # SCENARIO: Relevant doc at position 4 but k=2.
        # WHY: Only top-2 results are considered; doc at 4 is ignored.
        # EXPECTED: AP considers only positions 1-2.
        relevant = [{"d1", "d4"}]
        retrieved = [["d2", "d1", "d3", "d4"]]
        # Top-2: ["d2", "d1"].  d1 is relevant at rank 2.
        # AP = (1/2) / 2 = 0.25  (2 relevant total, 1 found)
        result = assert_map_at_k(
            relevant, retrieved, k=2, min_map=0.1,
        )
        assert result.passed is True
        assert abs(
            result.details["map_score"] - 0.25
        ) < 1e-9

    def test_result_has_timing(self) -> None:
        # SCENARIO: @timed_assertion must populate duration_ms.
        # WHY: Performance tracking requirement.
        # EXPECTED: duration_ms > 0.
        result = assert_map_at_k(
            [{"d1"}], [["d1"]], k=1, min_map=0.5,
        )
        assert result.duration_ms > 0
