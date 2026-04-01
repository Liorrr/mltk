"""Image loading and encoding utilities for multimodal assertions.

All multimodal assertions that accept images use the ``ImageInput``
type alias, which accepts file paths (str or Path) and raw bytes.
A single ``load_image()`` helper resolves any variant to raw bytes,
and ``image_to_base64()`` converts to a base64 string suitable for
embedding in LLM prompts.

Pillow is an **optional** dependency.  When assertions receive an
``image_description`` string instead of raw image data, Pillow is
never imported -- this lets users run multimodal evaluations on any
machine without installing image processing libraries.

Design decision (ADR D2): ``ImageInput = str | Path | bytes``.
PIL.Image objects are NOT accepted directly to avoid coupling the
public API to Pillow.  Users convert PIL images to bytes first
with ``image.tobytes()`` or by saving to a BytesIO buffer.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

ImageInput = str | Path | bytes
"""Type alias for image inputs accepted by multimodal assertions.

- ``str``: file path to an image on disk
- ``Path``: pathlib path to an image on disk
- ``bytes``: raw image data (e.g., PNG/JPEG bytes)
"""


def load_image(source: ImageInput) -> bytes:
    """Load an image from any supported source and return raw bytes.

    Accepts file paths (str or Path) and raw bytes.  For file paths,
    reads the file in binary mode.  For bytes, returns as-is.

    Pillow is NOT required for this function -- it performs raw I/O
    only.  Pillow is only needed if you want to validate that the
    bytes represent a valid image (use ``_validate_image_bytes``).

    Args:
        source: Image source -- file path or raw bytes.

    Returns:
        Raw image bytes.

    Raises:
        TypeError: If source is not str, Path, or bytes.
        FileNotFoundError: If source is a path that does not exist.

    Example:
        >>> data = load_image("photo.png")
        >>> isinstance(data, bytes)
        True
    """
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes()
    raise TypeError(
        f"Expected str, Path, or bytes; "
        f"got {type(source).__name__}"
    )


def image_to_base64(source: ImageInput) -> str:
    """Convert an image to a base64-encoded string.

    Loads the image via ``load_image()`` and encodes the raw bytes
    to base64.  The result is suitable for embedding in LLM prompts
    as a data URI or inline image reference.

    Args:
        source: Image source -- file path or raw bytes.

    Returns:
        Base64-encoded string of the image bytes.

    Example:
        >>> b64 = image_to_base64(b"\\x89PNG...")
        >>> isinstance(b64, str)
        True
    """
    raw = load_image(source)
    return base64.b64encode(raw).decode("ascii")


def _validate_image_pillow(source: ImageInput) -> dict:
    """Validate image bytes using Pillow and return metadata.

    This is the only function in _image.py that requires Pillow.
    It is called lazily -- only when a user does NOT provide an
    ``image_description`` and the assertion needs to inspect the
    image to build a prompt.

    Args:
        source: Image source to validate.

    Returns:
        Dict with keys: format, mode, width, height.

    Raises:
        ImportError: If Pillow is not installed, with install hint.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required to inspect images. "
            "Install with: pip install mltk[multimodal]"
        ) from None

    raw = load_image(source)
    img = Image.open(io.BytesIO(raw))
    return {
        "format": img.format or "UNKNOWN",
        "mode": img.mode,
        "width": img.size[0],
        "height": img.size[1],
    }


def _build_image_prompt(
    instruction: str,
    image: ImageInput | None = None,
    image_description: str | None = None,
) -> str:
    """Build an evaluation prompt that includes image context.

    Two paths (design decision D3):

    1. **Description path** (``image_description`` provided): Uses the
       text description directly.  No Pillow, no base64, no image
       loading.  Best for users who pre-describe images with a VLM.

    2. **Base64 path** (``image`` provided, no description): Loads and
       base64-encodes the image, embedding it in the prompt.  The
       user's ``judge_fn`` must support base64 image data.

    If both are provided, ``image_description`` takes precedence
    (cheaper, faster, no Pillow needed).

    Args:
        instruction: The evaluation instruction/rubric text.
        image: Optional image source for base64 encoding.
        image_description: Optional pre-computed text description.

    Returns:
        Complete prompt string with image context embedded.

    Raises:
        ValueError: If neither image nor image_description provided.
    """
    if image_description is not None:
        return (
            f"{instruction}\n\n"
            f"## Image Description\n"
            f"{image_description}\n"
        )

    if image is not None:
        b64 = image_to_base64(image)
        return (
            f"{instruction}\n\n"
            f"## Image (base64)\n"
            f"{b64}\n"
        )

    raise ValueError(
        "Either 'image' or 'image_description' must be provided."
    )
