"""
Agent module: single multimodal Gemini call per claim → EvidenceFacts.

Public interface
----------------
``extract_evidence(request)``
    Accept a ``VLMRequest`` from ``grounding.py``, execute one Gemini call
    with all submitted images and the assembled prompt, validate the JSON
    response via ``validator.py``, and return ``EvidenceFacts``.

    Maximum 3 total attempts (1 initial + 2 retries) with exponential backoff.
    Raises ``AgentError`` when all attempts are exhausted.

``AgentError``
    Raised when the Gemini call fails permanently after all retries.

Design notes
------------
Image transport (Gemini inline bytes):
  - All image bytes are loaded from disk and sent as inline base64 via the
    ``types.Blob`` / ``types.Part(inline_data=...)`` mechanism.
  - Images larger than ``config.MAX_IMAGE_BYTES`` are skipped with a warning;
    the claim still proceeds with the remaining images.
  - Supported MIME types: JPEG, PNG, WebP, GIF, HEIC, HEIF.
  - Unsupported extensions default to ``image/jpeg`` with a warning.

Gemini call structure:
  - ``system_instruction`` is passed as a separate API config field
    (not embedded in the user prompt).
  - All images follow the user prompt text in the same Content turn.
  - ``response_mime_type='application/json'`` enforces structured output.
  - ``temperature=0.0`` for maximum determinism.

Timeout:
  - Enforced via ``concurrent.futures.ThreadPoolExecutor`` with a 60-second
    wall-clock deadline per attempt.

Output contract:
  - Only ``EvidenceFacts`` is returned.
  - No claim_status, evidence_standard_met, risk_flags, or any decision-layer
    field may appear in the return value.
  - The validator strips any such fields defensively.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, MAX_IMAGE_BYTES, MAX_RETRIES, REQUEST_TIMEOUT, RETRY_BASE_DELAY, SEED
from grounding import VLMRequest
from schemas import EvidenceFacts
from validator import parse_response_text, validate_evidence_facts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}

_DEFAULT_MIME: str = "image/jpeg"


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class AgentError(RuntimeError):
    """Raised when all Gemini call attempts are exhausted."""


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Return a cached ``genai.Client``.

    Validates that ``GEMINI_API_KEY`` is set.

    Raises:
        RuntimeError: If ``GEMINI_API_KEY`` is empty.
    """
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set.  "
                "Add it to code/.env or export it as an environment variable."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
        logger.debug("Gemini client initialised.")
    return _client


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def extract_evidence(request: VLMRequest) -> EvidenceFacts:
    """Execute one Gemini multimodal call and return validated ``EvidenceFacts``.

    Performs up to ``MAX_RETRIES + 1`` total attempts (1 initial + 2 retries
    by default).  Exponential backoff: ``RETRY_BASE_DELAY * 2^attempt`` seconds
    between retries.

    Args:
        request: A frozen ``VLMRequest`` from ``grounding.build_vlm_request``.

    Returns:
        Validated ``EvidenceFacts`` with image observations and findings.
        Advisory fields (confidence, severity_estimate) are present but
        must not gate decisions in ``rules.py``.

    Raises:
        AgentError: If all ``MAX_RETRIES + 1`` attempts fail.

    Examples:
        >>> from grounding import build_vlm_request
        >>> request = build_vlm_request(parsed, context, use_checklist=True)
        >>> facts = extract_evidence(request)
        >>> len(facts.image_observations)
        3
    """
    client = _get_client()
    valid_image_ids = _image_ids_from_paths(request.image_paths)
    contents = _build_contents(request)
    config = _build_config(request)

    last_error: Optional[Exception] = None
    total_attempts = MAX_RETRIES + 1  # 1 initial + MAX_RETRIES retries

    for attempt in range(total_attempts):
        if attempt > 0:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed (%s); retrying in %.1fs.",
                attempt,
                total_attempts,
                last_error,
                delay,
            )
            time.sleep(delay)

        try:
            logger.debug(
                "Attempt %d/%d — model=%s images=%d prompt_chars=%d",
                attempt + 1,
                total_attempts,
                request.model,
                len(request.image_paths),
                len(request.user_prompt),
            )
            response = _call_with_timeout(
                client=client,
                model=request.model,
                contents=contents,
                config=config,
                timeout=REQUEST_TIMEOUT,
            )
            raw_dict = parse_response_text(response.text)
            facts = validate_evidence_facts(raw_dict, valid_image_ids)
            logger.info(
                "Gemini call succeeded on attempt %d: "
                "%d observations, %d findings.",
                attempt + 1,
                len(facts.image_observations),
                len(facts.findings),
            )
            return facts

        except TimeoutError as exc:
            logger.warning("Attempt %d timed out: %s", attempt + 1, exc)
            last_error = exc

        except Exception as exc:
            logger.warning("Attempt %d error: %s", attempt + 1, exc)
            last_error = exc

    raise AgentError(
        f"All {total_attempts} Gemini call attempts failed. "
        f"Last error: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _image_ids_from_paths(image_paths: tuple[str, ...]) -> list[str]:
    """Derive image_id strings from the image_paths in the request.

    image_id = filename stem (e.g. 'img_1' from 'img_1.jpg').
    """
    return [Path(p).stem for p in image_paths]


def _get_mime_type(path: str) -> str:
    """Return the MIME type for an image path based on its extension."""
    ext = Path(path).suffix.lower()
    mime = _SUPPORTED_MIME.get(ext)
    if mime is None:
        logger.warning(
            "Unsupported image extension %r for %s; defaulting to %s.",
            ext,
            path,
            _DEFAULT_MIME,
        )
        return _DEFAULT_MIME
    return mime


def _load_image_part(path: str) -> Optional[types.Part]:
    """Load an image file and return a Gemini ``Part``.

    Returns ``None`` if the file is missing, unreadable, or exceeds the
    ``MAX_IMAGE_BYTES`` size limit.  Returning ``None`` means the image is
    silently skipped — the claim still proceeds with remaining images.

    Args:
        path: Absolute path to the image file.

    Returns:
        ``types.Part`` with inline image data, or ``None`` on failure.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Image file not found: %s; skipping.", path)
        return None

    file_size = p.stat().st_size
    if file_size > MAX_IMAGE_BYTES:
        logger.warning(
            "Image %s is %dMB, exceeds limit of %dMB; skipping.",
            p.name,
            file_size // (1024 * 1024),
            MAX_IMAGE_BYTES // (1024 * 1024),
        )
        return None

    try:
        data = p.read_bytes()
        mime = _get_mime_type(path)
        return types.Part(
            inline_data=types.Blob(mime_type=mime, data=data)
        )
    except OSError as exc:
        logger.warning("Failed to read image %s: %s; skipping.", path, exc)
        return None


# ---------------------------------------------------------------------------
# Gemini call assembly
# ---------------------------------------------------------------------------


def _build_contents(request: VLMRequest) -> list[types.Content]:
    """Assemble the Gemini ``contents`` list.

    Structure:
      - One ``Content`` with role ``'user'``.
      - First Part: the full text prompt (rules + checklist + claim + schema).
      - Subsequent Parts: one inline image Part per successfully loaded image,
        in the order they appear in ``request.image_paths``.

    Images that fail to load are skipped (logged at WARNING level).

    Args:
        request: The ``VLMRequest`` from grounding.py.

    Returns:
        List with a single ``Content`` element.
    """
    parts: list[types.Part] = [types.Part(text=request.user_prompt)]

    loaded_count = 0
    for path in request.image_paths:
        img_part = _load_image_part(path)
        if img_part is not None:
            parts.append(img_part)
            loaded_count += 1

    if loaded_count == 0 and request.image_paths:
        logger.warning(
            "No images could be loaded for this claim (%d path(s) attempted). "
            "Proceeding with text-only call.",
            len(request.image_paths),
        )
    else:
        logger.debug("Loaded %d/%d images.", loaded_count, len(request.image_paths))

    return [types.Content(role="user", parts=parts)]


def _build_config(request: VLMRequest) -> types.GenerateContentConfig:
    """Build the Gemini ``GenerateContentConfig``.

    Sets:
    - ``system_instruction`` from the VLMRequest.
    - ``temperature=0.0`` (from request, always 0.0 per architecture).
    - ``response_mime_type='application/json'`` for structured output.
    - ``seed`` for reproducibility where supported.

    Args:
        request: The ``VLMRequest`` from grounding.py.

    Returns:
        Configured ``GenerateContentConfig``.
    """
    return types.GenerateContentConfig(
        system_instruction=request.system_instruction,
        temperature=request.temperature,
        response_mime_type=request.response_mime_type,
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Timeout wrapper
# ---------------------------------------------------------------------------


def _call_with_timeout(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
    timeout: float,
) -> types.GenerateContentResponse:
    """Call ``client.models.generate_content`` with a wall-clock timeout.

    Uses a ``ThreadPoolExecutor`` to enforce the deadline without requiring
    async infrastructure.

    Args:
        client: The Gemini client.
        model: Model identifier string.
        contents: List of ``Content`` objects.
        config: ``GenerateContentConfig``.
        timeout: Wall-clock timeout in seconds.

    Returns:
        The Gemini ``GenerateContentResponse``.

    Raises:
        TimeoutError: If the call does not complete within *timeout* seconds.
        Exception: Any exception raised by the Gemini SDK propagates as-is.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Gemini request timed out after {timeout:.0f}s."
            )
