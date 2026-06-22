"""
Central configuration for the multimodal evidence review system.

All secrets MUST be read from environment variables.
Never hardcode API keys, tokens, or credentials in this file.

VLM provider: Google Gemini (sole provider — no provider abstractions).
Model: gemini-2.5-flash (overridable via VLM_MODEL env var).
Required env var: GEMINI_API_KEY.
SDK: google-genai.

Image transport is handled in agent.py.
This module does not perform image loading or encoding.
"""

from __future__ import annotations

import os
from pathlib import Path

# Auto-load code/.env so callers don't need to call load_dotenv themselves.
# Silent no-op if python-dotenv is not installed or the file is absent.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Absolute path to code/ directory (where this file lives).
CODE_DIR: Path = Path(__file__).resolve().parent

# Repository root (one level above code/).
REPO_DIR: Path = CODE_DIR.parent

# Dataset directory.
DATASET_DIR: Path = REPO_DIR / "dataset"

# Dataset CSV paths.
CLAIMS_CSV: Path = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV: Path = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV: Path = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV: Path = DATASET_DIR / "evidence_requirements.csv"

# Images root.
IMAGES_DIR: Path = DATASET_DIR / "images"

# Primary prediction output.
OUTPUT_CSV: Path = REPO_DIR / "output.csv"

# Runtime cache directory for EvidenceFacts (created on first use).
CACHE_DIR: Path = CODE_DIR / ".cache"

# Structured log directory (created on first use).
LOGS_DIR: Path = CODE_DIR / ".logs"


# ---------------------------------------------------------------------------
# VLM configuration — Google Gemini sole provider
# ---------------------------------------------------------------------------

# Model identifier.  Override with the VLM_MODEL environment variable.
MODEL_NAME: str = os.environ.get("VLM_MODEL", "gemini-2.5-flash")

# Gemini API key — read from environment, never hardcoded.
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# Temperature for VLM calls.  0.0 for maximum determinism.
TEMPERATURE: float = 0.0

# Random seed passed to the API where supported.
SEED: int = 42


# ---------------------------------------------------------------------------
# Retry and timeout settings
# ---------------------------------------------------------------------------

# Maximum number of attempts per VLM call (1 initial + N-1 retries).
MAX_RETRIES: int = 3

# Base delay (seconds) for exponential backoff between retries.
RETRY_BASE_DELAY: float = 1.0

# Per-request timeout in seconds.
REQUEST_TIMEOUT: float = 60.0


# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------

# Maximum images sent in a single VLM call.  All images for one claim are
# sent together; this cap is a safety limit, not a batching target.
MAX_IMAGES_PER_CALL: int = 10

# Maximum image file size accepted for transport (bytes).
# Images exceeding this limit are flagged rather than sent.
MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Cache key components
#
# The EvidenceFacts cache key is the tuple:
#   (case_id, image_hash, claim_hash, model_name)
#
# image_hash  : SHA-256 of all image file bytes, sorted by image_id.
# claim_hash  : SHA-256 of the sanitized claim text.
# model_name  : value of MODEL_NAME at runtime.
#
# Decisions are NOT cached.  Only EvidenceFacts (observations) are cached.
# ---------------------------------------------------------------------------

CACHE_VERSION: str = "v1"  # Bump to invalidate all existing cache entries.
