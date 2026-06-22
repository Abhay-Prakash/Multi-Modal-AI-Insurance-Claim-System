"""
Retrieval module: deterministic, exact-match context retrieval.

Public interface
----------------
``retrieve_context(parsed, history, requirements)``
    Retrieve all context needed to process a single claim.  Returns a
    ``RetrievedContext`` with the matching ``UserHistory`` (or None) and the
    filtered ``EvidenceRequirement`` list.  The ``evidence_checklist`` field
    is left empty; it is populated later by ``grounding.build_evidence_checklist``.

``filter_requirements(requirements, claim_object)``
    Return the subset of requirements applicable to a given claim object type.
    Includes requirements with ``claim_object == 'all'``.

Design notes
------------
All retrieval is purely deterministic:
- User history lookup: O(1) dict lookup by ``user_id`` (exact match).
- Requirements filtering: linear scan, string equality, no fuzzy matching.
- No embeddings, no vector search, no AI of any kind.
- Order of filtered requirements follows the original CSV row order.

The ``evidence_checklist`` field on the returned ``RetrievedContext`` is
intentionally left empty here.  ``grounding.py`` constructs it from the
filtered requirements and the parsed claim atoms, so that checklist
construction can be isolated from data retrieval for testing.
"""

from __future__ import annotations

import logging
from typing import Optional

from schemas import (
    ClaimObject,
    EvidenceRequirement,
    ParsedClaim,
    RetrievedContext,
    UserHistory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def retrieve_context(
    parsed: ParsedClaim,
    history: dict[str, UserHistory],
    requirements: list[EvidenceRequirement],
) -> RetrievedContext:
    """Retrieve all context needed to process a single claim.

    Performs two deterministic lookups:
    1. Exact-match ``user_id`` lookup in the pre-loaded history dict.
    2. Filtering of requirements to those applicable to the claim's object type.

    The returned ``RetrievedContext.evidence_checklist`` is empty.
    Call ``grounding.build_evidence_checklist(parsed, context)`` afterwards
    to populate it.

    Args:
        parsed: The ``ParsedClaim`` for the claim being processed.
        history: Pre-loaded user history dict from ``ingestion.load_user_history``.
            Keyed by ``user_id`` for O(1) lookup.
        requirements: Full list of evidence requirements from
            ``ingestion.load_evidence_requirements``.

    Returns:
        ``RetrievedContext`` with:
        - ``user_history``: Matching ``UserHistory`` or ``None`` if not found.
        - ``applicable_requirements``: Filtered list for this claim's object type.
        - ``evidence_checklist``: Empty (populated by grounding.py).

    Examples:
        >>> context = retrieve_context(parsed, history, requirements)
        >>> context.user_history.rejected_claim
        0
        >>> len(context.applicable_requirements)
        5
    """
    user_id = parsed.claim_input.user_id
    claim_object = parsed.claim_input.claim_object

    user_history: Optional[UserHistory] = history.get(user_id)
    if user_history is None:
        logger.warning(
            "Row %d: no history found for user_id=%r; "
            "proceeding without history risk context.",
            parsed.claim_input.row_index,
            user_id,
        )

    applicable = filter_requirements(requirements, claim_object)

    logger.debug(
        "Row %d (user=%s): history=%s, applicable_requirements=%d",
        parsed.claim_input.row_index,
        user_id,
        "found" if user_history else "not found",
        len(applicable),
    )

    return RetrievedContext(
        user_history=user_history,
        applicable_requirements=applicable,
        # evidence_checklist left empty; populated by grounding.py
    )


def filter_requirements(
    requirements: list[EvidenceRequirement],
    claim_object: ClaimObject,
) -> list[EvidenceRequirement]:
    """Return requirements applicable to *claim_object*.

    Includes:
    - Requirements with ``claim_object == 'all'`` (universal requirements).
    - Requirements with ``claim_object`` matching the given ``ClaimObject``.

    Order follows the original CSV row order (deterministic).

    Args:
        requirements: Full list of all loaded evidence requirements.
        claim_object: The claim object type to filter for.

    Returns:
        Filtered list of ``EvidenceRequirement`` objects.

    Examples:
        >>> reqs = filter_requirements(all_requirements, ClaimObject.CAR)
        >>> all(r.claim_object in ("all", "car") for r in reqs)
        True
    """
    target = claim_object.value
    filtered = [
        r for r in requirements
        if r.claim_object == "all" or r.claim_object == target
    ]
    logger.debug(
        "filter_requirements(%s): %d/%d requirements match.",
        target,
        len(filtered),
        len(requirements),
    )
    return filtered
