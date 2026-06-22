"""
Parsing module: claim text parsing, atom extraction, and primary atom selection.

Public interface
----------------
``parse_claim(claim, sanitization)``
    Parse a ``ClaimInput`` (after sanitization) into a ``ParsedClaim``
    containing ``ConversationTurn[]``, ``ClaimAtom[]``, a ``primary_atom_index``,
    and a detected conversation language.

Internal helpers (exposed for unit testing)
-------------------------------------------
``split_turns(user_claim)``
    Split the pipe-delimited conversation string into a list of
    ``ConversationTurn`` objects.

``extract_atoms(customer_text, claim_object)``
    Extract ``ClaimAtom`` instances from concatenated customer turns using
    keyword pattern matching.  No VLM calls.

``select_primary_atom_index(atoms)``
    Choose the primary atom index by applying a deterministic priority
    ordering.

``detect_language(text)``
    Detect the primary language of a text string using script and
    vocabulary heuristics.  Returns an ISO 639-1 code.

Design notes
------------
Parsing is entirely heuristic and rule-based — no VLM calls are made.
The module must handle:
- Pipe-separated multi-turn conversations with mixed speaker labels.
- Multilingual conversations: Hindi/Hinglish, Spanish, Chinese, English.
- Code-switching (e.g. Hindi written in Latin script).
- Multi-damage claims: "front bumper and left headlight both damaged".
- Single-damage claims: "door has a deep dent".
- Vague or incomplete descriptions.

Conservative fallback:
    If no atoms can be extracted, a single fallback atom is constructed
    from the full customer text with ``issue_type_hint=None`` and
    ``object_part_hint=None``.  This guarantees the invariant:
    ``len(ParsedClaim.atoms) >= 1``.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from schemas import (
    ClaimAtom,
    ClaimInput,
    ClaimObject,
    ConversationTurn,
    IssueType,
    ObjectPart,
    ParsedClaim,
    SanitizationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Speaker role classification
# ---------------------------------------------------------------------------

# Maps lowercased speaker tokens to normalised roles.
_CUSTOMER_LABELS: frozenset[str] = frozenset({
    "customer", "client", "cliente", "user", "claimant",
    "me", "i",
})
_AGENT_LABELS: frozenset[str] = frozenset({
    "agent", "support", "soporte", "assistant", "rep", "representative",
    "advisor", "adjuster",
})

# Regex that matches "<Speaker>:" at the start of a turn segment.
_SPEAKER_RE: re.Pattern[str] = re.compile(
    r"^\s*(?P<speaker>[^:]{1,30}):\s*(?P<text>.*)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Issue type keyword index
# ---------------------------------------------------------------------------

# Maps each IssueType to a list of keyword patterns.  Longer / more specific
# phrases are listed first within each entry.
_ISSUE_KEYWORDS: dict[IssueType, tuple[str, ...]] = {
    IssueType.GLASS_SHATTER: (
        "shatter", "shattered", "smashed glass", "broken glass", "glass broke",
    ),
    IssueType.CRACK: (
        "crack", "cracked", "cracking", "fracture", "fractured", "hairline",
    ),
    IssueType.DENT: (
        "dent", "dented", "denting", "deformation", "deformed", "indentation",
        "buckled",
    ),
    IssueType.SCRATCH: (
        "scratch", "scratched", "scratching", "scuff", "scuffed", "gouge",
        "scrape", "scraped",
    ),
    IssueType.BROKEN_PART: (
        "broken", "broke", "break", "snapped", "snapped off", "snapped clean",
        "damaged beyond", "shattered part",
    ),
    IssueType.MISSING_PART: (
        "missing", "gone", "lost", "fell off", "detached", "not there",
        "came off",
    ),
    IssueType.TORN_PACKAGING: (
        "torn", "tore", "tear", "tearing", "ripped", "rip",
    ),
    IssueType.CRUSHED_PACKAGING: (
        "crushed", "crush", "squashed", "flat", "collapsed", "caved",
    ),
    IssueType.WATER_DAMAGE: (
        "water damage", "water damaged", "soaked", "wet", "moisture",
        "flood", "leaked", "leak",
    ),
    IssueType.STAIN: (
        "stain", "stained", "staining", "discolor", "discoloured",
        "discoloration", "marks",
    ),
}

# Build a flat list of (IssueType, compiled_pattern) sorted so that longer
# patterns are tried first (more specific wins).
_ISSUE_PATTERN_INDEX: list[tuple[IssueType, re.Pattern[str]]] = sorted(
    [
        (issue_type, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
        for issue_type, keywords in _ISSUE_KEYWORDS.items()
        for kw in keywords
    ],
    key=lambda pair: -len(pair[1].pattern),
)


# ---------------------------------------------------------------------------
# Object part keyword index
# ---------------------------------------------------------------------------

# Maps each ClaimObject to a dict of ObjectPart → keyword patterns.
_PART_KEYWORDS: dict[ClaimObject, dict[ObjectPart, tuple[str, ...]]] = {
    ClaimObject.CAR: {
        ObjectPart.FRONT_BUMPER: ("front bumper", "front-bumper", "bumper front"),
        ObjectPart.REAR_BUMPER: ("rear bumper", "back bumper", "rear-bumper"),
        ObjectPart.WINDSHIELD: ("windshield", "windscreen", "front glass", "front window"),
        ObjectPart.HEADLIGHT: ("headlight", "head light", "front light", "headlamp"),
        ObjectPart.TAILLIGHT: ("taillight", "tail light", "rear light", "brake light"),
        ObjectPart.SIDE_MIRROR: ("side mirror", "wing mirror", "mirror"),
        ObjectPart.HOOD: ("hood", "bonnet"),
        ObjectPart.DOOR: ("door", "car door"),
        ObjectPart.FENDER: ("fender", "quarter panel", "wheel arch"),
        ObjectPart.QUARTER_PANEL: ("quarter panel", "rear panel"),
        ObjectPart.BODY: ("body", "bodywork", "panel", "exterior"),
    },
    ClaimObject.LAPTOP: {
        ObjectPart.SCREEN: ("screen", "display", "lcd", "monitor", "lid screen"),
        ObjectPart.KEYBOARD: ("keyboard", "keys", "key"),
        ObjectPart.TRACKPAD: ("trackpad", "touchpad", "track pad"),
        ObjectPart.HINGE: ("hinge", "hinge joint"),
        ObjectPart.LID: ("lid", "top cover", "top panel"),
        ObjectPart.CORNER: ("corner", "edge"),
        ObjectPart.PORT: ("port", "usb", "hdmi", "charging port", "connector"),
        ObjectPart.BASE: ("base", "bottom", "underside"),
        ObjectPart.BODY: ("body", "casing", "chassis", "shell"),
    },
    ClaimObject.PACKAGE: {
        ObjectPart.BOX: ("box", "carton", "packaging box"),
        ObjectPart.PACKAGE_CORNER: ("corner", "package corner", "box corner"),
        ObjectPart.PACKAGE_SIDE: ("side", "package side", "side panel"),
        ObjectPart.SEAL: ("seal", "tape", "sealing", "adhesive"),
        ObjectPart.LABEL: ("label", "shipping label", "address label"),
        ObjectPart.CONTENTS: ("contents", "inside", "inner", "content"),
        ObjectPart.ITEM: ("item", "product", "goods"),
    },
}

# Build per-object flat lists of (ObjectPart, compiled_pattern), longer first.
_PART_PATTERN_INDEX: dict[ClaimObject, list[tuple[ObjectPart, re.Pattern[str]]]] = {
    obj: sorted(
        [
            (part, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
            for part, keywords in parts.items()
            for kw in keywords
        ],
        key=lambda pair: -len(pair[1].pattern),
    )
    for obj, parts in _PART_KEYWORDS.items()
}


# ---------------------------------------------------------------------------
# Language detection heuristics
# ---------------------------------------------------------------------------

# Devanagari Unicode block: U+0900–U+097F
_DEVANAGARI_RE: re.Pattern[str] = re.compile(r"[\u0900-\u097F]")

# CJK Unified Ideographs and common CJK blocks
_CJK_RE: re.Pattern[str] = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")

# Common Spanish-specific characters and words (not present in English)
_SPANISH_RE: re.Pattern[str] = re.compile(
    r"[áéíóúüñ¿¡]"
    r"|\b(el|la|los|las|un|una|del|por|para|con|que|esta|este|fue|hay|no)\b",
    re.IGNORECASE,
)

# Hinglish (Hindi written in Latin script) indicators
_HINGLISH_WORDS: frozenset[str] = frozenset({
    "gaya", "hua", "nahi", "kar", "kiya", "hai", "tha", "thi",
    "aur", "mera", "meri", "mujhe", "yeh", "woh", "kuch",
})
_HINGLISH_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(_HINGLISH_WORDS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Multi-damage segmentation
# ---------------------------------------------------------------------------

# Connectors that may separate two distinct damage claims in a single sentence.
_MULTI_DAMAGE_SPLIT_RE: re.Pattern[str] = re.compile(
    r"\s+(?:and\s+(?:also\s+)?|also\s+|additionally\s+|plus\s+|as\s+well\s+as\s+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def parse_claim(
    claim: ClaimInput,
    sanitization: SanitizationResult,
) -> ParsedClaim:
    """Parse a claim into a structured ``ParsedClaim``.

    Orchestrates the full parsing pipeline:
    1. Split the conversation into turns.
    2. Extract customer-only text.
    3. Extract ``ClaimAtom[]`` via keyword matching.
    4. Select the primary atom.
    5. Detect the conversation language.

    Args:
        claim: The raw ``ClaimInput`` as loaded by ``ingestion.py``.
        sanitization: The ``SanitizationResult`` from ``sanitization.py``.
            The ``sanitized_claim`` text is used for turn splitting so that
            any injection annotations are preserved in the turn structure.

    Returns:
        ``ParsedClaim`` with at least one atom guaranteed.

    Examples:
        >>> from schemas import ClaimInput, ClaimObject
        >>> from sanitization import sanitize
        >>> claim = ClaimInput(
        ...     user_id="u1",
        ...     image_paths_raw="",
        ...     user_claim="Customer: Door dent. | Agent: Noted.",
        ...     claim_object=ClaimObject.CAR,
        ...     row_index=0,
        ... )
        >>> result = parse_claim(claim, sanitize(claim))
        >>> result.atoms[0].issue_type_hint
        <IssueType.DENT: 'dent'>
    """
    # Use the sanitized text for turn splitting so injection annotations
    # appear in the turn structure and can be logged.
    turns = split_turns(sanitization.sanitized_claim)

    customer_text = _extract_customer_text(turns)
    language = detect_language(customer_text or claim.user_claim)
    atoms = extract_atoms(customer_text, claim.claim_object)

    # Conservative fallback: always produce at least one atom.
    if not atoms:
        atoms = [
            ClaimAtom(
                described_issue="unspecified damage",
                described_part="unknown part",
                issue_type_hint=None,
                object_part_hint=None,
            )
        ]
        logger.debug(
            "Row %d: no atoms extracted; using fallback atom.",
            claim.row_index,
        )

    primary_index = select_primary_atom_index(atoms)

    logger.debug(
        "Row %d: %d turn(s), %d atom(s), primary=%d, lang=%s",
        claim.row_index,
        len(turns),
        len(atoms),
        primary_index,
        language,
    )

    return ParsedClaim(
        claim_input=claim,
        raw_text=claim.user_claim,
        conversation_turns=turns,
        atoms=atoms,
        primary_atom_index=primary_index,
        conversation_language=language,
        sanitization=sanitization,
    )


# ---------------------------------------------------------------------------
# Turn splitting
# ---------------------------------------------------------------------------


def split_turns(user_claim: str) -> list[ConversationTurn]:
    """Split a pipe-delimited conversation string into ``ConversationTurn``s.

    Args:
        user_claim: Pipe-separated conversation string, e.g.::

            "Customer: I have a dent. | Agent: Where? | Customer: Door."

        May use ``|`` with or without surrounding whitespace.

    Returns:
        List of ``ConversationTurn`` objects in order.  Turns with no
        recognisable speaker prefix are assigned ``speaker_role='unknown'``.

    Examples:
        >>> turns = split_turns("Customer: Dent on door. | Agent: Noted.")
        >>> len(turns)
        2
        >>> turns[0].speaker_role
        'customer'
    """
    raw_segments = [seg.strip() for seg in user_claim.split("|") if seg.strip()]
    turns: list[ConversationTurn] = []

    for i, segment in enumerate(raw_segments):
        m = _SPEAKER_RE.match(segment)
        if m:
            speaker_raw = m.group("speaker").strip()
            text = m.group("text").strip()
        else:
            speaker_raw = "unknown"
            text = segment.strip()

        speaker_role = _classify_speaker(speaker_raw)
        turns.append(
            ConversationTurn(
                turn_index=i,
                speaker_raw=speaker_raw,
                speaker_role=speaker_role,
                text=text,
            )
        )

    return turns


def _classify_speaker(speaker_raw: str) -> str:
    """Return 'customer', 'agent', or 'unknown' for a raw speaker label."""
    normalised = speaker_raw.strip().lower()
    if normalised in _CUSTOMER_LABELS:
        return "customer"
    if normalised in _AGENT_LABELS:
        return "agent"
    # Partial matches for edge cases.
    if any(label in normalised for label in _CUSTOMER_LABELS):
        return "customer"
    if any(label in normalised for label in _AGENT_LABELS):
        return "agent"
    return "unknown"


def _extract_customer_text(turns: list[ConversationTurn]) -> str:
    """Concatenate all customer turn texts into a single string."""
    return " ".join(t.text for t in turns if t.speaker_role == "customer")


# ---------------------------------------------------------------------------
# Atom extraction
# ---------------------------------------------------------------------------


def extract_atoms(
    customer_text: str,
    claim_object: ClaimObject,
) -> list[ClaimAtom]:
    """Extract ``ClaimAtom`` instances from customer turn text.

    Uses keyword pattern matching to identify part mentions and issue
    type mentions.  Handles multi-damage claims by segmenting on damage
    connectors ("and", "also", etc.) and pairing the nearest part mention
    to each issue mention.

    Args:
        customer_text: Concatenated text from all customer turns.
        claim_object: The claim object type (car, laptop, package).

    Returns:
        List of ``ClaimAtom`` objects.  May be empty if no patterns match
        (caller must apply the fallback).

    Examples:
        >>> atoms = extract_atoms(
        ...     "front bumper looks damaged and left headlight also affected",
        ...     ClaimObject.CAR,
        ... )
        >>> len(atoms)
        2
        >>> atoms[0].object_part_hint
        <ObjectPart.FRONT_BUMPER: 'front_bumper'>
    """
    if not customer_text.strip():
        return []

    # Find all part mentions with their positions.
    part_mentions = _find_part_mentions(customer_text, claim_object)

    # Find all issue type mentions with their positions.
    issue_mentions = _find_issue_mentions(customer_text)

    if not part_mentions and not issue_mentions:
        return []

    # Attempt to pair parts with issues by proximity.
    atoms = _pair_mentions(customer_text, part_mentions, issue_mentions, claim_object)

    return atoms


def _find_part_mentions(
    text: str,
    claim_object: ClaimObject,
) -> list[tuple[int, int, ObjectPart]]:
    """Find all part keyword matches in *text*.

    Returns:
        List of ``(start, end, ObjectPart)`` tuples, ordered by position.
    """
    patterns = _PART_PATTERN_INDEX.get(claim_object, [])
    found: list[tuple[int, int, ObjectPart]] = []
    covered: list[tuple[int, int]] = []

    for part, pattern in patterns:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if not _overlaps(span, covered):
                found.append((m.start(), m.end(), part))
                covered.append(span)

    found.sort(key=lambda x: x[0])
    return found


def _find_issue_mentions(
    text: str,
) -> list[tuple[int, int, IssueType]]:
    """Find all issue type keyword matches in *text*.

    Returns:
        List of ``(start, end, IssueType)`` tuples, ordered by position.
    """
    found: list[tuple[int, int, IssueType]] = []
    covered: list[tuple[int, int]] = []

    for issue_type, pattern in _ISSUE_PATTERN_INDEX:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if not _overlaps(span, covered):
                found.append((m.start(), m.end(), issue_type))
                covered.append(span)

    found.sort(key=lambda x: x[0])
    return found


def _pair_mentions(
    text: str,
    part_mentions: list[tuple[int, int, ObjectPart]],
    issue_mentions: list[tuple[int, int, IssueType]],
    claim_object: ClaimObject,
) -> list[ClaimAtom]:
    """Pair part mentions with issue mentions to produce ClaimAtoms.

    Strategy:
    1. If there are multiple distinct parts, try to split the text on
       damage connectors and process each segment independently.
    2. Otherwise, pair each part with the nearest issue, or create a
       single atom from whatever is available.
    """
    # Try multi-segment split when multiple distinct parts are found.
    if len(part_mentions) > 1:
        segments = _MULTI_DAMAGE_SPLIT_RE.split(text)
        if len(segments) > 1:
            atoms: list[ClaimAtom] = []
            for seg in segments:
                seg_parts = _find_part_mentions(seg, claim_object)
                seg_issues = _find_issue_mentions(seg)
                atom = _make_atom(seg.strip(), seg_parts, seg_issues)
                # Discard atoms with no extractable hints — these are
                # connector-phrase artefacts (e.g. the bare "also" segment).
                if atom is not None and (
                    atom.issue_type_hint is not None
                    or atom.object_part_hint is not None
                ):
                    atoms.append(atom)
            if atoms:
                return atoms

    # Single-segment fallback: one atom from all found mentions.
    atom = _make_atom(text.strip(), part_mentions, issue_mentions)
    return [atom] if atom is not None else []


def _make_atom(
    segment_text: str,
    part_mentions: list[tuple[int, int, ObjectPart]],
    issue_mentions: list[tuple[int, int, IssueType]],
) -> Optional[ClaimAtom]:
    """Construct a single ``ClaimAtom`` from the best-matching mentions.

    Picks the first part mention and the first issue mention by position.
    Returns ``None`` only when the segment text is empty.
    """
    if not segment_text:
        return None

    best_part: Optional[ObjectPart] = part_mentions[0][2] if part_mentions else None
    best_issue: Optional[IssueType] = issue_mentions[0][2] if issue_mentions else None

    described_part = best_part.value if best_part else "unknown part"
    described_issue = best_issue.value if best_issue else "unspecified damage"

    return ClaimAtom(
        described_issue=described_issue,
        described_part=described_part,
        issue_type_hint=best_issue,
        object_part_hint=best_part,
    )


def _overlaps(span: tuple[int, int], covered: list[tuple[int, int]]) -> bool:
    """Return True if *span* overlaps with any interval in *covered*."""
    s, e = span
    return any(s < ce and e > cs for cs, ce in covered)


# ---------------------------------------------------------------------------
# Primary atom selection
# ---------------------------------------------------------------------------


def select_primary_atom_index(atoms: list[ClaimAtom]) -> int:
    """Choose the primary atom index using a deterministic priority ordering.

    Priority (descending):
    1. Atom with both ``issue_type_hint`` and ``object_part_hint`` set.
    2. Atom with ``issue_type_hint`` set.
    3. Atom with ``object_part_hint`` set.
    4. First atom (index 0) as tiebreaker.

    Args:
        atoms: Non-empty list of ``ClaimAtom`` objects.

    Returns:
        Integer index of the primary atom.

    Raises:
        ValueError: If *atoms* is empty.

    Examples:
        >>> atoms = [
        ...     ClaimAtom(described_issue="x", described_part="y"),
        ...     ClaimAtom(
        ...         described_issue="dent",
        ...         described_part="door",
        ...         issue_type_hint=IssueType.DENT,
        ...         object_part_hint=ObjectPart.DOOR,
        ...     ),
        ... ]
        >>> select_primary_atom_index(atoms)
        1
    """
    if not atoms:
        raise ValueError("atoms list must not be empty")

    def _priority(atom: ClaimAtom) -> int:
        has_issue = atom.issue_type_hint is not None
        has_part = atom.object_part_hint is not None
        if has_issue and has_part:
            return 3
        if has_issue:
            return 2
        if has_part:
            return 1
        return 0

    best_index = 0
    best_priority = _priority(atoms[0])

    for i, atom in enumerate(atoms[1:], start=1):
        p = _priority(atom)
        if p > best_priority:
            best_priority = p
            best_index = i

    return best_index


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str) -> str:
    """Detect the primary language of *text* using script heuristics.

    Heuristics applied in priority order:
    1. Devanagari script characters → 'hi' (Hindi).
    2. CJK ideograph characters → 'zh' (Chinese).
    3. Spanish-specific characters or common words → 'es'.
    4. Hinglish markers (Hindi romanised) → 'hi'.
    5. Default → 'en'.

    Args:
        text: The conversation text to analyse.

    Returns:
        ISO 639-1 language code string.

    Examples:
        >>> detect_language("Customer: Screen cracked.")
        'en'
        >>> detect_language("Cliente: La pantalla está rota.")
        'es'
    """
    if not text:
        return "en"

    if _DEVANAGARI_RE.search(text):
        return "hi"

    if _CJK_RE.search(text):
        return "zh"

    if _SPANISH_RE.search(text):
        return "es"

    if _HINGLISH_RE.search(text):
        return "hi"

    return "en"
