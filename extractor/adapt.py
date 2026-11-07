"""Learn customer value aliases from reviewer corrections (#1).

Reviewer edits captured by the ``/training`` endpoint are distilled
into ``extractor/customer_aliases.json``: a flat map from a normalized
mis-extracted draft value to the reviewed (canonical) value. The
coding stage consumes the file to auto-apply recurring corrections.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from bench.harness import normalize_value

ALIASES_PATH = (
    Path(__file__).resolve().parents[1] / "extractor" / "customer_aliases.json"
)

_LEARNED_FIELDS = frozenset({"product", "event"})
_CACHE: dict[str, str] | None = None


def build_aliases(
    records: list[dict[str, Any]],
    out_path: str = "extractor/customer_aliases.json",
) -> dict[str, str]:
    """Build value aliases from reviewer correction records.

    Parameters
    ----------
    records : list[dict]
        Training exports shaped like the ``/training`` endpoint:
        ``[{"case_id", "field", "draft_value", "final_value", "ts"}]``.
    out_path : str
        Where to write the alias JSON (default
        ``extractor/customer_aliases.json``).

    Returns
    -------
    dict
        ``{"<normalized draft>": "<final value>"}`` pairs for
        product/event fields where the normalized draft differs from
        the normalized final. The reviewed value is canonical; empty
        and identical pairs are skipped.
    """
    aliases: dict[str, str] = {}
    for record in records:
        if record.get("field") not in _LEARNED_FIELDS:
            continue
        draft = str(record.get("draft_value", ""))
        final = str(record.get("final_value", ""))
        if not draft.strip() or not final.strip():
            continue
        if normalize_value(draft) == normalize_value(final):
            continue
        aliases[normalize_value(draft)] = final
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(aliases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("aliases: {} entries -> {}", len(aliases), path)
    return aliases


def load_aliases() -> dict[str, str]:
    """Load ``customer_aliases.json`` lazily; {} when missing."""
    global _CACHE
    if _CACHE is None:
        if ALIASES_PATH.exists():
            _CACHE = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
        else:
            _CACHE = {}
    return _CACHE
