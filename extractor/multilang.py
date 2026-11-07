"""Multilingual label patterns merged into the heuristic extractor.

``EXTRA_LABELLED`` maps a FIELD_KEYS entry to additional compiled regexes
with a named group ``v`` — they are tried after the English label patterns.
``EXTRA_SERIOUS_TERMS`` is a regex alternation body (no group) of
seriousness keywords in other languages.
"""

import re

EXTRA_LABELLED: dict[str, tuple[re.Pattern[str], ...]] = {}

EXTRA_SERIOUS_TERMS = ""
