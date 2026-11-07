"""NAFDAC registered-products CSV ingestion.

SAMPLE DATA, NOT AUTHORITATIVE: ``data/nafdac/`` ships a handful of
illustrative rows only. Real registries must come from an official
NAFDAC product-registry CSV export (there is no public bulk API).
Rows are validated at the trust boundary: missing required columns
raise ``ValueError``; rows without a product name are skipped.
"""

import csv
import re
import sys
from pathlib import Path

_DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "nafdac"
    / "sample_registered_products.csv"
)
_COLUMNS = (
    "product_name",
    "active_ingredient",
    "manufacturer",
    "registration_number",
    "status",
)
_STRENGTH = re.compile(r"\d+(\.\d+)?\s*(mg|mcg|g|ml|iu)\b", re.IGNORECASE)


def _norm(value: str) -> str:
    """Normalize a term the way the coding stage compares aliases."""
    return " ".join(value.split()).casefold()


def load_products(path: str | Path) -> list[dict[str, str]]:
    """Load a NAFDAC registered-products CSV export.

    Header names are matched liberally (case- and space-insensitive)
    against ``product_name``, ``active_ingredient``,
    ``manufacturer``, ``registration_number``, ``status``. The
    product name is re-keyed to ``name`` and lowercased.

    Parameters
    ----------
    path : str or Path
        CSV file path.

    Returns
    -------
    list of dict
        One normalized dict per usable row; rows without a
        ``product_name`` are skipped.

    Raises
    ------
    ValueError
        When any required column is absent from the header.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = {
            (name or "").strip().casefold().replace(" ", "_"): name
            for name in (reader.fieldnames or [])
        }
        missing = [column for column in _COLUMNS if column not in header]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        products: list[dict[str, str]] = []
        for raw in reader:
            row = {
                column: (raw.get(header[column]) or "").strip()
                for column in _COLUMNS
            }
            if not row["product_name"]:
                continue
            row["name"] = row.pop("product_name").casefold()
            products.append(row)
    return products


def extend_aliases(products: list[dict[str, str]]) -> dict[str, str]:
    """Build coding-stage alias pairs from loaded NAFDAC products.

    Produces ``{"<normalized variant>": "<product name>"}`` entries in
    the same shape as ``extractor.adapt`` aliases: keys are normalized
    (whitespace-collapsed, casefolded) ingredient names and
    strength-stripped product-name variants; values are the product
    names. Returned only — nothing is written to
    ``customer_aliases.json``; the caller decides what to learn from.

    Parameters
    ----------
    products : list of dict
        Output of :func:`load_products`.

    Returns
    -------
    dict
        Alias pairs suitable for the coding stage's alias convention.
    """
    aliases: dict[str, str] = {}
    for product in products:
        name = product.get("name", "").strip()
        if not name:
            continue
        ingredient = product.get("active_ingredient", "").strip()
        if ingredient:
            aliases.setdefault(_norm(ingredient), name)
        bare = _norm(_STRENGTH.sub("", name))
        if bare and bare != _norm(name):
            aliases.setdefault(bare, name)
    return aliases


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_CSV
    try:
        loaded = load_products(csv_path)
    except (OSError, ValueError) as exc:
        print(f"nafdac: could not load {csv_path}: {exc}")
        raise SystemExit(1) from exc
    print(f"nafdac: {len(loaded)} products from {csv_path}")
    for product in loaded[:3]:
        print(f"  {product}")
