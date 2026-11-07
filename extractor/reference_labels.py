"""Curated reference table of labelled (expected) events per drug.

CURATED, REVIEWER-OWNED data: pharmacology-based expectations used by
``extractor.reference.refine_expectedness``. Keys are normalized
canonical drug terms from ``extractor.dictionary.DRUGS``; values are
event codes from ``EVENTS``. Only defensible label reactions are listed
(e.g. hypoglycaemia is uncommon but labelled for metformin).
"""

from extractor.dictionary import EVENTS


def _codes(*terms: str) -> list[str]:
    """Look up event codes for curated event term names."""
    return [EVENTS[term] for term in terms]


PRODUCT_KEY: dict[str, list[str]] = {
    "paracetamol": _codes("fever", "rash"),
    "amoxicillin": _codes(
        "rash",
        "pruritus",
        "urticaria",
        "nausea",
        "diarrhoea",
        "anaphylaxis",
    ),
    "ibuprofen": _codes(
        "dyspepsia",
        "nausea",
        "gastritis",
        "gastrointestinal haemorrhage",
        "rash",
        "dizziness",
        "oedema",
        "hypertension",
        "renal impairment",
    ),
    "aspirin": _codes(
        "dyspepsia",
        "nausea",
        "gastrointestinal haemorrhage",
        "rash",
        "tinnitus",
    ),
    "metformin": _codes(
        # hypoglycaemia: uncommon but labelled for metformin
        "nausea",
        "vomiting",
        "diarrhoea",
        "abdominal pain",
        "lactic acidosis",
        "hypoglycaemia",
    ),
    "lisinopril": _codes(
        "hypotension",
        "cough",
        "hyperkalaemia",
        "angio-oedema",
        "dizziness",
        "renal impairment",
    ),
    "ciprofloxacin": _codes(
        "nausea",
        "diarrhoea",
        "headache",
        "tendinopathy",
        "rash",
        "insomnia",
    ),
    "artesunate": _codes(
        "nausea",
        "vomiting",
        "dizziness",
        "headache",
        "pruritus",
    ),
    "artemether/lumefantrine": _codes(
        "dizziness",
        "nausea",
        "vomiting",
        "headache",
        "pruritus",
        "insomnia",
    ),
    "oral rehydration salts": _codes("nausea", "vomiting"),
    "combined oral contraceptive": _codes(
        "nausea",
        "headache",
        "hypertension",
        "thrombosis",
        "oedema",
    ),
    "tramadol": _codes(
        "nausea",
        "vomiting",
        "dizziness",
        "constipation",
        "headache",
    ),
    "diclofenac": _codes(
        "dyspepsia",
        "nausea",
        "gastritis",
        "gastrointestinal haemorrhage",
        "rash",
        "hypertension",
    ),
    "ampicillin": _codes(
        "rash",
        "urticaria",
        "diarrhoea",
        "nausea",
        "anaphylaxis",
    ),
    "ceftriaxone": _codes("rash", "urticaria", "diarrhoea", "pruritus"),
    "quinine": _codes(
        "tinnitus",
        "dizziness",
        "headache",
        "nausea",
        "hypoglycaemia",
    ),
    "chloroquine": _codes("pruritus", "nausea", "headache", "dizziness"),
    "azithromycin": _codes("nausea", "diarrhoea", "abdominal pain", "rash"),
    "omeprazole": _codes("headache", "diarrhoea", "constipation", "nausea"),
    "amlodipine": _codes("oedema", "headache", "dizziness", "nausea"),
    "glibenclamide": _codes("hypoglycaemia", "nausea", "dizziness"),
    "hydrochlorothiazide": _codes(
        "hypokalaemia",
        "hypotension",
        "dizziness",
        "nausea",
    ),
    "ferrous sulphate": _codes(
        "constipation",
        "nausea",
        "diarrhoea",
        "abdominal pain",
        "vomiting",
    ),
    "salbutamol": _codes("arrhythmia", "insomnia"),
    "prednisolone": _codes("hyperglycaemia", "insomnia", "hypertension"),
    "warfarin": _codes("gastrointestinal haemorrhage", "anaemia"),
    "insulin": _codes("hypoglycaemia"),
    "sulfadoxine/pyrimethamine": _codes(
        "rash",
        "pruritus",
        "nausea",
        "vomiting",
        "stevens-johnson syndrome",
    ),
}
