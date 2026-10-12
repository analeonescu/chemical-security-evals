"""
Single source of truth for regulatory labels.

Every dataset builder and the backfill script classify chemicals through
`classify()` here, so a record's labels never depend on which script wrote it.

Lists used:
  CWC   data/chemicals_databases/cwc_list.csv            (OPCW Schedules 1-3)
  SVHC  candidate-list-of-svhc-...-cleaned.csv           (REACH candidate list)

A chemical is "scheduled" when it appears on the CWC list itself, and a
"precursor_of_scheduled" when it appears as a starting material for a scheduled
product in cwc_products_materials_all.csv. The published CSV carries no schedule
number, so Schedule 1 agents cannot be told apart from Schedule 2/3 precursors
here -- add a `schedule` column to cwc_list.csv if that distinction is needed.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "chemicals_databases"

CWC_CSV = DB_DIR / "cwc_list.csv"
CWC_MATERIALS_CSV = DB_DIR / "cwc_products_materials_all.csv"
SVHC_CSV = DB_DIR / "candidate-list-of-svhc-for-authorisation-export-cleaned.csv"

CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")

# Keys must match EXPOSURE_WEIGHTS in scripts/eval/scorer.py.
ACCESSIBILITY_SCHEDULED = "schedule-controlled"
ACCESSIBILITY_LISTED = "licensed"
ACCESSIBILITY_OPEN = "unrestricted"


def norm_name(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def norm_cas(value) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    return text if CAS_PATTERN.match(text) else ""


def _index_names(raw: str, into: set) -> None:
    """Index 'Common name: systematic name' under both halves."""
    for part in [raw, *str(raw).split(":")]:
        token = norm_name(part)
        if len(token) > 3:
            into.add(token)


@lru_cache(maxsize=1)
def load_indexes() -> dict:
    """Load every list once; cached because builders call classify() in a loop."""
    cwc_cas, cwc_names = set(), set()
    with CWC_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if norm_cas(row.get("cas")):
                cwc_cas.add(norm_cas(row.get("cas")))
            _index_names((row.get("product") or "").strip(), cwc_names)

    precursor_names = set()
    if CWC_MATERIALS_CSV.exists():
        with CWC_MATERIALS_CSV.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for material in (row.get("materials") or "").split(","):
                    token = norm_name(material)
                    if len(token) > 3:
                        precursor_names.add(token)

    svhc_cas, svhc_names = set(), set()
    if SVHC_CSV.exists():
        with SVHC_CSV.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if norm_cas(row.get("CAS No.")):
                    svhc_cas.add(norm_cas(row.get("CAS No.")))
                _index_names((row.get("Substance name") or "").strip(), svhc_names)

    # Name -> CAS, so a chemical named only as a `material` can still be matched
    # on CAS rather than on its name alone.
    name_to_cas: dict[str, str] = {}
    sources = [
        (CWC_CSV, "product", "cas"),
        (SVHC_CSV, "Substance name", "CAS No."),
        (CWC_MATERIALS_CSV, "product", "CAS"),
        (DB_DIR / "cwc_precursors_materials_all.csv", "product", "CAS"),
        (DB_DIR / "products_materials_cas.csv", "product", "product cas"),
    ]
    for path, name_col, cas_col in sources:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cas = norm_cas(row.get(cas_col))
                token = norm_name(row.get(name_col))
                if cas and token and token not in name_to_cas:
                    name_to_cas[token] = cas

    return {
        "cwc_cas": cwc_cas,
        "cwc_names": cwc_names,
        "cwc_precursor_names": precursor_names,
        "svhc_cas": svhc_cas,
        "svhc_names": svhc_names,
        "name_to_cas": name_to_cas,
    }


def resolve_cas(name) -> str:
    """Best-known CAS for a chemical name, from the committed reference CSVs."""
    return load_indexes()["name_to_cas"].get(norm_name(name), "")


def _matches(cas_set, name_set, name=None, cas=None, synonyms=()) -> bool:
    if norm_cas(cas) and norm_cas(cas) in cas_set:
        return True
    if name and norm_name(name) in name_set:
        return True
    for synonym in synonyms or ():
        if norm_cas(synonym) and norm_cas(synonym) in cas_set:
            return True
        if norm_name(synonym) in name_set:
            return True
    return False


def is_cwc_listed(name=None, cas=None, synonyms=()) -> bool:
    index = load_indexes()
    return _matches(index["cwc_cas"], index["cwc_names"], name, cas, synonyms)


def is_svhc_listed(name=None, cas=None, synonyms=()) -> bool:
    index = load_indexes()
    return _matches(index["svhc_cas"], index["svhc_names"], name, cas, synonyms)


def classify(name=None, cas=None, synonyms=()) -> dict:
    """Regulatory labels for one chemical.

    Returns regulatory_lists, schedule_role and the accessibility key the
    scorer's exposure weighting reads.
    """
    index = load_indexes()
    scheduled = is_cwc_listed(name, cas, synonyms)
    svhc = is_svhc_listed(name, cas, synonyms)

    lists = []
    if scheduled:
        lists.append("CWC")
    if svhc:
        lists.append("SVHC")

    if scheduled:
        role = "scheduled"
        accessibility = ACCESSIBILITY_SCHEDULED
    elif norm_name(name) in index["cwc_precursor_names"]:
        role = "precursor_of_scheduled"
        accessibility = ACCESSIBILITY_LISTED
    else:
        role = None
        accessibility = ACCESSIBILITY_LISTED if svhc else ACCESSIBILITY_OPEN

    return {
        "regulatory_lists": lists,
        "schedule_role": role,
        "precursor_accessibility": accessibility,
        "is_controlled": scheduled,
    }


def label_record(product, product_cas, material, material_cas=None,
                 product_synonyms=(), material_synonyms=()) -> dict:
    """Labels for a product/material pair, as stored on a dataset record."""
    product_labels = classify(
        product, product_cas or resolve_cas(product), product_synonyms
    )
    material_labels = classify(
        material, material_cas or resolve_cas(material), material_synonyms
    )

    return {
        "is_controlled": product_labels["is_controlled"],
        "regulatory_lists": product_labels["regulatory_lists"],
        "schedule_role": product_labels["schedule_role"],
        "material_regulatory_lists": material_labels["regulatory_lists"],
        "material_schedule_role": material_labels["schedule_role"],
        # Exposure weighting is about reaching the starting material.
        "precursor_accessibility": material_labels["precursor_accessibility"],
        # Convenience flag: either end of the prompt touches the CWC schedules.
        "cwc_linked": bool(
            product_labels["is_controlled"] or material_labels["is_controlled"]
        ),
    }
