"""
Validate CAS numbers in an existing chemical safety dataset using PubChem.

Input:
    chemsafety_reconstructed.json

Output:
    cas_validation_report.json

Checks:
    - Is CAS valid?
    - Does PubChem recognise the CAS?
    - What compound does this CAS correspond to?
    - Does the PubChem identity match the product name?
"""

import json
import time
import re
import urllib.request
import urllib.parse
from pathlib import Path


PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

INPUT_FILE = "cwc_precursors_chemsafety_reconstructed.json"
OUTPUT_FILE = "cwc_precursors_cas_validation_report.json"


# -----------------------------
# HTTP helper
# -----------------------------

def http_get_json(url):

    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return json.loads(response.read().decode())

    except Exception as e:
        print(f"Request failed: {e}")
        return None



# -----------------------------
# CAS utilities
# -----------------------------

def clean_cas(cas):

    """
    Convert pandas NaN / None / empty values
    into a clean CAS string or None.
    """

    if cas is None:
        return None

    if isinstance(cas, float):
        return None

    cas = str(cas).strip()

    if cas.lower() in [
        "",
        "nan",
        "none",
        "-"
    ]:
        return None

    return cas



def valid_cas_format(cas):

    """
    Basic CAS format:
    XXXXXXX-XX-X
    """

    if not cas:
        return False

    return bool(
        re.match(
            r"^\d{2,7}-\d{2}-\d$",
            cas
        )
    )



# -----------------------------
# PubChem
# -----------------------------

def get_cid_from_cas(cas):

    if not cas:
        return None


    url = (
        f"{PUG_BASE}/compound/name/"
        f"{urllib.parse.quote(cas)}/cids/JSON"
    )


    data = http_get_json(url)


    if not data:
        return None


    return (
        data
        .get("IdentifierList", {})
        .get("CID", [None])[0]
    )



def get_synonyms(cid):

    url = (
        f"{PUG_BASE}/compound/cid/"
        f"{cid}/synonyms/JSON"
    )


    data = http_get_json(url)


    if not data:
        return []


    return (
        data
        .get("InformationList", {})
        .get("Information", [{}])[0]
        .get("Synonym", [])
    )



# -----------------------------
# Dataset cleaning
# -----------------------------

def looks_like_chemical_name(name):

    """
    Remove regulatory descriptions accidentally
    imported as chemical names.
    """

    if not name:
        return False


    # very long paragraphs are not chemicals
    if len(name) > 200:
        return False


    return True



def name_matches_product(product, synonyms):

    """
    Loose identity check.
    """

    if not product:
        return []


    product = product.lower()

    matches = []


    for syn in synonyms:

        if product in syn.lower():
            matches.append(syn)


        # check major words
        words = [
            w for w in product.split()
            if len(w) > 4
        ]


        if words:
            overlap = sum(
                w in syn.lower()
                for w in words
            )

            if overlap >= 2:
                matches.append(syn)


    return list(set(matches))[:5]



# -----------------------------
# Validation
# -----------------------------

def validate_record(record, idx):

    product = record.get("product")
    cas = clean_cas(record.get("cas"))


    print("\n")
    print(f"[{idx}] {product}")
    print(f"CAS: {cas}")


    result = {
        **record,

        "cas": cas,

        "cas_valid": False,

        "cas_pubchem_cid": None,

        "cas_synonyms": [],

        "name_matches": [],

        "validation_status": None
    }


    if not cas:

        result["validation_status"] = (
            "No CAS provided"
        )

        return result



    if not valid_cas_format(cas):

        result["validation_status"] = (
            "Invalid CAS format"
        )

        return result



    cid = get_cid_from_cas(cas)


    time.sleep(0.3)



    if cid is None:

        result["validation_status"] = (
            "CAS not found in PubChem"
        )

        return result



    print(
        f"CID: {cid}"
    )


    synonyms = get_synonyms(cid)


    time.sleep(0.3)



    result["cas_valid"] = True

    result["cas_pubchem_cid"] = cid

    result["cas_synonyms"] = synonyms[:30]


    matches = name_matches_product(
        product,
        synonyms
    )


    result["name_matches"] = matches



    if matches:

        result["validation_status"] = (
            "CAS and product name match"
        )

    else:

        result["validation_status"] = (
            "CAS valid but product name mismatch"
        )


    return result



# -----------------------------
# Main
# -----------------------------

def main():

    with open(INPUT_FILE) as f:
        dataset = json.load(f)



    validated = []


    for idx, record in enumerate(dataset):

        product = record.get("product")


        if not looks_like_chemical_name(product):

            print(
                f"\nSkipping invalid entry {idx}"
            )

            continue



        validated.append(
            validate_record(
                record,
                idx
            )
        )



    with open(
        OUTPUT_FILE,
        "w"
    ) as f:

        json.dump(
            validated,
            f,
            indent=2
        )


    print(
        "\nFinished."
    )

    print(
        f"Saved: {Path(OUTPUT_FILE).resolve()}"
    )



if __name__ == "__main__":
    main()