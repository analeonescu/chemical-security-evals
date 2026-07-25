import json
import re
import time
import urllib.parse
import urllib.request
import pandas as pd

INPUT_JSON = "cwc_cas_validation_report.json"
OUTPUT_CSV = "cwc_precursors.csv"

CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def fetch_json(url):
    try:
        with urllib.request.urlopen(url) as response:
            return json.load(response)
    except Exception:
        return None


def get_cid(name):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{urllib.parse.quote(name)}/cids/JSON"
    )

    data = fetch_json(url)
    if not data:
        return None

    try:
        return data["IdentifierList"]["CID"][0]
    except Exception:
        return None


def get_synonyms(cid):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid}/synonyms/JSON"
    )

    data = fetch_json(url)
    if not data:
        return []

    try:
        return data["InformationList"]["Information"][0]["Synonym"]
    except Exception:
        return []


def extract_cas(synonyms):
    for synonym in synonyms:
        if CAS_PATTERN.match(synonym):
            return synonym
    return None


def main():
    with open(INPUT_JSON, encoding="utf-8") as f:
        chemicals = json.load(f)

    materials = sorted(
        {
            record["material"].strip()
            for record in chemicals
            if record.get("material")
        }
    )

    rows = []

    for i, material in enumerate(materials):
        print(f"[{i+1}/{len(materials)}] {material}")

        cid = get_cid(material)

        if cid is None:
            print("    No PubChem match")
            cas = None
        else:
            synonyms = get_synonyms(cid)
            cas = extract_cas(synonyms)
            print(f"    CAS: {cas}")

        rows.append(
            {
                "product": material,
                "cas": cas,
            }
        )

        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} materials to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()