import pandas as pd
import re

csv_path = "data/products_materials_all.csv"

# existing results
df = pd.read_csv(csv_path)

# read recovered Gemini outputs
with open("recovered.csv", "r", encoding="utf-8") as f:
    text = f.read()

# parse Product / Materials pairs
pattern = r"\d+: Product:\s*(.*?)\nMaterials:\s*(.*?)(?=\n\n|\Z)"

matches = re.findall(pattern, text, flags=re.S)

recovered = []

for product, materials in matches:
    recovered.append({
        "product": product.strip(),
        "CAS": "-",
        "materials": materials.strip()
    })

recovered_df = pd.DataFrame(recovered)

# append
combined = pd.concat(
    [df, recovered_df],
    ignore_index=True
)

combined.to_csv(
    csv_path,
    index=False
)

print(f"Added {len(recovered_df)} rows")
print(f"New size: {combined.shape}")