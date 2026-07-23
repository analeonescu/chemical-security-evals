"""For the chemical products extracted from different databases 
    mentioned in dataset/build_dataset.py, this script queries Gemini 
    API to find the most common starting material used to synthesize each product.
"""
from google import genai
import os
import pandas as pd
import time

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)
# client.models.list()

# load dataset of products
products_df = pd.read_csv('./cwc_precursors.csv')


def call_gemini(prompt, max_retries=10):
    """
    Call Gemini API with automatic retry if rate limit is exceeded.
    """
    retries = 0

    while retries < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",

                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            error_message = str(e)

            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                wait_time = 30 * (retries + 1)

                print(
                    f"Rate limit reached. "
                    f"Waiting {wait_time}s before retry..."
                )

                time.sleep(wait_time)
                retries += 1

            else:
                # If it is not a rate-limit error, stop
                raise e

    raise RuntimeError("Maximum retries exceeded")


materials = []


for idx in range(products_df.shape[0]):
    product = products_df['product'].iloc[idx]
    cas = products_df['cas'].iloc[idx]

    prompt = f"""
You are a chemistry expert.

For the chemical product below, provide the most common
starting material used to synthesise it. If that is not straightforward,
suggest a chemically similar compound.

Make sure you only suggest one starting material. If there are more possibilities, 
choose the option most similar to the product and submit it.

If there are discrepancies between the name and the CAS number,
always trust the CAS number.

If there are no known starting materials, suggest a chemical similar to the product.

Product:
{product}

Product's CAS No.:
{cas}

Return only a comma-separated list of materials.
Do not include explanations.
"""

    output = call_gemini(prompt)

    print(
        f"{idx}: Product: {product}\n"
        f"Materials: {output}\n"
    )

    materials.append({
        "product": product,
        "CAS": cas,
        "materials": output
    })

    # normal delay between successful requests
    time.sleep(5)


    df = pd.DataFrame(materials)

    df.to_csv(
        'data/cwc_precursors_materials_all.csv',
        index=False
        )

print(materials)