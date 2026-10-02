import os

import httpx
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ["HUGGINGFACE_API_KEY"]

MODELOS = [
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
TEXTOS = [
    "query: ¿cómo se autentica una instancia sin claves?",
    "passage: Instance Principal permite autenticar una instancia OCI sin claves.",
]


def forma(x):
    dims = []
    while isinstance(x, list):
        dims.append(len(x))
        x = x[0] if x else None
    return tuple(dims)


for modelo in MODELOS:
    url = f"https://router.huggingface.co/hf-inference/models/{modelo}/pipeline/feature-extraction"
    try:
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={"inputs": TEXTOS},
            timeout=90,
        )
    except Exception as e:
        print(f"{modelo}: ERROR de red -> {e}")
        continue

    if r.status_code == 200:
        print(f"{modelo}: OK, forma={forma(r.json())}")
    else:
        print(f"{modelo}: HTTP {r.status_code} -> {r.text[:200]}")