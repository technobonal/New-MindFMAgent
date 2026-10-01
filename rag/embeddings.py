"""
rag/embeddings.py

Embeddings de NuevaMente: multilingual-e5-base vía Hugging Face Inference API
(remoto), llamando directo al endpoint HTTP con httpx.

Por qué no HuggingFaceEndpointEmbeddings: huggingface_hub rechaza
feature_extraction para este modelo porque lo tiene registrado como
'sentence-similarity'. El servidor sí responde (verificado con
probe_embeddings.py: forma (n, 768)), así que se evita el cliente.

- Prefijos de e5 aplicados acá: "passage: " al indexar, "query: " al consultar.
  El texto guardado en Chroma queda SIN prefijo.
- Normalización L2 propia (Chroma en espacio coseno).
- Reintentos con backoff ante 429/5xx y errores de red; los 4xx de otro tipo
  (401, 403, 404) fallan de inmediato con el mensaje del servidor.

Variable de entorno requerida: HUGGINGFACE_API_KEY.
"""

from __future__ import annotations

import math
import os
import threading
import time

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"
URL_EMBEDDINGS = (
    f"https://router.huggingface.co/hf-inference/models/{MODELO_EMBEDDINGS}/pipeline/feature-extraction"
)
PREFIJO_PASSAGE = "passage: "
PREFIJO_QUERY = "query: "

TAMANO_LOTE = 32
MAX_REINTENTOS = 3
ESTADOS_REINTENTABLES = {429, 500, 502, 503, 504}


def _normalizar(vector: list[float]) -> list[float]:
    norma = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norma for x in vector]


class EmbeddingsE5(Embeddings):
    """Interfaz LangChain Embeddings con los prefijos de e5 incorporados."""

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def _embeber(self, textos: list[str]) -> list[list[float]]:
        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                r = self._http.post(URL_EMBEDDINGS, json={"inputs": textos})
                if r.status_code in ESTADOS_REINTENTABLES and intento < MAX_REINTENTOS:
                    time.sleep(2 ** intento)
                    continue
                if r.status_code >= 400:
                    raise RuntimeError(f"HF Inference respondió {r.status_code}: {r.text[:200]}")

                vectores = r.json()
                es_lista_de_vectores = (
                    isinstance(vectores, list)
                    and vectores
                    and isinstance(vectores[0], list)
                    and vectores[0]
                    and isinstance(vectores[0][0], (int, float))
                )
                if not es_lista_de_vectores or len(vectores) != len(textos):
                    raise RuntimeError("Respuesta inesperada de HF Inference (forma distinta a (n, dim)).")
                return [_normalizar(v) for v in vectores]
            except httpx.TransportError:
                if intento == MAX_REINTENTOS:
                    raise
                time.sleep(2 ** intento)
        raise RuntimeError("No se pudo obtener embeddings tras los reintentos.")

    def _embeber_por_lotes(self, textos: list[str], prefijo: str) -> list[list[float]]:
        salida: list[list[float]] = []
        for i in range(0, len(textos), TAMANO_LOTE):
            lote = [prefijo + t for t in textos[i:i + TAMANO_LOTE]]
            salida.extend(self._embeber(lote))
        return salida

    # --- interfaz estándar de LangChain (la usa Chroma) ---

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeber_por_lotes(texts, PREFIJO_PASSAGE)

    def embed_query(self, text: str) -> list[float]:
        return self._embeber_por_lotes([text], PREFIJO_QUERY)[0]

    # --- extra: varias consultas de una (la usa el cálculo de anclaje) ---

    def embed_queries(self, textos: list[str]) -> list[list[float]]:
        return self._embeber_por_lotes(textos, PREFIJO_QUERY)


_modelo: EmbeddingsE5 | None = None
_lock = threading.Lock()


def obtener_embeddings() -> EmbeddingsE5:
    """
    Singleton perezoso y thread-safe: la ingesta corre en asyncio.to_thread y
    el Investigador en el pool de hilos de LangGraph. Mismo modelo para
    indexar y consultar.
    """
    global _modelo
    with _lock:
        if _modelo is None:
            token = os.environ.get("HUGGINGFACE_API_KEY")
            if not token:
                raise RuntimeError("Falta HUGGINGFACE_API_KEY en el entorno (.env).")
            http = httpx.Client(
                headers={"Authorization": f"Bearer {token}"},
                timeout=90,
            )
            _modelo = EmbeddingsE5(http)
        return _modelo