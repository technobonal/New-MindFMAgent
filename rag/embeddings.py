"""
rag/embeddings.py

Embeddings de NuevaMente con proveedor conmutable por variable de entorno:

  EMBEDDINGS_PROVIDER=hf      -> multilingual-e5-base vía Hugging Face Inference API (768 dim)
  EMBEDDINGS_PROVIDER=cohere  -> embed-multilingual-v3.0 vía API de Cohere (1024 dim)

Ambos se llaman directo por HTTP con httpx (sin SDKs).

- HF: prefijos de e5 aplicados acá ("passage: " al indexar, "query: " al consultar).
- Cohere: sin prefijos; usa input_type (search_document / search_query).
  El texto guardado en Chroma queda SIN prefijo en ambos casos.
- Normalización L2 propia (Chroma en espacio coseno).
- Reintentos con backoff ante 429/5xx y errores de red; los 4xx de otro tipo
  (401, 402, 403, 404) fallan de inmediato con ErrorEmbeddings.
- Guardia de red: al armar el singleton se valida el destino (HF o Cohere) con
  seguridad.guardia_red; si no cumple la política se levanta ErrorEmbeddings.
  El cliente httpx no sigue redirecciones.

IMPORTANTE: las dimensiones son distintas (768 vs 1024). Por eso vectorstore.py
usa un directorio de Chroma distinto por proveedor: nunca se mezclan.

Variables de entorno:
  EMBEDDINGS_PROVIDER   "hf" (por defecto) o "cohere"
  HUGGINGFACE_API_KEY   si el proveedor es hf
  COHERE_API_KEY        si el proveedor es cohere
"""

from __future__ import annotations

import math
import os
import threading
import time

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from seguridad.guardia_red import ErrorGuardiaRed, validar_endpoint_proveedor

load_dotenv()

PROVEEDORES_VALIDOS = {"hf", "cohere"}
_proveedor_crudo = os.environ.get("EMBEDDINGS_PROVIDER", "hf").strip().lower()
# Si el valor es inválido se conserva un marcador neutro: así ni el nombre del
# directorio de Chroma ni los mensajes repiten el contenido del .env.
PROVEEDOR_EMBEDDINGS = _proveedor_crudo if _proveedor_crudo in PROVEEDORES_VALIDOS else "invalido"

# --- Hugging Face ---
MODELO_HF = "intfloat/multilingual-e5-base"
URL_HF = f"https://router.huggingface.co/hf-inference/models/{MODELO_HF}/pipeline/feature-extraction"
PREFIJO_PASSAGE = "passage: "
PREFIJO_QUERY = "query: "
TAMANO_LOTE_HF = 32

# --- Cohere ---
MODELO_COHERE = "embed-multilingual-v3.0"
URL_COHERE = "https://api.cohere.com/v2/embed"
TAMANO_LOTE_COHERE = 96  # máximo permitido por llamada

MAX_REINTENTOS = 4
ESTADOS_REINTENTABLES = {429, 500, 502, 503, 504}

# --- Umbrales de similitud coseno por proveedor ---
# Cada modelo reparte la escala distinto: los scores de e5 quedan comprimidos
# (~0.7-0.9) y los de Cohere son más bajos y más abiertos. Un mismo número NO
# sirve para ambos.
#   similitud     -> Investigador (y Modificador): chunk relevante para la consulta
#   soportada     -> Revisor: afirmación bien respaldada por la fuente
#   parcial       -> Revisor: respaldo parcial (por debajo = ítem débil)
#   score_minimo  -> Revisor: promedio mínimo para "superó el umbral"
# Los valores de "hf" son los originales. Los de "cohere" son un PUNTO DE PARTIDA
# estimado con pocas muestras: calibrar con un documento real.
UMBRALES: dict[str, dict[str, float]] = {
    "hf": {"similitud": 0.78, "soportada": 0.80, "parcial": 0.70, "score_minimo": 0.75},
    "cohere": {"similitud": 0.45, "soportada": 0.60, "parcial": 0.45, "score_minimo": 0.50},
}
UMBRALES_ACTIVOS: dict[str, float] = UMBRALES.get(PROVEEDOR_EMBEDDINGS, UMBRALES["hf"])


class ErrorEmbeddings(RuntimeError):
    """
    Falla del servicio de embeddings. Sigue siendo RuntimeError, así que el
    código existente que captura RuntimeError no cambia.

    `codigo` es el HTTP status (o None) y `mensaje_usuario` es un texto corto
    pensado para mostrarse en Streamlit.
    """

    def __init__(self, detalle: str, codigo: int | None = None, mensaje_usuario: str | None = None):
        super().__init__(detalle)
        self.codigo = codigo
        self.mensaje_usuario = mensaje_usuario or "Falló el servicio de embeddings."


def _mensaje_usuario(proveedor: str, codigo: int) -> str:
    nombre = "Hugging Face" if proveedor == "hf" else "Cohere"
    if codigo == 402:
        return f"El servicio de embeddings ({nombre}) se quedó sin créditos."
    if codigo == 429:
        return f"El servicio de embeddings ({nombre}) alcanzó su límite de uso. Esperá un minuto y reintentá."
    if codigo in (401, 403):
        return f"La clave de API de {nombre} no es válida o no tiene permisos. Revisá el .env."
    return f"El servicio de embeddings ({nombre}) respondió con error {codigo}."


def _normalizar(vector: list[float]) -> list[float]:
    norma = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norma for x in vector]


class EmbeddingsRemotos(Embeddings):
    """Interfaz LangChain Embeddings sobre HF (e5) o Cohere, según el proveedor."""

    def __init__(self, http: httpx.Client, proveedor: str) -> None:
        self._http = http
        self._proveedor = proveedor
        self._tamano_lote = TAMANO_LOTE_COHERE if proveedor == "cohere" else TAMANO_LOTE_HF

    # --- llamadas HTTP (una por proveedor) ---

    def _peticion(self, textos: list[str], tipo: str) -> httpx.Response:
        if self._proveedor == "cohere":
            return self._http.post(
                URL_COHERE,
                json={
                    "model": MODELO_COHERE,
                    "texts": textos,
                    "input_type": "search_document" if tipo == "documento" else "search_query",
                    "embedding_types": ["float"],
                },
            )
        prefijo = PREFIJO_PASSAGE if tipo == "documento" else PREFIJO_QUERY
        return self._http.post(URL_HF, json={"inputs": [prefijo + t for t in textos]})

    def _extraer_vectores(self, respuesta: httpx.Response) -> list:
        datos = respuesta.json()
        if self._proveedor == "cohere":
            return (datos.get("embeddings") or {}).get("float") if isinstance(datos, dict) else None
        return datos

    # --- lógica común ---

    def _embeber(self, textos: list[str], tipo: str) -> list[list[float]]:
        nombre = "Cohere" if self._proveedor == "cohere" else "HF Inference"
        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                r = self._peticion(textos, tipo)
                if r.status_code in ESTADOS_REINTENTABLES and intento < MAX_REINTENTOS:
                    time.sleep(2 ** (intento + 1))  # 4s, 8s, 16s
                    continue
                if r.status_code >= 400:
                    raise ErrorEmbeddings(
                        f"{nombre} respondió {r.status_code}: {r.text[:200]}",
                        codigo=r.status_code,
                        mensaje_usuario=_mensaje_usuario(self._proveedor, r.status_code),
                    )

                vectores = self._extraer_vectores(r)
                es_lista_de_vectores = (
                    isinstance(vectores, list)
                    and vectores
                    and isinstance(vectores[0], list)
                    and vectores[0]
                    and isinstance(vectores[0][0], (int, float))
                )
                if not es_lista_de_vectores or len(vectores) != len(textos):
                    raise ErrorEmbeddings(f"Respuesta inesperada de {nombre} (forma distinta a (n, dim)).")
                return [_normalizar(v) for v in vectores]
            except httpx.TransportError as e:
                if intento == MAX_REINTENTOS:
                    raise ErrorEmbeddings(
                        f"Error de red con {nombre}: {e}",
                        mensaje_usuario=f"No se pudo conectar con el servicio de embeddings ({nombre}).",
                    ) from e
                time.sleep(2 ** intento)
        raise ErrorEmbeddings("No se pudo obtener embeddings tras los reintentos.")

    def _embeber_por_lotes(self, textos: list[str], tipo: str) -> list[list[float]]:
        salida: list[list[float]] = []
        for i in range(0, len(textos), self._tamano_lote):
            salida.extend(self._embeber(textos[i:i + self._tamano_lote], tipo))
        return salida

    # --- interfaz estándar de LangChain (la usa Chroma) ---

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeber_por_lotes(texts, "documento")

    def embed_query(self, text: str) -> list[float]:
        return self._embeber_por_lotes([text], "consulta")[0]

    # --- extra: varias consultas de una (la usa el cálculo de anclaje) ---

    def embed_queries(self, textos: list[str]) -> list[list[float]]:
        return self._embeber_por_lotes(textos, "consulta")


# Alias por compatibilidad con imports anteriores.
EmbeddingsE5 = EmbeddingsRemotos

_modelo: EmbeddingsRemotos | None = None
_lock = threading.Lock()


def obtener_embeddings() -> EmbeddingsRemotos:
    """
    Singleton perezoso y thread-safe: la ingesta corre en asyncio.to_thread y
    el Investigador en el pool de hilos de LangGraph. Mismo modelo para
    indexar y consultar.
    """
    global _modelo
    with _lock:
        if _modelo is None:
            if PROVEEDOR_EMBEDDINGS not in PROVEEDORES_VALIDOS:  # "invalido"
                raise ErrorEmbeddings(
                    "EMBEDDINGS_PROVIDER no es válido (no se muestra el valor por seguridad). "
                    "Debe ser exactamente 'hf' o 'cohere', sin comillas ni espacios.",
                    mensaje_usuario="Configuración de embeddings inválida (EMBEDDINGS_PROVIDER).",
                )
            if PROVEEDOR_EMBEDDINGS == "cohere":
                variable = "COHERE_API_KEY"
                url_destino = URL_COHERE
            else:
                variable = "HUGGINGFACE_API_KEY"
                url_destino = URL_HF
            token = os.environ.get(variable)
            if not token:
                raise ErrorEmbeddings(
                    f"Falta {variable} en el entorno (.env).",
                    mensaje_usuario=f"Falta {variable} en el .env.",
                )
            # Política de red: el destino debe ser el aprobado para el proveedor.
            try:
                validar_endpoint_proveedor(PROVEEDOR_EMBEDDINGS, url_destino)
            except ErrorGuardiaRed as e:
                raise ErrorEmbeddings(
                    f"Destino de embeddings rechazado por la política de red: {e}",
                    mensaje_usuario="El destino del servicio de embeddings no está permitido por la política de red.",
                ) from e
            http = httpx.Client(
                headers={"Authorization": f"Bearer {token}"},
                timeout=90,
                follow_redirects=False,
            )
            _modelo = EmbeddingsRemotos(http, PROVEEDOR_EMBEDDINGS)
        return _modelo