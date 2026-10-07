"""
llm_client.py

Factory de LLM compartida por los agentes de NuevaMente que llaman a
Groq (Supervisor, Investigador RAG, Redactor Pedagógico, Crítico/Revisor,
Modificador). Adaptado de DataQualityAgent (core/llm_client.py) vía
MCPMULTIAgentsResearch a LangChain nativo (Opción A): en vez del loop
manual con litellm, arma un ChatOpenAI con .with_fallbacks(), reordenado
según el cupo real que reporta el RateLimiter compartido.

Cadena: openai/gpt-oss-120b -> openai/gpt-oss-20b -> qwen/qwen3.8-27b
(sección 6 del spec -- misma cascada reusada tal cual del otro proyecto,
al ser agnóstica de frontend/flujo). Nota: el tercer modelo es
qwen/qwen3.8-27b, no qwen3.6-27b -- verificado contra los límites reales
habilitados en la cuenta de Groq (console.groq.com/settings/limits).

CLIENTE HTTP Y EVENT LOOP
-------------------------
Si no se le pasa un cliente, langchain_openai reutiliza un httpx.AsyncClient
por defecto que queda cacheado entre instancias. Cuando la app corre varios
asyncio.run(...) (Streamlit + MCP), ese cliente conserva conexiones atadas a un
loop ya cerrado y falla con "RuntimeError: Event loop is closed" al intentar
limpiarlas.

Para evitarlo, `get_llm` acepta un `http_async_client` opcional. Quien llama
desde una corutina crea el cliente con `crear_http_async_client()` dentro de
`async with`, de modo que nace y muere en el loop activo:

    async with crear_http_async_client() as http_client:
        llm = get_llm(mensajes, rate_limiter, http_async_client=http_client)
        resultado = await llm.ainvoke(mensajes)

El parámetro es opcional: los agentes que no lo pasan siguen funcionando igual.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from seguridad.rate_limiter import RateLimiter, estimar_tokens

load_dotenv()

logger = logging.getLogger("nuevamente.llm_client")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_DEFAULT_CHAIN = "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b"
GROQ_MODEL_CHAIN = [
    m.strip() for m in os.getenv("GROQ_MODEL_CHAIN", _DEFAULT_CHAIN).split(",") if m.strip()
]

#: Timeout total de una llamada a Groq (segundos) y de la conexión inicial.
HTTP_TIMEOUT_SEGUNDOS = 90.0
HTTP_CONNECT_TIMEOUT_SEGUNDOS = 15.0


class LLMClientError(Exception):
    """Falta GROQ_API_KEY, o toda la cadena de fallback falló."""


def crear_http_async_client(timeout: float = HTTP_TIMEOUT_SEGUNDOS) -> httpx.AsyncClient:
    """
    Cliente httpx propio para UNA ejecución async.

    Usalo siempre con `async with` desde la corutina que hace la llamada, para
    que el pool de conexiones se cree y se cierre en el mismo event loop.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=HTTP_CONNECT_TIMEOUT_SEGUNDOS)
    )


class _RegistroRateLimiterCallback(BaseCallbackHandler):
    """
    Enganchado a UN modelo puntual de la cadena. Registra cada intento
    (registrar_request) al arrancar la llamada, y el gasto real de
    tokens (registrar_uso) cuando termina con éxito, vía los hooks
    nativos de LangChain.
    """

    def __init__(self, modelo: str, rate_limiter: RateLimiter, tokens_estimados: int):
        self.modelo = modelo
        self.rate_limiter = rate_limiter
        self.tokens_estimados = tokens_estimados

    def on_llm_start(self, *args, **kwargs) -> None:
        self.rate_limiter.registrar_request(self.modelo)

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_reales = self.tokens_estimados
        try:
            uso = response.llm_output.get("token_usage", {})
            tokens_reales = uso.get("total_tokens", self.tokens_estimados)
        except Exception:
            pass
        self.rate_limiter.registrar_uso(self.modelo, tokens_reales)
        logger.info(f"[llm_client] Respondió groq/{self.modelo} ({tokens_reales} tokens)")


def _build_chat_model(
    modelo: str,
    rate_limiter: RateLimiter,
    tokens_estimados: int,
    temperature: float,
    max_tokens: int,
    http_async_client: httpx.AsyncClient | None = None,
) -> ChatOpenAI:
    callback = _RegistroRateLimiterCallback(modelo, rate_limiter, tokens_estimados)

    extra: dict[str, Any] = {}
    if http_async_client is not None:
        extra["http_async_client"] = http_async_client
    if modelo.startswith("openai/gpt-oss"):
        extra["model_kwargs"] = {"reasoning_effort": "low"}

    return ChatOpenAI(
        model=modelo,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=[callback],
        **extra,
    )


def get_llm(
    mensajes: list[dict[str, Any]],
    rate_limiter: RateLimiter,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    http_async_client: httpx.AsyncClient | None = None,
) -> ChatOpenAI:
    """
    Devuelve un ChatOpenAI con .with_fallbacks() ya armado, reordenado
    según qué modelo de la cadena tiene cupo disponible AHORA MISMO --
    se llama una vez por turno de cada agente, no una sola vez al
    importar el módulo, para que la cascada refleje el cupo real de
    cada llamada.

    `http_async_client` (opcional): cliente httpx creado en el event loop
    activo (ver crear_http_async_client). Se comparte entre todos los modelos
    de la cadena. Si es None, se usa el cliente por defecto de langchain_openai.

    Uso típico en un agente de NuevaMente:
        llm = get_llm(mensajes_del_turno, rate_limiter)
        llm_estructurado = llm.with_structured_output(IntencionOut)  # ej. Supervisor
        respuesta = llm_estructurado.invoke(mensajes_del_turno)
    """
    if not GROQ_API_KEY:
        raise LLMClientError(
            "Falta GROQ_API_KEY en el archivo .env. Conseguila gratis en "
            "https://console.groq.com/keys"
        )

    tokens_estimados = estimar_tokens(mensajes, tools)
    modelo_sugerido = rate_limiter.siguiente_modelo_disponible(GROQ_MODEL_CHAIN, tokens_estimados)

    if modelo_sugerido is None:
        # Ningún modelo tiene cupo (TPM / RPM / TPD)
        estado = rate_limiter.estado_modelos()
        logger.error(
            "[llm_client] Todos los modelos de la cadena están sin cupo. "
            "tokens_estimados=%s | estado=%s",
            tokens_estimados,
            estado,
        )
        raise LLMClientError(
            f"Todos los modelos de Groq están sin cupo (TPM/RPM/TPD). "
            f"Pedido actual: {tokens_estimados} tokens. "
            f"Esperá a que se resetee la cuota diaria o subí de plan. "
            f"Estado actual: {estado}"
        )

    candidatos = [modelo_sugerido] + [m for m in GROQ_MODEL_CHAIN if m != modelo_sugerido]
    orden = rate_limiter.filtrar_modelos_viables(candidatos, tokens_estimados)

    if not orden:
        raise LLMClientError(
            f"El pedido ({tokens_estimados} tokens) supera el límite absoluto de "
            "TODOS los modelos de la cadena de Groq."
        )

    principal = _build_chat_model(
        orden[0], rate_limiter, tokens_estimados, temperature, max_tokens, http_async_client
    )
    respaldos = [
        _build_chat_model(
            m, rate_limiter, tokens_estimados, temperature, max_tokens, http_async_client
        )
        for m in orden[1:]
    ]

    if respaldos:
        return principal.with_fallbacks(respaldos)
    return principal