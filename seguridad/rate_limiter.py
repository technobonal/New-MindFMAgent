"""
seguridad/rate_limiter.py

Rate limiter multi-modelo compartido entre los agentes de NuevaMente
que llaman a Groq (Supervisor, Investigador, Redactor Pedagógico,
Crítico/Revisor, Modificador). A diferencia de MCPMULTIAgentsResearch
(fan-out paralelo), acá el grafo es secuencial -- un solo agente llama
al LLM por vez -- pero igual conviene una ÚNICA instancia compartida
(no una por agente) en vez de una por nodo: el loop Revisor<->Redactor
(hasta 2 reintentos) y Revisor<->Modificador pueden acumular varias
llamadas seguidas al mismo modelo dentro de una sola sesión de usuario,
y llevar la cuenta real de cupo evita pisar el límite de Groq aunque
las llamadas sean una detrás de la otra, no simultáneas.

Adaptado de DataQualityAgent (security/rate_limiter.py) vía
MCPMULTIAgentsResearch: misma lógica de ventana deslizante por modelo
(TPM + RPM), sin dependencia de litellm -- usa tiktoken, ya presente
como dependencia de langchain-openai.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None

# Límites reales confirmados del free tier de Groq para la cadena de
# este proyecto (openai/gpt-oss-120b -> openai/gpt-oss-20b ->
# qwen/qwen3.8-27b, sección 6 del spec). Verificados contra
# console.groq.com/settings/limits el 29/09/2026. Si la cuenta cambia
# de plan, actualizar acá.
MODEL_TPM_LIMITS: dict[str, int] = {
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
    "qwen/qwen3.8-27b": 8000,
}
MODEL_RPM_LIMITS: dict[str, int] = {
    "openai/gpt-oss-120b": 30,
    "openai/gpt-oss-20b": 30,
    "qwen/qwen3.8-27b": 30,
}
_FALLBACK_TPM_LIMIT = 6000
_FALLBACK_RPM_LIMIT = 30
_DEFAULT_SAFETY_MARGIN = 0.8
_WINDOW_SECONDS = 60


@dataclass
class _VentanaModelo:
    tpm_limit: int
    rpm_limit: int = _FALLBACK_RPM_LIMIT
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS
    _historial: deque = field(default_factory=deque, repr=False)
    _historial_requests: deque = field(default_factory=deque, repr=False)

    @property
    def presupuesto_efectivo(self) -> int:
        return int(self.tpm_limit * self.safety_margin)

    @property
    def rpm_efectivo(self) -> int:
        return max(1, int(self.rpm_limit * self.safety_margin))

    def _limpiar_historial(self) -> None:
        ahora = time.monotonic()
        while self._historial and (ahora - self._historial[0][0]) > self.window_seconds:
            self._historial.popleft()

    def _limpiar_historial_requests(self) -> None:
        ahora = time.monotonic()
        while self._historial_requests and (ahora - self._historial_requests[0]) > self.window_seconds:
            self._historial_requests.popleft()

    def tokens_usados(self) -> int:
        self._limpiar_historial()
        return sum(tokens for _, tokens in self._historial)

    def requests_usados(self) -> int:
        self._limpiar_historial_requests()
        return len(self._historial_requests)

    def puede_proceder(self, tokens_estimados: int) -> bool:
        tpm_ok = (self.tokens_usados() + tokens_estimados) <= self.presupuesto_efectivo
        rpm_ok = self.requests_usados() < self.rpm_efectivo
        return tpm_ok and rpm_ok

    def registrar_uso(self, tokens: int) -> None:
        self._historial.append((time.monotonic(), tokens))
        self._limpiar_historial()

    def registrar_request(self) -> None:
        self._historial_requests.append(time.monotonic())
        self._limpiar_historial_requests()


@dataclass
class RateLimiter:
    """
    Instancia ÚNICA, compartida entre los agentes que llaman a Groq
    (Supervisor, Investigador, Redactor, Revisor, Modificador) -- se
    crea una sola vez al arrancar app.py y se pasa por referencia a
    cada agente.
    """
    model_limits: dict[str, int] = field(default_factory=lambda: dict(MODEL_TPM_LIMITS))
    model_rpm_limits: dict[str, int] = field(default_factory=lambda: dict(MODEL_RPM_LIMITS))
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS
    _ventanas: dict[str, _VentanaModelo] = field(default_factory=dict, repr=False)

    def _ventana(self, model: str) -> _VentanaModelo:
        if model not in self._ventanas:
            tpm = self.model_limits.get(model, _FALLBACK_TPM_LIMIT)
            rpm = self.model_rpm_limits.get(model, _FALLBACK_RPM_LIMIT)
            self._ventanas[model] = _VentanaModelo(
                tpm_limit=tpm, rpm_limit=rpm,
                safety_margin=self.safety_margin, window_seconds=self.window_seconds,
            )
        return self._ventanas[model]

    def cabe_en_limite_absoluto(self, model: str, tokens_estimados: int) -> bool:
        return tokens_estimados <= self._ventana(model).presupuesto_efectivo

    def filtrar_modelos_viables(self, cadena: list[str], tokens_estimados: int) -> list[str]:
        return [m for m in cadena if self.cabe_en_limite_absoluto(m, tokens_estimados)]

    def puede_proceder(self, model: str, tokens_estimados: int) -> bool:
        return self._ventana(model).puede_proceder(tokens_estimados)

    def registrar_uso(self, model: str, tokens: int) -> None:
        self._ventana(model).registrar_uso(tokens)

    def registrar_request(self, model: str) -> None:
        self._ventana(model).registrar_request()

    def siguiente_modelo_disponible(self, cadena: list[str], tokens_estimados: int) -> str | None:
        for modelo in cadena:
            if self.puede_proceder(modelo, tokens_estimados):
                return modelo
        return None


def estimar_tokens(messages: list[Any], tools: list[dict[str, Any]] | None = None) -> int:
    """Estimación de tokens vía tiktoken; cae a ~4 caracteres/token si no está disponible.
    Soporta tanto dicts ({"role": ..., "content": ...}) como objetos
    BaseMessage de langchain_core (SystemMessage, HumanMessage, ToolMessage, etc.)."""
    def _contenido(m: Any) -> str:
        if isinstance(m, dict):
            return str(m.get("content") or "")
        return str(getattr(m, "content", "") or "")

    texto_total = " ".join(_contenido(m) for m in messages)
    if tools:
        import json
        texto_total += " " + json.dumps(tools, ensure_ascii=False)

    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(texto_total))
        except Exception:
            pass
    return max(1, len(texto_total) // 4)