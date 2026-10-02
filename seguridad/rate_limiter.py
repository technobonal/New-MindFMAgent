"""
seguridad/rate_limiter.py

Rate limiter multi-modelo compartido entre los agentes de NuevaMente
que llaman a Groq (Supervisor, Investigador, Redactor Pedagógico,
Crítico/Revisor, Modificador).

Trackea:
  - TPM  (Tokens Per Minute)  → ventana deslizante de 60s
  - RPM  (Requests Per Minute) → ventana deslizante de 60s
  - TPD  (Tokens Per Day)     → contador diario por modelo (reset UTC)

Adaptado de DataQualityAgent + extensión TPD para respetar los límites
reales del free tier de Groq.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None

logger = logging.getLogger("nuevamente.rate_limiter")

# ---------------------------------------------------------------------------
# Límites reales confirmados del free tier de Groq
# (verificados contra console.groq.com/settings/limits + error real TPD)
# ---------------------------------------------------------------------------
MODEL_TPM_LIMITS: dict[str, int] = {
    "openai/gpt-oss-120b": 8_000,
    "openai/gpt-oss-20b": 8_000,
    "qwen/qwen3.8-27b": 8_000,
}

MODEL_RPM_LIMITS: dict[str, int] = {
    "openai/gpt-oss-120b": 30,
    "openai/gpt-oss-20b": 30,
    "qwen/qwen3.8-27b": 30,
}

# Tokens Per Day (TPD) — el que te estaba matando
MODEL_TPD_LIMITS: dict[str, int] = {
    "openai/gpt-oss-120b": 200_000,
    "openai/gpt-oss-20b": 200_000,      # ajustar si tu cuenta tiene otro valor
    "qwen/qwen3.8-27b": 200_000,        # ajustar si tu cuenta tiene otro valor
}

_FALLBACK_TPM_LIMIT = 6_000
_FALLBACK_RPM_LIMIT = 30
_FALLBACK_TPD_LIMIT = 100_000
_DEFAULT_SAFETY_MARGIN = 0.80
_WINDOW_SECONDS = 60


@dataclass
class _VentanaModelo:
    tpm_limit: int
    rpm_limit: int = _FALLBACK_RPM_LIMIT
    tpd_limit: int = _FALLBACK_TPD_LIMIT
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS

    # Ventanas de 60s
    _historial: deque = field(default_factory=deque, repr=False)
    _historial_requests: deque = field(default_factory=deque, repr=False)

    # Contador diario (TPD)
    _tokens_diarios: int = 0
    _dia_utc_actual: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    # ------------------------------------------------------------------
    # Propiedades de presupuesto efectivo
    # ------------------------------------------------------------------
    @property
    def presupuesto_efectivo(self) -> int:
        """TPM efectivo con safety margin."""
        return int(self.tpm_limit * self.safety_margin)

    @property
    def rpm_efectivo(self) -> int:
        return max(1, int(self.rpm_limit * self.safety_margin))

    @property
    def tpd_efectivo(self) -> int:
        """TPD efectivo con safety margin."""
        return int(self.tpd_limit * self.safety_margin)

    # ------------------------------------------------------------------
    # Limpieza de ventanas de 60s
    # ------------------------------------------------------------------
    def _limpiar_historial(self) -> None:
        ahora = time.monotonic()
        while self._historial and (ahora - self._historial[0][0]) > self.window_seconds:
            self._historial.popleft()

    def _limpiar_historial_requests(self) -> None:
        ahora = time.monotonic()
        while self._historial_requests and (ahora - self._historial_requests[0]) > self.window_seconds:
            self._historial_requests.popleft()

    # ------------------------------------------------------------------
    # Reset diario (UTC)
    # ------------------------------------------------------------------
    def _asegurar_dia_actual(self) -> None:
        """Si cambió el día UTC, resetea el contador diario."""
        hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if hoy != self._dia_utc_actual:
            logger.info(
                "RateLimiter: reset TPD del modelo (día anterior=%s → hoy=%s). "
                "Tokens diarios anteriores=%s",
                self._dia_utc_actual,
                hoy,
                self._tokens_diarios,
            )
            self._tokens_diarios = 0
            self._dia_utc_actual = hoy

    # ------------------------------------------------------------------
    # Consultas de uso
    # ------------------------------------------------------------------
    def tokens_usados(self) -> int:
        """Tokens en la ventana de 60s."""
        self._limpiar_historial()
        return sum(tokens for _, tokens in self._historial)

    def requests_usados(self) -> int:
        """Requests en la ventana de 60s."""
        self._limpiar_historial_requests()
        return len(self._historial_requests)

    def tokens_diarios_usados(self) -> int:
        """Tokens consumidos en el día UTC actual."""
        self._asegurar_dia_actual()
        return self._tokens_diarios

    # ------------------------------------------------------------------
    # Decisión de admisión
    # ------------------------------------------------------------------
    def puede_proceder(self, tokens_estimados: int) -> bool:
        self._asegurar_dia_actual()

        tpm_ok = (self.tokens_usados() + tokens_estimados) <= self.presupuesto_efectivo
        rpm_ok = self.requests_usados() < self.rpm_efectivo
        tpd_ok = (self._tokens_diarios + tokens_estimados) <= self.tpd_efectivo

        if not tpd_ok:
            logger.warning(
                "RateLimiter: modelo sin cupo TPD. "
                "usados=%s + pedido=%s > límite_efectivo=%s (límite_real=%s)",
                self._tokens_diarios,
                tokens_estimados,
                self.tpd_efectivo,
                self.tpd_limit,
            )

        return tpm_ok and rpm_ok and tpd_ok

    # ------------------------------------------------------------------
    # Registro de uso
    # ------------------------------------------------------------------
    def registrar_uso(self, tokens: int) -> None:
        """Registra tokens tanto en la ventana de 60s como en el contador diario."""
        self._asegurar_dia_actual()
        self._historial.append((time.monotonic(), tokens))
        self._limpiar_historial()
        self._tokens_diarios += tokens

    def registrar_request(self) -> None:
        self._historial_requests.append(time.monotonic())
        self._limpiar_historial_requests()


@dataclass
class RateLimiter:
    """
    Instancia ÚNICA, compartida entre los agentes que llaman a Groq.
    Se crea una sola vez al arrancar app.py y se pasa por referencia.
    """
    model_limits: dict[str, int] = field(default_factory=lambda: dict(MODEL_TPM_LIMITS))
    model_rpm_limits: dict[str, int] = field(default_factory=lambda: dict(MODEL_RPM_LIMITS))
    model_tpd_limits: dict[str, int] = field(default_factory=lambda: dict(MODEL_TPD_LIMITS))
    safety_margin: float = _DEFAULT_SAFETY_MARGIN
    window_seconds: int = _WINDOW_SECONDS
    _ventanas: dict[str, _VentanaModelo] = field(default_factory=dict, repr=False)

    def _ventana(self, model: str) -> _VentanaModelo:
        if model not in self._ventanas:
            tpm = self.model_limits.get(model, _FALLBACK_TPM_LIMIT)
            rpm = self.model_rpm_limits.get(model, _FALLBACK_RPM_LIMIT)
            tpd = self.model_tpd_limits.get(model, _FALLBACK_TPD_LIMIT)
            self._ventanas[model] = _VentanaModelo(
                tpm_limit=tpm,
                rpm_limit=rpm,
                tpd_limit=tpd,
                safety_margin=self.safety_margin,
                window_seconds=self.window_seconds,
            )
        return self._ventanas[model]

    def cabe_en_limite_absoluto(self, model: str, tokens_estimados: int) -> bool:
        """True si el pedido cabe en el límite absoluto (TPM y TPD) del modelo."""
        v = self._ventana(model)
        return (
            tokens_estimados <= v.presupuesto_efectivo
            and tokens_estimados <= v.tpd_efectivo
        )

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

    # ------------------------------------------------------------------
    # Utilidad de diagnóstico (opcional pero muy útil)
    # ------------------------------------------------------------------
    def estado_modelos(self) -> dict[str, dict[str, Any]]:
        """Devuelve un snapshot del uso actual de cada modelo (para logs/debug)."""
        resultado = {}
        for modelo, ventana in self._ventanas.items():
            resultado[modelo] = {
                "tpm_usados": ventana.tokens_usados(),
                "tpm_limite_efectivo": ventana.presupuesto_efectivo,
                "rpm_usados": ventana.requests_usados(),
                "rpm_limite_efectivo": ventana.rpm_efectivo,
                "tpd_usados": ventana.tokens_diarios_usados(),
                "tpd_limite_efectivo": ventana.tpd_efectivo,
                "dia_utc": ventana._dia_utc_actual,
            }
        return resultado


def estimar_tokens(messages: list[Any], tools: list[dict[str, Any]] | None = None) -> int:
    """Estimación de tokens vía tiktoken; cae a ~4 caracteres/token si no está disponible."""
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