"""
NuevaMente — Configuración central.

Toda la configuración del sistema entra por aquí. Ningún otro módulo lee
variables de entorno directamente: importan `settings` de este archivo.

Principio de diseño: FALLAR TEMPRANO Y CLARO.
Si falta una clave, el error se ve al arrancar con un mensaje que dice qué
falta y cómo arreglarlo — no a los 40 segundos, en medio de una demo,
con un stack trace de una librería HTTP.

Uso:
    from config import settings
    print(settings.llm_provider)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    # python-dotenv es opcional: en producción las variables pueden venir
    # del entorno directamente (p. ej. en una VM de OCI Compute).
    pass


# =============================================================================
# Helpers de lectura
# =============================================================================


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"La variable {name} debe ser un número entero. Valor recibido: {raw!r}"
        ) from exc


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"La variable {name} debe ser un número. Valor recibido: {raw!r}"
        ) from exc


class ConfigError(RuntimeError):
    """Error de configuración, con mensaje accionable para el desarrollador."""


# =============================================================================
# Qué clave necesita cada proveedor
# =============================================================================

# Ollama corre local: no lleva clave.
PROVIDER_API_KEY_VAR: dict[str, str | None] = {
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "kimi": "KIMI_API_KEY",
    "ollama": None,
}

PROVEEDORES_LLM_VALIDOS = set(PROVIDER_API_KEY_VAR)
PROVEEDORES_EMBEDDINGS_VALIDOS = {"ollama", "gemini"}


# =============================================================================
# Configuración
# =============================================================================


@dataclass(frozen=True)
class Settings:
    # --- LLM redactor ---
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "gemini").lower())
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL"))
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.2))
    llm_timeout_seconds: int = field(default_factory=lambda: _get_int("LLM_TIMEOUT_SECONDS", 60))
    llm_max_retries: int = field(default_factory=lambda: _get_int("LLM_MAX_RETRIES", 2))

    # --- LLM verificador (D-02) ---
    verifier_provider: str = field(
        default_factory=lambda: _get("VERIFIER_PROVIDER", _get("LLM_PROVIDER", "gemini")).lower()
    )
    verifier_model: str = field(default_factory=lambda: _get("VERIFIER_MODEL"))
    verifier_temperature: float = field(
        default_factory=lambda: _get_float("VERIFIER_TEMPERATURE", 0.0)
    )

    # --- Embeddings ---
    embeddings_provider: str = field(
        default_factory=lambda: _get("EMBEDDINGS_PROVIDER", "ollama").lower()
    )
    embeddings_model: str = field(default_factory=lambda: _get("EMBEDDINGS_MODEL"))

    # --- Ollama ---
    ollama_base_url: str = field(
        default_factory=lambda: _get("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # --- Umbrales de producto (D-03) ---
    fidelidad_umbral: float = field(default_factory=lambda: _get_float("FIDELIDAD_UMBRAL", 0.80))
    fidelidad_max_reintentos: int = field(
        default_factory=lambda: _get_int("FIDELIDAD_MAX_REINTENTOS", 1)
    )

    # --- RAG ---
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))
    retrieval_top_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_K", 5))
    vector_store_path: Path = field(
        default_factory=lambda: Path(_get("VECTOR_STORE_PATH", "./data/vector_store"))
    )

    # --- OCI ---
    oci_bucket_name: str = field(default_factory=lambda: _get("OCI_BUCKET_NAME"))
    oci_namespace: str = field(default_factory=lambda: _get("OCI_NAMESPACE"))
    oci_region: str = field(default_factory=lambda: _get("OCI_REGION"))
    oci_config_profile: str = field(default_factory=lambda: _get("OCI_CONFIG_PROFILE", "DEFAULT"))
    oci_config_file: str = field(default_factory=lambda: _get("OCI_CONFIG_FILE", "~/.oci/config"))

    # --- Aplicación ---
    max_document_size_mb: int = field(default_factory=lambda: _get_int("MAX_DOCUMENT_SIZE_MB", 10))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO").upper())

    # -------------------------------------------------------------------------

    def api_key_for(self, provider: str) -> str | None:
        """Devuelve la clave del proveedor, o None si no necesita (Ollama)."""
        var = PROVIDER_API_KEY_VAR.get(provider)
        if var is None:
            return None
        return _get(var) or None

    def validate(self) -> None:
        """
        Valida la configuración al arrancar.

        Se llama una vez, al inicio de la aplicación. Cualquier problema sale
        aquí, en claro, y no a mitad de una generación.
        """
        errores: list[str] = []

        # Proveedores válidos
        if self.llm_provider not in PROVEEDORES_LLM_VALIDOS:
            errores.append(
                f"LLM_PROVIDER={self.llm_provider!r} no es válido. "
                f"Opciones: {', '.join(sorted(PROVEEDORES_LLM_VALIDOS))}"
            )
        if self.verifier_provider not in PROVEEDORES_LLM_VALIDOS:
            errores.append(
                f"VERIFIER_PROVIDER={self.verifier_provider!r} no es válido. "
                f"Opciones: {', '.join(sorted(PROVEEDORES_LLM_VALIDOS))}"
            )
        if self.embeddings_provider not in PROVEEDORES_EMBEDDINGS_VALIDOS:
            errores.append(
                f"EMBEDDINGS_PROVIDER={self.embeddings_provider!r} no es válido. "
                f"Opciones: {', '.join(sorted(PROVEEDORES_EMBEDDINGS_VALIDOS))}"
            )

        # Claves presentes para los proveedores realmente en uso.
        # Se agrupa por variable para no repetir el mismo error tres veces
        # cuando varios papeles comparten proveedor.
        faltantes: dict[str, list[str]] = {}
        for papel, proveedor in (
            ("LLM_PROVIDER", self.llm_provider),
            ("VERIFIER_PROVIDER", self.verifier_provider),
            ("EMBEDDINGS_PROVIDER", self.embeddings_provider),
        ):
            if proveedor not in PROVEEDORES_LLM_VALIDOS:
                continue  # ya reportado arriba
            var = PROVIDER_API_KEY_VAR.get(proveedor)
            if var and not _get(var):
                faltantes.setdefault(var, []).append(f"{papel}={proveedor}")

        for var, papeles in faltantes.items():
            errores.append(
                f"Falta {var}: la necesitan {', '.join(papeles)}. "
                f"Agregala a tu archivo .env"
            )

        # Umbral de fidelidad en rango
        if not 0.0 <= self.fidelidad_umbral <= 1.0:
            errores.append(
                f"FIDELIDAD_UMBRAL debe estar entre 0.0 y 1.0. "
                f"Valor actual: {self.fidelidad_umbral}"
            )

        # Coherencia del chunking
        if self.chunk_overlap >= self.chunk_size:
            errores.append(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) debe ser menor que "
                f"CHUNK_SIZE ({self.chunk_size})."
            )

        # OCI: requisito obligatorio del Hackathon
        if not self.oci_bucket_name:
            errores.append("Falta OCI_BUCKET_NAME (requisito obligatorio del Hackathon).")
        if not self.oci_namespace:
            errores.append("Falta OCI_NAMESPACE. Se obtiene con: oci os ns get")

        if errores:
            detalle = "\n".join(f"  · {e}" for e in errores)
            raise ConfigError(
                "\n\nConfiguración incompleta o inválida:\n"
                f"{detalle}\n\n"
                "Revisá tu archivo .env. Si no existe: cp .env.example .env\n"
            )

    def resumen_seguro(self) -> str:
        """Resumen para logs. NUNCA incluye claves."""
        return (
            f"LLM={self.llm_provider}({self.llm_model or 'default'}) · "
            f"Verificador={self.verifier_provider} · "
            f"Embeddings={self.embeddings_provider} · "
            f"Umbral fidelidad={self.fidelidad_umbral} · "
            f"Bucket OCI={self.oci_bucket_name or 'SIN CONFIGURAR'}"
        )


settings = Settings()


if __name__ == "__main__":
    # Diagnóstico rápido:  python config.py
    try:
        settings.validate()
    except ConfigError as exc:
        print(f"❌ {exc}")
        raise SystemExit(1)
    print("✅ Configuración válida")
    print(f"   {settings.resumen_seguro()}")
