"""
seguridad/cuotas.py

Cuotas de uso por día (app pública): protegen la cuota compartida de Groq y
Cohere para que un solo usuario no se la agote a los demás.

  - Corridas por usuario por día   (NUEVAMENTE_MAX_CORRIDAS_DIA, default 5)
  - Corridas globales por día      (NUEVAMENTE_MAX_CORRIDAS_GLOBAL_DIA, default 100)

Una "corrida" es una ejecución del pipeline: se consume cuando el usuario
confirma "¿Ejecutar?" (ahí arrancan los costos: embeddings y LLM). Buscar un
documento, elegir parámetros, responder una aclaración o pedir una modificación
NO consumen otra corrida.

El conteo vive en SQLite (archivo local, NUEVAMENTE_CUOTAS_DB, default
"cuotas.sqlite"), con transacción inmediata para que dos sesiones simultáneas
no se salten el tope. El día se cuenta en UTC.

Los documentos por usuario y el tamaño de archivo los limita el servidor MCP
(servidor_objeStorageOracle.py), que es quien conoce el bucket.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

LIMITE_USUARIO_POR_DEFECTO = 10
LIMITE_GLOBAL_POR_DEFECTO = 100


class ErrorCuota(Exception):
    """Cuota agotada. `mensaje` es seguro para mostrar al usuario."""

    def __init__(self, codigo: str, mensaje: str):
        super().__init__(mensaje)
        self.codigo = codigo
        self.mensaje = mensaje


def _entero_env(nombre: str, default: int) -> int:
    try:
        valor = int(os.environ.get(nombre, default))
        return valor if valor > 0 else default
    except ValueError:
        return default


def limite_usuario() -> int:
    return _entero_env("NUEVAMENTE_MAX_CORRIDAS_DIA", LIMITE_USUARIO_POR_DEFECTO)


def limite_global() -> int:
    return _entero_env("NUEVAMENTE_MAX_CORRIDAS_GLOBAL_DIA", LIMITE_GLOBAL_POR_DEFECTO)


def _ruta_db(ruta: str | None) -> str:
    return ruta or os.environ.get("NUEVAMENTE_CUOTAS_DB", "cuotas.sqlite")


def _dia(ahora: datetime | None) -> str:
    return (ahora or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%d")


def _conectar(ruta: str | None) -> sqlite3.Connection:
    # isolation_level=None: las transacciones se manejan a mano (BEGIN IMMEDIATE).
    conexion = sqlite3.connect(_ruta_db(ruta), timeout=10, isolation_level=None)
    conexion.execute(
        "CREATE TABLE IF NOT EXISTS corridas ("
        " user_id TEXT NOT NULL, dia TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 0,"
        " PRIMARY KEY (user_id, dia))"
    )
    return conexion


def _validar_user_id(user_id) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id inválido.")
    return user_id.strip()


def consumir_corrida(
    user_id: str,
    *,
    ahora: datetime | None = None,
    ruta_db: str | None = None,
    max_usuario: int | None = None,
    max_global: int | None = None,
) -> int:
    """
    Registra una corrida del usuario. Lanza ErrorCuota si agotó su tope diario o
    si se agotó el tope global. Devuelve cuántas corridas le quedan hoy.
    """
    user_id = _validar_user_id(user_id)
    tope_usuario = max_usuario if max_usuario is not None else limite_usuario()
    tope_global = max_global if max_global is not None else limite_global()
    dia = _dia(ahora)

    with closing(_conectar(ruta_db)) as con:
        con.execute("BEGIN IMMEDIATE")
        try:
            fila = con.execute(
                "SELECT n FROM corridas WHERE user_id = ? AND dia = ?", (user_id, dia)
            ).fetchone()
            usadas = fila[0] if fila else 0
            total = con.execute(
                "SELECT COALESCE(SUM(n), 0) FROM corridas WHERE dia = ?", (dia,)
            ).fetchone()[0]

            if usadas >= tope_usuario:
                raise ErrorCuota(
                    "CUOTA_USUARIO",
                    f"Alcanzaste el máximo de {tope_usuario} corridas por día. "
                    "Volvé a probar mañana.",
                )
            if total >= tope_global:
                raise ErrorCuota(
                    "CUOTA_GLOBAL",
                    "NuevaMente alcanzó el límite de uso de hoy. Volvé a probar mañana.",
                )

            con.execute(
                "INSERT INTO corridas (user_id, dia, n) VALUES (?, ?, 1) "
                "ON CONFLICT(user_id, dia) DO UPDATE SET n = n + 1",
                (user_id, dia),
            )
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
    return tope_usuario - usadas - 1


def corridas_restantes(
    user_id: str,
    *,
    ahora: datetime | None = None,
    ruta_db: str | None = None,
    max_usuario: int | None = None,
) -> int:
    """Corridas que le quedan hoy al usuario (sin consumir)."""
    user_id = _validar_user_id(user_id)
    tope = max_usuario if max_usuario is not None else limite_usuario()
    with closing(_conectar(ruta_db)) as con:
        fila = con.execute(
            "SELECT n FROM corridas WHERE user_id = ? AND dia = ?", (user_id, _dia(ahora))
        ).fetchone()
    return max(tope - (fila[0] if fila else 0), 0)