"""Validación determinística de la entrada de NuevaMente antes de delegarla a los agentes.

Dónde se usa cada función en el grafo:
  - nodo_validacion            -> validar_entrada_usuario(estado)
  - nodo_ingesta (antes)       -> validar_objeto_fuente(objeto_id, user_id)
  - nodo_ingesta (después)     -> validar_documento_extraido(texto)
  - tras el Supervisor         -> validar_tema_consulta(tema)
  - confirmar_modificacion     -> validar_instruccion_modificacion(instruccion)

El texto del documento se trata como material de referencia no confiable: aquí se
valida tamaño y codificación y se limpian caracteres de control (típicos de la
extracción de PDF), pero no se bloquea por mencionar seguridad, instrucciones o
ataques.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Mapping

from src.contracts import (
    FORMATOS_IMPLEMENTADOS_MVP,
    FormatoSalida,
    NichoSector,
    NivelDetalle,
    PerfilDestinatario,
)
from src.contracts.request import LONGITUD_MINIMA_CONTENIDO

# Mismo valor que agentes.parametros.PREFIJO_FUENTES y servidor_objeStorageOracle.
PREFIJO_FUENTES = "fuentes/"

MAX_LONGITUD_TEMA = 1_000
MAX_LONGITUD_MODIFICACION = 2_000
MAX_LONGITUD_OBJETO_ID = 1_024


def _leer_limite_documento_mb() -> int:
    """Límite del texto extraído (MB). Por defecto 15, igual que MAX_UPLOAD_BYTES de app.py."""
    try:
        valor = int(os.getenv("NUEVAMENTE_MAX_DOCUMENTO_MB", "15"))
    except ValueError:
        return 15
    return valor if valor > 0 else 15


MAX_DOCUMENTO_BYTES = _leer_limite_documento_mb() * 1024 * 1024

_CONTROLES_NO_PERMITIDOS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")
_INYECCIONES_DIRECTAS = (
    re.compile(
        r"^\s*(?:ahora\s+)?(?:ignora|olvida|desatiende)\s+(?:todas?\s+)?"
        r"(?:las?\s+)?(?:instrucciones|reglas|pol[ií]ticas)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:ignore|forget|disregard)\s+(?:all\s+)?(?:the\s+)?"
        r"(?:previous|prior|above|system)\s+(?:instructions|rules|policies)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:revela|muestra|imprime|copia|env[ií]a)\s+(?:el\s+|la\s+)?"
        r"(?:prompt\s+del\s+sistema|system\s+prompt|clave|credencial|secreto|token)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:reveal|show|print|copy|send)\s+(?:the\s+)?"
        r"(?:system\s+prompt|api\s+key|credential|secret|token)\b",
        re.IGNORECASE,
    ),
)

_TIPOS_ENUM = {
    "perfil_destinatario": PerfilDestinatario,
    "formato_salida": FormatoSalida,
    "nicho_sector": NichoSector,
    "nivel_detalle": NivelDetalle,
}


class ErrorValidacionEntrada(ValueError):
    """Error seguro y estable para cortar el flujo antes de llamar a un LLM."""

    def __init__(self, codigo: str, mensaje: str):
        super().__init__(mensaje)
        self.codigo = codigo
        self.mensaje = mensaje

    def __str__(self) -> str:
        return self.mensaje


def normalizar_entrada_usuario(texto: str, *, campo: str, maximo: int) -> str:
    """Normaliza Unicode y saltos de línea sin borrar contenido silenciosamente."""
    if not isinstance(texto, str):
        raise ErrorValidacionEntrada("ENTRADA_INVALIDA", f"El campo {campo} debe ser texto.")
    normalizado = unicodedata.normalize("NFC", texto).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalizado:
        raise ErrorValidacionEntrada("ENTRADA_VACIA", f"El campo {campo} no puede quedar vacío.")
    if len(normalizado) > maximo:
        raise ErrorValidacionEntrada("ENTRADA_DEMASIADO_LARGA", f"El campo {campo} supera el límite permitido.")
    if _CONTROLES_NO_PERMITIDOS.search(normalizado):
        raise ErrorValidacionEntrada("CARACTERES_NO_PERMITIDOS", f"El campo {campo} contiene caracteres no permitidos.")
    return normalizado


def _detectar_instruccion_directa(texto: str) -> bool:
    """Busca órdenes directas de anulación o revelación, no vocabulario educativo."""
    inicio = texto[:1_000]
    return any(patron.search(inicio) for patron in _INYECCIONES_DIRECTAS)


def _validar_enum_opcional(estado: Mapping[str, Any], campo: str, tipo_enum: type) -> Any:
    valor = estado.get(campo)
    if valor is None:
        return None
    try:
        return tipo_enum(valor)
    except (TypeError, ValueError):
        raise ErrorValidacionEntrada(
            "PARAMETRO_INVALIDO", f"El campo {campo} no coincide con los valores permitidos."
        ) from None


def validar_entrada_usuario(estado: Mapping[str, Any]) -> str:
    """Valida el tema del chat y los parámetros tipados antes de cualquier llamada a un LLM.

    Devuelve el tema normalizado (NFC, sin espacios sobrantes) para guardarlo en el estado.
    """
    tema = normalizar_entrada_usuario(
        estado.get("tema_pedido_chat"), campo="tema", maximo=MAX_LONGITUD_TEMA
    )
    if _detectar_instruccion_directa(tema):
        raise ErrorValidacionEntrada(
            "INSTRUCCION_NO_PERMITIDA",
            "La solicitud contiene una instrucción que no se puede procesar. Reformula el pedido de adaptación.",
        )

    formato = None
    for campo, tipo_enum in _TIPOS_ENUM.items():
        valor = _validar_enum_opcional(estado, campo, tipo_enum)
        if campo == "formato_salida":
            formato = valor
    if formato is not None and formato not in FORMATOS_IMPLEMENTADOS_MVP:
        raise ErrorValidacionEntrada(
            "FORMATO_NO_DISPONIBLE",
            "El formato solicitado está definido, pero no está disponible en esta versión.",
        )
    return tema


def validar_documento_extraido(texto: Any) -> str:
    """Valida y limpia el texto extraído de un documento (PDF, Markdown o texto).

    Descarta caracteres de control y símbolos no codificables en lugar de rechazar
    el documento, porque la extracción de PDF los deja con frecuencia. Rechaza el
    documento si queda sin texto suficiente o supera el tamaño permitido.
    Devuelve el texto limpio: es el que debe indexarse.
    """
    if not isinstance(texto, str):
        raise ErrorValidacionEntrada("DOCUMENTO_INVALIDO", "El contenido del documento no tiene un formato válido.")
    limpio = _CONTROLES_NO_PERMITIDOS.sub("", texto.encode("utf-8", "ignore").decode("utf-8"))
    if len(limpio.strip()) < LONGITUD_MINIMA_CONTENIDO:
        raise ErrorValidacionEntrada(
            "DOCUMENTO_DEMASIADO_CORTO", "El documento no contiene texto suficiente para adaptarlo."
        )
    if len(limpio.encode("utf-8")) > MAX_DOCUMENTO_BYTES:
        raise ErrorValidacionEntrada("DOCUMENTO_DEMASIADO_GRANDE", "El documento supera el tamaño permitido.")
    return limpio


def validar_tema_consulta(tema: Any) -> str | None:
    """Limita el tema extraído por el Supervisor antes de pasarlo al Investigador."""
    if tema is None:
        return None
    if not isinstance(tema, str):
        raise ErrorValidacionEntrada("TEMA_INVALIDO", "El tema solicitado no tiene un formato válido.")
    return normalizar_entrada_usuario(tema, campo="tema de consulta", maximo=MAX_LONGITUD_TEMA)


def validar_instruccion_modificacion(instruccion: Any) -> str:
    """Valida una instrucción puntual antes de entregarla al Modificador."""
    if not isinstance(instruccion, str):
        raise ErrorValidacionEntrada("MODIFICACION_INVALIDA", "La instrucción de modificación debe ser texto.")
    return normalizar_entrada_usuario(
        instruccion, campo="instrucción de modificación", maximo=MAX_LONGITUD_MODIFICACION
    )


def validar_objeto_fuente(objeto_id: Any, user_id: str | None = None) -> str:
    """Limita las lecturas del servidor MCP al prefijo fuentes/.

    Con `user_id` exige además que el documento viva en fuentes/<user_id>/...
    (aislamiento por usuario). Sin `user_id` solo valida el prefijo fuentes/.
    Devuelve el objeto_id tal cual llegó: no se normaliza, para que la descarga
    use exactamente el nombre que tiene el objeto en el bucket.
    """
    if not isinstance(objeto_id, str) or len(objeto_id) > MAX_LONGITUD_OBJETO_ID:
        raise ErrorValidacionEntrada("DOCUMENTO_NO_AUTORIZADO", "El documento seleccionado no es válido.")
    if not objeto_id.startswith(PREFIJO_FUENTES) or "\\" in objeto_id:
        raise ErrorValidacionEntrada("DOCUMENTO_NO_AUTORIZADO", "Solo se pueden leer documentos del área de fuentes.")
    segmentos = objeto_id[len(PREFIJO_FUENTES):].split("/")
    if any(not parte or parte in {".", ".."} for parte in segmentos):
        raise ErrorValidacionEntrada("DOCUMENTO_NO_AUTORIZADO", "El documento seleccionado no es válido.")
    if _CONTROLES_NO_PERMITIDOS.search(objeto_id):
        raise ErrorValidacionEntrada("DOCUMENTO_NO_AUTORIZADO", "El documento seleccionado no es válido.")

    if user_id is not None:
        propietario = user_id.strip() if isinstance(user_id, str) else ""
        if not propietario or "/" in propietario or propietario in {".", ".."}:
            raise ErrorValidacionEntrada("DOCUMENTO_NO_AUTORIZADO", "No se pudo verificar el espacio del usuario.")
        if len(segmentos) < 2 or segmentos[0] != propietario:
            raise ErrorValidacionEntrada(
                "DOCUMENTO_NO_AUTORIZADO", "El documento no pertenece a tu espacio de fuentes."
            )
    return objeto_id