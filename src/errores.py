"""
NuevaMente — Catálogo de errores y contrato de respuesta de error.

Resuelve la ambigüedad A-15: el documento fuente sólo trae el ejemplo de éxito.
Este contrato es decisión del equipo (Decision Gate §3).

Dos reglas que gobiernan todo este archivo:

 1. `mensaje_usuario` NUNCA contiene trazas, rutas ni nombres de servicios
    internos (NFR-SEC-02). El detalle técnico va al log, no a la pantalla.

 2. Un fallo de carga a OCI NO produce un error (decisión D-05): produce un
    paquete con `status='exito_con_advertencias'`. Por eso FALLO_PERSISTENCIA
    está en el catálogo pero no se usa para abortar.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .contracts.enums import StatusOperacion, _EnumConAlias

TextoNoVacio = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


# =============================================================================
# Etapas y códigos
# =============================================================================


class EtapaPipeline(_EnumConAlias):
    INGESTA = "ingesta"
    VALIDACION = "validacion"
    INDEXACION = "indexacion"
    RECUPERACION = "recuperacion"
    GENERACION = "generacion"
    VERIFICACION = "verificacion"
    PERSISTENCIA = "persistencia"


class CodigoError(_EnumConAlias):
    # --- ingesta ---
    DOCUMENTO_NO_SOPORTADO = "DOCUMENTO_NO_SOPORTADO"
    DOCUMENTO_VACIO = "DOCUMENTO_VACIO"
    DOCUMENTO_DEMASIADO_GRANDE = "DOCUMENTO_DEMASIADO_GRANDE"
    DOCUMENTO_ILEGIBLE = "DOCUMENTO_ILEGIBLE"
    # --- validación ---
    PARAMETRO_INVALIDO = "PARAMETRO_INVALIDO"
    FORMATO_NO_DISPONIBLE_EN_MVP = "FORMATO_NO_DISPONIBLE_EN_MVP"
    # --- pipeline ---
    FALLO_INDEXACION = "FALLO_INDEXACION"
    FALLO_RECUPERACION = "FALLO_RECUPERACION"
    FALLO_LLM = "FALLO_LLM"
    SALIDA_INVALIDA = "SALIDA_INVALIDA"
    FALLO_VERIFICACION = "FALLO_VERIFICACION"
    # --- persistencia: NO aborta (decisión D-05) ---
    FALLO_PERSISTENCIA = "FALLO_PERSISTENCIA"
    # --- otros ---
    CONFIGURACION_INVALIDA = "CONFIGURACION_INVALIDA"
    ERROR_INTERNO = "ERROR_INTERNO"


#: Mensaje por defecto para el usuario. Sin jerga, sin rutas, sin nombres internos.
MENSAJE_POR_DEFECTO: dict[CodigoError, str] = {
    CodigoError.DOCUMENTO_NO_SOPORTADO: (
        "Ese tipo de archivo no se puede procesar. Aceptamos PDF, Markdown (.md) "
        "y texto plano (.txt)."
    ),
    CodigoError.DOCUMENTO_VACIO: (
        "No se pudo extraer texto del archivo. Puede estar vacio o ser un PDF "
        "escaneado como imagen."
    ),
    CodigoError.DOCUMENTO_DEMASIADO_GRANDE: (
        "El documento supera el tamano maximo permitido. Proba con un capitulo "
        "o una seccion."
    ),
    CodigoError.DOCUMENTO_ILEGIBLE: (
        "El archivo esta danado o no se puede leer. Proba volviendo a exportarlo."
    ),
    CodigoError.PARAMETRO_INVALIDO: (
        "Uno de los parametros no es valido. Revisa el perfil, el formato y el nicho."
    ),
    CodigoError.FORMATO_NO_DISPONIBLE_EN_MVP: (
        "Ese formato todavia no esta disponible en esta version."
    ),
    CodigoError.FALLO_INDEXACION: (
        "No se pudo preparar el documento para la busqueda. Intenta de nuevo."
    ),
    CodigoError.FALLO_RECUPERACION: (
        "No se pudo recuperar el contexto del documento. Intenta de nuevo."
    ),
    CodigoError.FALLO_LLM: (
        "El servicio de inteligencia artificial no respondio. Intenta en unos minutos."
    ),
    CodigoError.SALIDA_INVALIDA: (
        "El contenido generado no cumplio el formato esperado. Intenta de nuevo."
    ),
    CodigoError.FALLO_VERIFICACION: (
        "No se pudo verificar la fidelidad del contenido generado."
    ),
    CodigoError.FALLO_PERSISTENCIA: (
        "El contenido se genero, pero no se pudo guardar en el almacenamiento."
    ),
    CodigoError.CONFIGURACION_INVALIDA: (
        "El sistema no esta configurado correctamente. Avisa al equipo tecnico."
    ),
    CodigoError.ERROR_INTERNO: (
        "Ocurrio un error inesperado. Intenta de nuevo."
    ),
}

ETAPA_POR_CODIGO: dict[CodigoError, EtapaPipeline] = {
    CodigoError.DOCUMENTO_NO_SOPORTADO: EtapaPipeline.INGESTA,
    CodigoError.DOCUMENTO_VACIO: EtapaPipeline.INGESTA,
    CodigoError.DOCUMENTO_DEMASIADO_GRANDE: EtapaPipeline.INGESTA,
    CodigoError.DOCUMENTO_ILEGIBLE: EtapaPipeline.INGESTA,
    CodigoError.PARAMETRO_INVALIDO: EtapaPipeline.VALIDACION,
    CodigoError.FORMATO_NO_DISPONIBLE_EN_MVP: EtapaPipeline.VALIDACION,
    CodigoError.FALLO_INDEXACION: EtapaPipeline.INDEXACION,
    CodigoError.FALLO_RECUPERACION: EtapaPipeline.RECUPERACION,
    CodigoError.FALLO_LLM: EtapaPipeline.GENERACION,
    CodigoError.SALIDA_INVALIDA: EtapaPipeline.GENERACION,
    CodigoError.FALLO_VERIFICACION: EtapaPipeline.VERIFICACION,
    CodigoError.FALLO_PERSISTENCIA: EtapaPipeline.PERSISTENCIA,
    CodigoError.CONFIGURACION_INVALIDA: EtapaPipeline.VALIDACION,
    CodigoError.ERROR_INTERNO: EtapaPipeline.GENERACION,
}


# =============================================================================
# Contrato de respuesta de error
# =============================================================================


class DetalleError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codigo: CodigoError
    mensaje_usuario: TextoNoVacio = Field(
        description="Para mostrar en pantalla. Sin jerga ni trazas (NFR-SEC-02)."
    )
    etapa: EtapaPipeline
    campo: str | None = Field(
        default=None, description="Campo de la solicitud que causo el problema, si aplica."
    )
    valores_admitidos: list[str] | None = Field(
        default=None, description="Para errores de parametro: que valores si sirven."
    )


class RespuestaError(BaseModel):
    """Forma estable de todo error. Cubre FR-ERR-03."""

    model_config = ConfigDict(extra="forbid")

    status: StatusOperacion = StatusOperacion.ERROR
    error: DetalleError


# =============================================================================
# Excepciones internas
# =============================================================================


class NuevaMenteError(Exception):
    """
    Base de los errores del sistema.

    Lleva consigo todo lo necesario para construir la respuesta al usuario, de
    modo que la capa de interfaz no tenga que interpretar excepciones ajenas.
    """

    codigo: CodigoError = CodigoError.ERROR_INTERNO

    def __init__(
        self,
        mensaje_tecnico: str = "",
        *,
        mensaje_usuario: str | None = None,
        campo: str | None = None,
        valores_admitidos: list[str] | None = None,
    ) -> None:
        super().__init__(mensaje_tecnico or self.codigo.value)
        self.mensaje_tecnico = mensaje_tecnico
        self.mensaje_usuario = mensaje_usuario or MENSAJE_POR_DEFECTO[self.codigo]
        self.campo = campo
        self.valores_admitidos = valores_admitidos

    def como_respuesta(self) -> RespuestaError:
        return RespuestaError(
            error=DetalleError(
                codigo=self.codigo,
                mensaje_usuario=self.mensaje_usuario,
                etapa=ETAPA_POR_CODIGO[self.codigo],
                campo=self.campo,
                valores_admitidos=self.valores_admitidos,
            )
        )


class ErrorIngesta(NuevaMenteError):
    codigo = CodigoError.DOCUMENTO_NO_SOPORTADO


class ErrorDocumentoVacio(NuevaMenteError):
    codigo = CodigoError.DOCUMENTO_VACIO


class ErrorDocumentoGrande(NuevaMenteError):
    codigo = CodigoError.DOCUMENTO_DEMASIADO_GRANDE


class ErrorParametro(NuevaMenteError):
    codigo = CodigoError.PARAMETRO_INVALIDO


class ErrorFormatoNoDisponible(NuevaMenteError):
    codigo = CodigoError.FORMATO_NO_DISPONIBLE_EN_MVP


class ErrorIndexacion(NuevaMenteError):
    codigo = CodigoError.FALLO_INDEXACION


class ErrorLLM(NuevaMenteError):
    codigo = CodigoError.FALLO_LLM


class ErrorSalidaInvalida(NuevaMenteError):
    codigo = CodigoError.SALIDA_INVALIDA


class ErrorVerificacion(NuevaMenteError):
    codigo = CodigoError.FALLO_VERIFICACION


class ErrorPersistencia(NuevaMenteError):
    """
    ⚠ Decisión D-05: este error NO aborta la operación.

    Se captura en la capa de orquestación, que entrega el paquete con
    `status='exito_con_advertencias'` y `status_upload='fallido'`.
    """

    codigo = CodigoError.FALLO_PERSISTENCIA
