"""
seguridad/auth.py

Login con Google para la app pública (Streamlit nativo: st.login / st.user).
Reusa la misma app de Google Cloud de MCPMULTIAgentsResearch: solo hay que
agregar el redirect URI de Streamlit (https://<tu-dominio>/oauth2callback) al
cliente OAuth existente.

Identidad:
  - user_id = `sub` de Google (estable e inmutable). NO se usa el email: Gmail
    ignora puntos y "+etiqueta", así que con el email alguien podría duplicar su
    identidad para esquivar las cuotas. Además el sub no deja datos personales
    en los nombres de objeto del bucket.
  - Se exige email_verified.
  - Cualquier cuenta de Google con email verificado puede entrar.

Requisitos: streamlit >= 1.42 y Authlib >= 1.3.2 (requirements.txt).

.streamlit/secrets.toml (los valores reales NO van al repo):

    [auth]
    redirect_uri = "https://TU-DOMINIO/oauth2callback"
    cookie_secret = "una-cadena-larga-y-aleatoria"

    [auth.google]
    client_id = "XXXX.apps.googleusercontent.com"
    client_secret = "XXXX"
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

En Google Cloud, la pantalla de consentimiento tiene que estar "En producción"
(en "Prueba" solo entran usuarios cargados a mano y la sesión vence a los 7 días).
Con los permisos básicos (openid, email, profile) no pide verificación de Google.

Uso en app.py:

    usuario = requerir_login()   # corta la página con la pantalla de login si no hay sesión
    usuario.user_id              # para config["configurable"]["user_id"], cuotas y OCI
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

PROVEEDOR = "google"

_USER_ID_VALIDO = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ErrorAutenticacion(Exception):
    """La sesión no sirve para identificar al usuario. `mensaje` es seguro para mostrar."""

    def __init__(self, codigo: str, mensaje: str):
        super().__init__(mensaje)
        self.codigo = codigo
        self.mensaje = mensaje


@dataclass(frozen=True)
class Usuario:
    user_id: str          # `sub` de Google
    email: str            # solo para mostrar en la UI
    nombre: str | None = None


def _es_verdadero(valor: Any) -> bool:
    if isinstance(valor, str):
        return valor.strip().lower() == "true"
    return valor is True


def identidad_desde_claims(claims: Mapping[str, Any]) -> Usuario:
    """
    Convierte los claims del ID token (st.user) en un Usuario, o lanza
    ErrorAutenticacion. Función pura: se prueba sin Streamlit.
    """
    sub = claims.get("sub")
    if not isinstance(sub, str) or not _USER_ID_VALIDO.fullmatch(sub):
        raise ErrorAutenticacion(
            "IDENTIDAD_INVALIDA", "No se pudo identificar tu cuenta de Google. Vuelve a iniciar sesión."
        )
    email = claims.get("email")
    if not isinstance(email, str) or not email.strip():
        raise ErrorAutenticacion(
            "EMAIL_AUSENTE", "Tu cuenta de Google no compartió un email. Vuelve a iniciar sesión."
        )
    if not _es_verdadero(claims.get("email_verified")):
        raise ErrorAutenticacion(
            "EMAIL_NO_VERIFICADO", "Tu cuenta de Google no tiene el email verificado."
        )
    nombre = claims.get("name")
    return Usuario(
        user_id=sub,
        email=email.strip().lower(),
        nombre=nombre.strip() if isinstance(nombre, str) and nombre.strip() else None,
    )


def _claims_de_streamlit(st_module) -> Mapping[str, Any] | None:
    """Devuelve los claims de st.user si hay sesión iniciada; None si no."""
    usuario = getattr(st_module, "user", None)
    if usuario is None or not getattr(usuario, "is_logged_in", False):
        return None
    try:
        return usuario.to_dict()
    except AttributeError:
        return dict(usuario)


def requerir_login() -> Usuario:
    """
    Exige sesión. Sin sesión, muestra la pantalla de ingreso y detiene la página
    (st.stop). Con sesión válida devuelve el Usuario. Con sesión inválida (p. ej.
    email sin verificar) muestra el motivo y ofrece cerrar sesión.
    """
    import streamlit as st

    if not hasattr(st, "login"):
        st.error("Esta app necesita Streamlit 1.42 o superior para el inicio de sesión.")
        st.stop()

    claims = _claims_de_streamlit(st)
    if claims is None:
        _pantalla_login(st)
        st.stop()

    try:
        return identidad_desde_claims(claims)
    except ErrorAutenticacion as exc:
        st.error(exc.mensaje)
        st.button("Cerrar sesión", on_click=st.logout)
        st.stop()


def _pantalla_login(st) -> None:
    st.markdown(
        """
        <div style="max-width:560px;margin:12vh auto 24px auto;text-align:center">
          <div style="font-size:2.4rem;font-weight:800;color:#FFFFFF;margin-bottom:10px">NuevaMente</div>
          <div style="color:#8B96A8;font-size:1.05rem;line-height:1.6;margin-bottom:8px">
            Convierte documentación técnica en contenido educativo personalizado.
            Inicia sesión con tu cuenta de Google para empezar.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, centro, _ = st.columns([1, 1, 1])
    with centro:
        st.button(
            "Iniciar sesión con Google",
            on_click=st.login,
            args=(PROVEEDOR,),
            use_container_width=True,
            type="primary",
        )
    st.markdown(
        '<div style="text-align:center;color:#8B96A8;font-size:0.8rem;margin-top:18px">'
        "Solo usamos tu cuenta para identificarte. Tus documentos son visibles únicamente para ti."
        "</div>",
        unsafe_allow_html=True,
    )


def cerrar_sesion() -> None:
    import streamlit as st

    st.logout()