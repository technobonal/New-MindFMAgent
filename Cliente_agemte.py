"""
Cliente_agemte.py

Cliente MCP reusable: conexión a servidor_objeStorageOracle.py + carga de
tools para LangChain + interpretación de resultados. El Buscador de
Documentos y el resto de los agentes de NuevaMente importan estas
funciones en vez de reimplementar la conexión cada uno.

Mismo patrón que Cliente_agemte.py de MCPMULTIAgentsResearch, adaptado
al servidor de Object Storage de este proyecto.
"""

import asyncio
import logging
import os
import anyio  # <- agregar esta línea
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger("nuevamente.cliente_mcp")

MAX_REINTENTOS_CONEXION = 4  # antes 2 -- con una falla intermitente, más intentos baratos (30s c/u) es mejor que pocos con más timeout cada uno

@asynccontextmanager
async def conectar_mcp():
    print(f"[MCP] Abriendo subproceso...", flush=True)
    parametros_servidor = StdioServerParameters(
        command="python",
        args=["servidor_objeStorageOracle.py"],
        env=os.environ.copy(),
    )

    async with stdio_client(parametros_servidor) as (lectura, escritura):
        async with ClientSession(lectura, escritura) as sesion:
            await sesion.initialize()
            print(f"[MCP] Sesión inicializada", flush=True)
            yield sesion
    print(f"[MCP] Subproceso cerrado", flush=True)  # temporal        


async def obtener_tools_langchain(sesion: ClientSession):
    """
    Carga las tools expuestas por servidor_objeStorageOracle.py ya envueltas
    como BaseTool de LangChain (listar_documentos_fuente, subir_documento_fuente,
    descargar_documento, guardar_resultado, guardar_resultado_formateado),
    listas para bind_tools() o para llamar directo por nombre.
    """
    return await load_mcp_tools(sesion)


def extraer_texto_resultado(resultado_tool) -> str:
    """load_mcp_tools devuelve el contenido como lista de dicts
    [{'type': 'text', 'text': '...'}] -- extrae el texto plano."""
    if isinstance(resultado_tool, list) and resultado_tool:
        primero = resultado_tool[0]
        if isinstance(primero, dict) and "text" in primero:
            return primero["text"]
    return str(resultado_tool)

def extraer_lista_resultado(resultado_tool) -> list:
    """
    Para tools MCP que devuelven una lista de dicts (ej. listar_documentos_fuente).
    A diferencia de extraer_texto_resultado(), acá NO alcanza con leer el primer
    content block: cuando la lista tiene un único elemento, MCP la serializa como
    UN bloque con el objeto suelto (sin corchetes de lista); cuando tiene varios,
    puede venir como un solo bloque con el array completo, o como un bloque por
    item -- esta función cubre ambos casos y siempre devuelve una lista de dicts.
    """
    import json

    if not isinstance(resultado_tool, list):
        return []

    items = []
    for bloque in resultado_tool:
        if isinstance(bloque, dict) and "text" in bloque:
            parseado = json.loads(bloque["text"])
            if isinstance(parseado, list):
                items.extend(parseado)
            else:
                items.append(parseado)
    return items


async def con_reintento_mcp(coro_factory, max_reintentos: int = MAX_REINTENTOS_CONEXION):
    """
    Reintenta una operación completa contra el servidor MCP (conexión +
    trabajo) ante fallos transitorios -- ej. el subproceso de
    servidor_objeStorageOracle.py tarda en levantar, o hay un hiccup de stdio.

    IMPORTANTE (evidencia de pruebas repetidas en Windows): envolver la
    llamada en un cancel scope (asyncio.wait_for o anyio.fail_after)
    provoca que se cuelgue de forma consistente -- interfiere con el
    manejo interno de I/O de anyio sobre stdio en el ProactorEventLoop de
    Windows. Por eso esta función NO usa timeout, a pesar de ser lo
    intuitivo -- la evidencia de 6/6 corridas sin wrapper funcionando vs.
    6/6 con wrapper colgándose es más fuerte que la intuición de diseño.
    Revisar si esto sigue siendo necesario al migrar a Linux (producción).

    coro_factory: función SIN argumentos que, al llamarla, devuelve la
    corutina completa a ejecutar.
    """
    ultimo_error: Exception | None = None
    for intento in range(1, max_reintentos + 2):
        try:
            return await coro_factory()
        except Exception as e:
            ultimo_error = e
            logger.warning(f"[con_reintento_mcp] intento {intento} falló: {e}")
    raise ultimo_error


async def _prueba_conexion():
    """
    Prueba manual de conexión y listado de tools. No se usa en
    producción -- sirve para verificar que el servidor responde.
    """
    print("Iniciando conexión con el Servidor MCP...")

    async with conectar_mcp() as sesion:
        print("✅ Conectado exitosamente.\n")

        tools = await obtener_tools_langchain(sesion)

        print("🛠️ Herramientas cargadas para LangChain:")
        for tool in tools:
            print(f" - Nombre: {tool.name}")
            print(f" - Descripción: {tool.description}\n")

        print("🤖 Ejecutando 'listar_documentos_fuente' para chequear el bucket...\n")
        tool_listar = next(t for t in tools if t.name == "listar_documentos_fuente")

        resultado = await tool_listar.ainvoke({})

        print("📄 Datos devueltos:")
        print(extraer_texto_resultado(resultado))


if __name__ == "__main__":
    asyncio.run(_prueba_conexion())