import asyncio

# Solo el import, nada más -- para ver si el import en sí rompe algo
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from Cliente_agemte import conectar_mcp, obtener_tools_langchain, extraer_texto_resultado


async def _test():
    print(">>> Conectando (con langgraph ya importado arriba)...")
    async with conectar_mcp() as sesion:
        print(">>> Conectado, cargando tools...")
        tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
        print(">>> Tools cargadas, descargando...")
        resultado = await tools["descargar_documento"].ainvoke({"objeto_id": "fuentes/prueba_conexion.txt"})
        print(">>> Respuesta recibida:")
        print(extraer_texto_resultado(resultado))


asyncio.run(_test())