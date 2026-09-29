import asyncio
from Cliente_agemte import conectar_mcp, obtener_tools_langchain, extraer_texto_resultado


async def _test():
    print(">>> Conectando...", flush=True)
    async with conectar_mcp() as sesion:
        print(">>> Conectado, cargando tools...", flush=True)
        tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
        print(">>> Tools cargadas, descargando...", flush=True)
        resultado = await tools["descargar_documento"].ainvoke({"objeto_id": "fuentes/prueba_conexion.txt"})
        print(">>> Respuesta recibida:", flush=True)
        print(extraer_texto_resultado(resultado))


asyncio.run(_test())