import asyncio
from grafo import nodo_ingesta


async def _test():
    print(">>> Llamando a nodo_ingesta() directo, sin grafo...")
    resultado = await nodo_ingesta({"objeto_id_confirmado": "fuentes/prueba_conexion.txt"})
    print(">>> Resultado:", resultado)


asyncio.run(_test())