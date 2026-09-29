import asyncio
from langgraph.types import Command
from grafo import construir_grafo


async def _test():
    print(">>> Construyendo grafo...")
    grafo = await construir_grafo()

    config = {"configurable": {"thread_id": "prueba-001"}}

    print(">>> Resumiendo con confirmado=True...")
    try:
        resultado = await asyncio.wait_for(
            grafo.ainvoke(
                Command(resume={"confirmado": True, "objeto_id_confirmado": None}),
                config=config,
            ),
            timeout=240,  # margen generoso: varios nodos, cada uno con sus propios reintentos internos
        )
        print(">>> ainvoke() terminó")
        print("=== Estado tras resumir ===")
        print(resultado)
    except asyncio.TimeoutError:
        print(">>> TIMEOUT: se colgó más de 240 segundos, algo sigue mal")

    await grafo.checkpointer.conn.close()
    print(">>> Listo.")


asyncio.run(_test())