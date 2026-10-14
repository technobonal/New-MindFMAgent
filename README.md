# NuevaMente :: Agente Educativo Multiagente

## Descripción del proyecto

**NuevaMente** es una plataforma que transforma **documentación técnica** (manuales, artículos, bases de conocimiento en PDF, Markdown o texto) en **contenido educativo personalizado y estructurado**, usando IA generativa, RAG y orquestación de agentes con LangGraph.

Las personas que aprenden o capacitan a otras (equipos de desarrollo, líderes técnicos, gestores) suelen enfrentarse a documentación larga, densa y escrita para otro público. Revisarla y adaptarla a mano consume tiempo y se presta a errores.

Este proyecto resuelve ese problema: el usuario conversa con el sistema, elige un documento y cuatro parámetros de personalización, y un equipo de agentes entrega un material listo para estudiar, **anclado únicamente a las fuentes originales**, sin inventar información que no esté en el documento.

### Parámetros de personalización

| Parámetro | Opciones |
|---|---|
| **Perfil del destinatario** | Principiante · Desarrollador · Líder Técnico · Gestor Ejecutivo |
| **Formato pedagógico** | Tutorial · Flashcards · Quiz · Resumen Ejecutivo · Guion de Clase |
| **Nicho / contexto** | Fintech · Salud · E-commerce · General (por defecto) |
| **Nivel de detalle** | Didáctico · Estándar (por defecto) · Profundo |

El asistente del chat pide los parámetros uno por uno, empezando por el tema o documento. Nicho y nivel de detalle entran con valores por defecto y se pueden cambiar en la pantalla de confirmación.

### Qué entrega

- Contenido adaptado según el formato elegido, con metadatos de aprendizaje (conceptos clave, prerrequisitos, tiempo estimado de estudio).
- Evaluación de calidad con `anclaje_fuente_score`: proporción de afirmaciones respaldadas por los fragmentos del documento fuente.
- Descarga en **PDF** y **Markdown**, y botón de **imprimir**.
- Persistencia automática del resultado en **Oracle Cloud Infrastructure (OCI) Object Storage**.
- **Modificaciones iterativas**: después del resultado, el usuario puede pedir cambios (hasta 5 vueltas por sesión). Cada versión se guarda como archivo nuevo, sin sobrescribir.

## Arquitectura

El sistema combina un grafo de agentes (LangGraph) con un pipeline **RAG**. El acceso a OCI es **determinístico** (llamadas directas a funciones, sin que un LLM decida si usar la herramienta) y se hace a través de un servidor **MCP** propio.

```
┌──────────────────────┐        ┌────────────────────────────────┐
│  Usuario (Streamlit) │        │  OCI Object Storage (bucket)   │
│  chat + parámetros   │        │  fuentes/ · generados/formateados/ │
└──────────┬───────────┘        └───────────────▲────────────────┘
           │                                    │ MCP (servidor_objeStorageOracle.py)
           ▼                                    │
┌──────────────────────┐   listar / descargar   │
│ Buscador de          │◄───────────────────────┤
│ Documentos           │                        │
└──────────┬───────────┘                        │
           ▼                                    │
┌──────────────────────┐                        │
│ Parámetros (asistente│  perfil · formato      │
│ paso a paso)         │  nicho · nivel         │
└──────────┬───────────┘                        │
           ▼                                    │
┌──────────────────────┐                        │
│ ¿Ejecutar? (HITL)    │                        │
└──────────┬───────────┘                        │
           ▼                                    │
┌──────────────────────┐   chunking + embeddings│
│ Ingesta + Validación │──► Chroma (vector store)
└──────────┬───────────┘                        │
           ▼                                    │
┌──────────────────────┐                        │
│ Supervisor           │  extrae el tema        │
└──────────┬───────────┘                        │
           ▼                                    │
┌──────────────────────┐  no coincide ┌──────────────────┐
│ Investigador (RAG)   │─────────────►│ Aclaración (HITL)│
└──────────┬───────────┘◄─────────────└──────────────────┘
           ▼
┌──────────────────────┐
│ Redactor Pedagógico  │◄──────┐ errores de formato / typos
└──────────┬───────────┘       │ (máx. 2 reintentos)
           ▼                   │
┌──────────────────────┐       │
│ Crítico / Revisor    │───────┘
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Ilustrador (íconos)  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐   guarda en OCI   ┌─────────────────────┐
│ Guardado final       │──────────────────►│ ¿Modificar? (HITL)  │
└──────────────────────┘                   │ → Modificador →     │
                                           │   Revisor (loop)    │
                                           └─────────────────────┘
```

### Agentes

| Agente | Rol |
|---|---|
| **Supervisor** | Clasifica la intención y extrae el tema. No genera contenido. |
| **Investigador** | Consulta Chroma y verifica que el tema pedido coincida con el documento. Si no coincide, pide **una** aclaración al usuario; si sigue sin coincidir, corta con error. |
| **Redactor Pedagógico** | Genera el contenido en el formato pedido, respetando la especificación pedagógica del perfil y sin agregar datos fuera de los fragmentos fuente. |
| **Crítico / Revisor** | Revisa formato y typos y, después, fidelidad a la fuente. Si se agotan los reintentos, entrega el mejor resultado con status `exito_con_advertencias`. |
| **Modificador** | Aplica los cambios que pide el usuario y pasa de nuevo por el Revisor. |
| **Ilustrador** | Asigna un ícono de una lista permitida a cada flashcard (determinístico, sin LLM). |

### Puntos de intervención humana (HITL)

El grafo usa `interrupt()` de LangGraph con un checkpointer SQLite para pausar y retomar el flujo entre turnos:

1. Elegir documento si hay 0 o varios candidatos.
2. Confirmar parámetros y ejecución.
3. Aclaración del Investigador cuando el tema no coincide.
4. Pregunta de modificación después de cada resultado.

## Tecnologías utilizadas

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12+ |
| Orquestación de agentes | LangGraph (`StateGraph`, `interrupt`, `AsyncSqliteSaver`) |
| LLM | Groq con cascada de respaldo: `openai/gpt-oss-120b` → `openai/gpt-oss-20b` → `qwen/qwen3.8-27b` |
| Embeddings | Cohere `embed-multilingual-v3.0` (por defecto en la nube) o Hugging Face `multilingual-e5-base`, conmutables con `EMBEDDINGS_PROVIDER` |
| Base de datos vectorial | Chroma (similitud coseno, filtro por usuario y documento) |
| Integración con OCI | Servidor MCP propio (FastMCP) + `langchain-mcp-adapters` |
| Almacenamiento | OCI Object Storage |
| Extracción de documentos | pypdf |
| Exportación | reportlab (PDF) + Markdown |
| Validación de contratos | Pydantic |
| Autenticación | `st.login` de Streamlit con Google (OIDC) |
| Interfaz web | Streamlit |
| Despliegue | Streamlit Community Cloud (actual) · instancia OCI con Nginx/HTTPS (planificado) |

## Seguridad y control de uso

- **Login con Google** y correo verificado obligatorio. Cada usuario ve solo sus propios documentos y resultados (los prefijos del bucket incluyen el identificador del usuario).
- **Cuotas**: 10 corridas por usuario por día y 100 globales por día (`NUEVAMENTE_MAX_CORRIDAS_DIA`, `NUEVAMENTE_MAX_CORRIDAS_GLOBAL_DIA`). En Community Cloud el contador se reinicia cuando la app reinicia.
- **Límites de entrada**: tamaño máximo de documento (15 MB por defecto) y de longitud del tema y de las modificaciones.
- **Validadores**: sanitización de entradas y heurística contra *prompt injection*.
- **Guardia de red**: allowlist de dominios de salida (Groq, Hugging Face, Cohere y el endpoint regional de OCI Object Storage).
- **Rate limiter** compartido entre agentes para respetar los límites de tokens de Groq.
- **Anti-alucinación**: el Redactor solo usa fragmentos recuperados; si no hay evidencia suficiente, se abstiene en lugar de inventar.

## Estructura del proyecto

```
New-MindFMAgent/
├── app.py                          # UI de Streamlit (chat, timeline en vivo, resultados)
├── grafo.py                        # Grafo LangGraph y enrutamiento
├── agent_state.py                  # Estado compartido del grafo
├── llm_client.py                   # Cliente Groq con cascada de modelos
├── Cliente_agemte.py               # Cliente MCP reutilizable
├── servidor_objeStorageOracle.py   # Servidor MCP: tools de OCI Object Storage
├── render_documento.py             # Render de los resultados
├── agentes/                        # Supervisor, Investigador, Redactor, Revisor, Modificador, Ilustrador
├── rag/                            # Embeddings y vector store (Chroma)
├── seguridad/                      # Auth, cuotas, validadores, guardia de red, rate limiter
├── src/                            # Contratos (enums, request, response, formatos) y prompts
└── requirements.txt
```

## Instrucciones de instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/technobonal/New-MindFMAgent.git
cd New-MindFMAgent
```

### 2. Crear y activar el entorno virtual
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Crear un archivo `.env` en la raíz del proyecto:
```env
GROQ_API_KEY=tu_clave

# Embeddings: "cohere" o "hf"
EMBEDDINGS_PROVIDER=cohere
COHERE_API_KEY=tu_clave
# HUGGINGFACE_API_KEY=tu_clave   # solo si EMBEDDINGS_PROVIDER=hf

# Desarrollo local contra OCI (usa tu ~/.oci/config)
OCI_LOCAL_DEV=1

# Opcionales
NUEVAMENTE_MAX_CORRIDAS_DIA=10
NUEVAMENTE_MAX_CORRIDAS_GLOBAL_DIA=100
```

Para el login de Google en local, crear `.streamlit/secrets.toml` con las secciones `[auth]` (`redirect_uri`, `cookie_secret`) y `[auth.google]` (`client_id`, `client_secret`, `server_metadata_url`). Este archivo no se sube al repositorio.

### 5. Preparar OCI Object Storage
Crear un bucket (este proyecto usa `nuevamente-storage`) sin acceso público y configurar `~/.oci/config` con tu usuario IAM. El servidor MCP usa los prefijos `fuentes/` para documentos de entrada y `generados/formateados/` para los resultados.

### 6. Correr la aplicación
```bash
streamlit run app.py
```

## Autenticación contra OCI según el entorno

El servidor MCP elige el método de autenticación en este orden:

| Entorno | Método | Cómo se activa |
|---|---|---|
| Desarrollo local | API key de `~/.oci/config` | `OCI_LOCAL_DEV=1` |
| Streamlit Community Cloud | API key de un usuario IAM dedicado | Variables `OCI_USER_OCID`, `OCI_TENANCY_OCID`, `OCI_FINGERPRINT`, `OCI_REGION`, `OCI_PRIVATE_KEY` |
| Instancia OCI | Instance Principal (sin claves) | Ninguna de las anteriores |

> `OCI_LOCAL_DEV` **no** debe definirse en la nube.

## Despliegue

### Streamlit Community Cloud
La aplicación está desplegada en:
👉 [nuevamente-app.streamlit.app](https://nuevamente-app.streamlit.app)

Configuración usada:
- Repositorio `technobonal/New-MindFMAgent`, rama `master`, archivo principal `app.py`.
- Secrets en el panel de la app: las variables simples (`GROQ_API_KEY`, `EMBEDDINGS_PROVIDER`, `COHERE_API_KEY`, `OCI_*`, cuotas) van **antes** de cualquier tabla, para que se expongan como variables de entorno; después `[auth]` y `[auth.google]`.
- Redirect URI de Google: `https://nuevamente-app.streamlit.app/oauth2callback`.

> ⚠️ El disco de Community Cloud es efímero: `chroma_db_*`, `checkpoints.sqlite` y `cuotas.sqlite` se reinician cuando la app reinicia. Los documentos y resultados no se pierden porque viven en OCI Object Storage.

### Oracle Cloud Infrastructure (planificado)
El despliegue en una instancia propia de OCI (Nginx como reverse proxy, HTTPS y Instance Principal) está planificado para una etapa posterior.

## Pruebas

Scripts de verificación incluidos en el repositorio:

```bash
python test_grafo.py              # estructura del grafo
python test_grafo_ejecucion.py    # ejecución de punta a punta
python test_grafo_resume.py       # interrupts y reanudación con checkpointer
python test_mcp_aislado.py        # servidor MCP aislado
python servidor_objeStorageOracle.py --test   # tools de OCI contra el bucket real
```

## Uso de Git

```bash
git status
git add .
git commit -m "mensaje descriptivo del cambio"
git push origin master
git log --oneline
```

## Historial de commits

| Commit | Fecha | Descripción |
|---|---|---|
| `9a5e590` | 2026-09-29 | Inicializa repo con .gitignore |
| `33432f1` | 2026-09-29 | Estado inicial: servidor MCP, cliente, grafo, agentes, app.py |
| `f68c23a` | 2026-09-29 | Cachea cliente y namespace de OCI para evitar re-autenticación en cada tool |
| `9c0b38a` | 2026-09-29 | Corrige nombre de modelo Qwen: qwen3.6-27b → qwen3.8-27b (verificado contra límites reales de la cuenta) |
| `4a6a31a` | 2026-09-30 | Integra RAG (Chroma) y cablea agentes reales en el grafo y la UI |
| `5b3e6b9` | 2026-10-01 | Rate limiter con soporte TPD + fail-fast en `llm_client` |
| `003c4ee` | 2026-10-01 | Redactor: mejor logging y manejo de fallos de structured output |
| `dd1db2d` | 2026-10-01 | Ajustes en grafo, supervisor, modificador y contratos |
| `405003e` | 2026-10-01 | Agrega agente ilustrador y scripts de prueba de embeddings/chunks |
| `a896afe` | 2026-10-01 | Corrige extensión de `Redactor_pedagogico.py` |
| `9fbc11b` | 2026-10-07 | Contratos: especificación pedagógica y parámetros del redactor |
| `fb949a8` | 2026-10-07 | Agente redactor pedagógico y su prompt |
| `f3d81f4` | 2026-10-07 | Integra el redactor en grafo y supervisor |
| `3840fe8` | 2026-10-07 | UI: render de documento y conexión en app |
| `92aab4a` | 2026-10-12 | Ignora secretos, bases vectoriales y sqlite locales |
| `d02d41f` | 2026-10-12 | Seguridad: salida de red protegida, validadores y permisos por etapa |
| `8b9ca56` | 2026-10-12 | RAG: embeddings Cohere, Chroma por proveedor y umbrales por proveedor |
| `db70324` | 2026-10-12 | Login con Google, aislamiento por usuario y cuotas diarias |
| `71e6b16` | 2026-10-12 | Flashcards: un ícono por card para cualquier cantidad |
| `be1ba7f` | 2026-10-12 | Unifica versión de Streamlit (>=1.42) |
| `a37a5c2` | 2026-10-12 | Restaura tildes de los comentarios de `requirements.txt` |
| `f1d3f7c` | 2026-10-12 | Ajuste menor en `agente_critico_revisor.py` |
| `b7c1793` | 2026-10-13 | Aumenta el número de corridas por día a 10 |
| `4934866` | 2026-10-13 | Auth de OCI por API key vía variables de entorno y `sys.executable` en el cliente MCP |
| `eea6a75` | 2026-10-13 | Ignora archivos `.pem` |
| `e77efa4` | 2026-10-13 | Agrega carpeta Dev Container |

## Autor

Franklin Bonalde · [@technobonal](https://github.com/technobonal)