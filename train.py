import logging
from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ──────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("./data")
QDRANT_PATH     = "./qdrant"
COLLECTION_NAME = "knowledge_base"

EMBED_MODEL_ID  = "intfloat/multilingual-e5-large-instruct"
EMBED_DIM       = 1024

CHUNK_SIZE      = 1_000
CHUNK_OVERLAP   = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Qdrant‑Collection anlegen
# ──────────────────────────────────────────────────────────────────────────────
client = QdrantClient(path=QDRANT_PATH)
if not client.collection_exists(COLLECTION_NAME):
    logging.info("Erstelle Qdrant-Collection »%s« …", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID),
)

# ──────────────────────────────────────────────────────────────────────────────
# Dokumente laden
# ──────────────────────────────────────────────────────────────────────────────
logging.info("Lade Dokumente aus %s …", DATA_DIR)

loaders = [
    DirectoryLoader(str(DATA_DIR), glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
    DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

if not docs:
    logging.warning("Keine Dokumente gefunden - Abbruch.")
    exit(0)

# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks   = splitter.split_documents(docs)

# ──────────────────────────────────────────────────────────────────────────────
# Metadaten anreichern
# ──────────────────────────────────────────────────────────────────────────────
for chunk in chunks:
    # Ursprungsdatei
    source_file = Path(chunk.metadata.get("source", "unknown")).name

    chunk.metadata.update(
        {
            "source":  source_file,
            "snippet": chunk.page_content[:120].replace("\n", " "),
            "page":    chunk.metadata.get("page", 1),
        }
    )

# ──────────────────────────────────────────────────────────────────────────────
# In den Vektor‑Store schreiben
# ──────────────────────────────────────────────────────────────────────────────
logging.info("Schreibe %s Chunks in Qdrant …", len(chunks))
store.add_documents(chunks)
logging.info("Fertig.")
