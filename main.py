import logging, torch
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)


# ──────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
COLLECTION    = "knowledge_base"
DATA_DIR      = ROOT / "data"
EMB_MODEL_ID  = "intfloat/multilingual-e5-large-instruct"
LLM_MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.1"


# ──────────────────────────────────────────────────────────────────────────────
# QDrant client
# ──────────────────────────────────────────────────────────────────────────────
def get_qdrant() -> QdrantVectorStore:
    client = QdrantClient(path=str(ROOT / "qdrant"))
    store  = QdrantVectorStore(
        client          = client,
        collection_name = COLLECTION,
        embedding       = HuggingFaceEmbeddings(model_name=EMB_MODEL_ID),
    )
    return store

# ──────────────────────────────────────────────────────────────────────────────
# LLM pipeline
# ──────────────────────────────────────────────────────────────────────────────
def get_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # <– direkt anpassen
    )

    tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    mod = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, device_map="auto",
        quantization_config=bnb_config, torch_dtype=torch.float16
    )
    pipe = pipeline(
        "text-generation", model=mod, tokenizer=tok,
        return_full_text=False, max_new_tokens=512,
        temperature=0.7, top_p=0.9,
        pad_token_id=tok.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)

PROMPT = PromptTemplate.from_template(
    """<s>[INST]You are a helpful, knowledgeable assistant.
        1. Use context to answer questions
        2. If unsure, say you don't know
        3. Respond concisely (1-2 sentences)
        4. Be honest and ethical

Context:
{context}

Frage:
{question}[/INST]"""
)


def pretty_sources(docs):
    return [
        f"{doc.page_content[:120].replace('\n', ' ')}…  [{doc.metadata['source']}]"
        for doc in docs
    ]


# ──────────────────────────────────────────────────────────────────────────────
# RAG chain
# ──────────────────────────────────────────────────────────────────────────────
def build_chain():
    vector_store = get_qdrant()
    retriever    = vector_store.as_retriever(search_kwargs={"k": 4})
    llm          = get_llm()

    def enrich_with_context(d):
        docs = retriever.invoke(d["question"])
        return {
            "question": d["question"],
            "docs": docs,
            "context": "\n\n".join(doc.page_content for doc in docs)
        }
    retrieval = RunnableLambda(enrich_with_context)
    
    rag_chain = retrieval | RunnableParallel(
        answer = PROMPT | llm,
        sources = lambda d: pretty_sources(d["docs"]),
    )
    return rag_chain


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    chain = build_chain()

    print("Hi (':q' zum Beenden)")

    while True:
        question = input("\n> ").strip()
        if question == ":q":
            break

        result = chain.invoke({"question": question})
        print("\n=== Response ===")
        print(result["answer"])

        print("\n=== Sources ===")
        for i, src in enumerate(result["sources"], 1):
            print(f"{i}. {src}")
