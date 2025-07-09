import logging

import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Konfiguration ---
QDRANT_PATH = "./qdrant"
COLLECTION_NAME = "knowlege_base"
DATA_PATH = "./data"
LLM_MODEL_PATH = "./models/llama-2-7b.gguf"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_MODEL_SIZE = 1024
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

def init_qdrant() -> QdrantClient:
    logging.info(f"Initialising QDrant client at {QDRANT_PATH}")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        logging.info(f"Creating collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_MODEL_SIZE, distance=Distance.COSINE),
        )
    return qdrant_client

def init_embedding_model() -> Embeddings:
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    return embedding_model

def init_vector_store(qdrant_client: QdrantClient, embedding_model: Embeddings) -> VectorStore:
    logging.info(f"Initializing vector store")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    return vector_store

def init_llm() -> BaseLLM:
    logging.info(f"Initializing LLM model {LLM_MODEL}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float32
    )

    logging.info(f"Setting up text generation pipeline with model {LLM_MODEL}")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def init_prompt_template() -> PromptTemplate:
    template = """<s>[INST]You are a helpful, knowledgeable assistant.
        Follow these guidelines:
        1. Use context to answer questions
        2. If unsure, say you don't know
        3. Respond concisely (1-2 sentences)
        4. Be honest and ethical

        Context: {context}
        <</SYS>>
        {question} [/INST]"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    return prompt

def main():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    qdrant_client = init_qdrant()
    embedding_model = init_embedding_model()
    vector_store = init_vector_store(qdrant_client, embedding_model)
    retriever = vector_store.as_retriever()

    # Train if collection is empty
    if not qdrant_client.count(collection_name=COLLECTION_NAME).count:
        from train import train
        logging.info("Training vector store with data")
        train(DATA_PATH, vector_store)

    llm = init_llm()
    prompt = init_prompt_template()
    llm_chain = prompt | llm

    print("Hi (quit with ':q'):")

    while True:
        question = input("\n> ").strip()
        if question == ":q":
            break

        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        result = llm_chain.invoke({"context": context, "question": question})

        print("=== Response ===")
        print(result)

        print("=== Sources ===")
        for i, doc in enumerate(docs, 1):
            print(
                f"{i}. {doc.page_content[:80]}{'...' if len(doc.page_content) > 80 else ''} [{doc.metadata['source']}]")


if __name__ == "__main__":
    main()
