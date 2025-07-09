import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


def train(data_path: str, vector_store) -> None:
    """Train the model with the data in DATA_PATH."""
    logging.info("Training the model with data from DATA_PATH")

    for file in os.listdir(data_path):
        logging.info(f"Loading content from {file}")
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            raw_text = f.read()

        logging.info("Splitting document into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([raw_text], metadatas=[{"source": file}])

        logging.info(f"Creating embeddings for {len(chunks)} document chunks")
        vector_store.add_documents(chunks)
