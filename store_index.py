# Run once to upload PDF chunks into Pinecone.

import os

from dotenv import load_dotenv

from pinecone import Pinecone
from pinecone import ServerlessSpec

from langchain_pinecone import PineconeVectorStore

from src.helper import load_pdf
from src.helper import split_text
from src.helper import get_embeddings

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

documents = load_pdf(
    "data/medical_book.pdf"
)

chunks = split_text(documents)

embeddings = get_embeddings()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

if INDEX_NAME not in pc.list_indexes().names():

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Medical book indexed successfully.")