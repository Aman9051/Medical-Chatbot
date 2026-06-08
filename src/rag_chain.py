import os
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatLlamaCpp

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.prompt import SYSTEM_PROMPT

load_dotenv()


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def load_chain():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatLlamaCpp(
        model_path=r"model\Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        temperature=0.3,
        max_tokens=512,
        n_ctx=4096,
        verbose=False
    )

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain