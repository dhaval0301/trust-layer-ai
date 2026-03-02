from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

INDEX_PATH = "faiss_index"

def load_retriever():

    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(search_kwargs={"k": 5})