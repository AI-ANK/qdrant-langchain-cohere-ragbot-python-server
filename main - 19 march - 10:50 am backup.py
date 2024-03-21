#!/usr/bin/env python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatCohere
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from langserve import add_routes
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import CohereEmbeddings

from fastapi.middleware.cors import CORSMiddleware

os.environ["COHERE_API_KEY"] = "sMF0Zvjvf5Kezq4YGHukV91mRARqfEExA2HiQscX"

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

loader = PyPDFLoader("docs/cover_letter.pdf")
data = loader.load()
print(type(data))
# Concatenate text across all pages
# full_text = ''.join(doc.page_content for doc in data)

print(type(data))

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=100)
all_splits = text_splitter.split_documents(data)
# print("splits:", all_splits[0])
print(len(all_splits))

embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

# Add to vectorDB
qdrant = Qdrant.from_documents(
    all_splits,
    embeddings,
    url=
    "https://cd70b5b7-8cbf-4b34-b706-2a690d044749.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="7T-T4HLAYtThlT-uilLdxtyLjylHWn7USVZ9O5rVZ5sLY6iq0ZB1Gg",
    collection_name="my_documents",
)

retriever = qdrant.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatCohere()

# RAG chain
chain = (RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})
         | prompt
         | model
         | StrOutputParser())


# Add typing for input
class Question(BaseModel):
  __root__: str


chain = chain.with_types(input_type=Question)

# prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    chain,
    path="/rag",
)


@app.get("/")
async def redirect_root_to_docs():
  return RedirectResponse("/docs")


if __name__ == "__main__":
  import uvicorn
  port = int(os.environ.get("PORT", 8000))
  uvicorn.run(app, host="0.0.0.0", port=port)
