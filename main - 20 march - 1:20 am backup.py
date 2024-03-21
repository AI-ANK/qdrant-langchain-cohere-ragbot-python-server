#!/usr/bin/env python
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatCohere
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import CohereEmbeddings
import os
from langchain_core.pydantic_v1 import BaseModel
from typing import List
import shutil

os.environ["COHERE_API_KEY"] = "sMF0Zvjvf5Kezq4YGHukV91mRARqfEExA2HiQscX"

# Load environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = "https://cd70b5b7-8cbf-4b34-b706-2a690d044749.us-east4-.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "7T-T4HLAYtThlT-uilLdxtyLjylHWn7USVZ9O5rVZ5sLY6iq0ZB1Gg"

# Global variable to track if the document has been uploaded and processed
document_processed = False

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

# Set CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize embeddings and Qdrant client outside of the route to reuse
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

collection_name = "my_documents"
global qdrant_client
qdrant_client = Qdrant(client, collection_name, embeddings)

# qdrant = Qdrant.from_documents(
#     data,
#     embeddings,
#     url=
#     "https://cd70b5b7-8cbf-4b34-b706-2a690d044749.us-east4-0.gcp.cloud.qdrant.io:6333",
#     api_key="7T-T4HLAYtThlT-uilLdxtyLjylHWn7USVZ9O5rVZ5sLY6iq0ZB1Gg",
#     collection_name="my_documents",
# )


# Dependency for retriever
def get_retriever():
  return qdrant_client.as_retriever()


# Dependency for chat chain
def get_chain(retriever=Depends(get_retriever)):
  template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
  prompt = ChatPromptTemplate.from_template(template)
  model = ChatCohere(api_key=COHERE_API_KEY)
  chain = RunnableParallel({
      "context": retriever,
      "question": RunnablePassthrough()
  }) | prompt | model | StrOutputParser()
  return chain


# Models
class Question(BaseModel):
  __root__: str


# Routes
@app.post("/upload", status_code=200)
async def upload_document(file: UploadFile = File(...)):
  global document_processed
  # Save the uploaded file to disk
  with open(file.filename, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)

  # Process the uploaded document
  loader = PyPDFLoader(file.filename)
  data = loader.load()
  # full_text = ''.join(doc.page_content for doc in data)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=100)
  all_splits = text_splitter.split_documents(data)
  print("allsplits", all_splits)

  # Update embeddings in the vector DB
  # Note: This is a simplified example. You should manage the document IDs and updates properly.
  # qdrant_client.update_embeddings(all_splits, embeddings)
  global qdrant_client
  qdrant_client = Qdrant.from_documents(
      all_splits,
      embeddings,
      url=
      "https://cd70b5b7-8cbf-4b34-b706-2a690d044749.us-east4-0.gcp.cloud.qdrant.io:6333",
      api_key="7T-T4HLAYtThlT-uilLdxtyLjylHWn7USVZ9O5rVZ5sLY6iq0ZB1Gg",
      collection_name="my_documents",
  )
  document_processed = True
  return {"filename": file.filename}


class ChatResponse(BaseModel):
  response: str


print(QDRANT_URL)
print(QDRANT_API_KEY)


@app.post("/rag", response_model=ChatResponse)
async def rag_endpoint(question: Question, chain=Depends(get_chain)):
  if not document_processed:
    raise HTTPException(status_code=400,
                        detail="Document not uploaded and processed yet")

  result = chain.invoke(question.__root__)  # Pass the string directly
  return {"response": result}


@app.get("/")
async def redirect_root_to_docs():
  return RedirectResponse("/docs")


# Run the server
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
