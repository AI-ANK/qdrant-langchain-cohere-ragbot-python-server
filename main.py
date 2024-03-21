#!/usr/bin/env python
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

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
from fastapi import Query, Response, Cookie  # Import Query for optional query parameters
from qdrant_client.http import models
import secrets

# Load environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

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
#     "",
#     api_key="",
#     collection_name="my_documents",
# )

# Dependency for retriever
# def get_retriever(group_id: str = "default_user"):
#   return qdrant_client.as_retriever(
#       search_kwargs={'filter': {
#           'group_id': group_id
#       }})


def get_retriever(group_id: str):
  return qdrant_client.as_retriever(
      search_kwargs={'filter': {
          'group_id': group_id
      }})

  # Dependency for chat chain


from langchain.chains import RetrievalQA
from langchain.chains import StuffDocumentsChain, LLMChain


def get_chain(retriever=Depends(get_retriever)):
  template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
  prompt = ChatPromptTemplate.from_template(template)
  model = ChatCohere(api_key=COHERE_API_KEY)
  output_parser = StrOutputParser()

  # Create the LLMChain
  llm_chain = LLMChain(llm=model, prompt=prompt, output_parser=output_parser)

  # Create the StuffDocumentsChain
  stuff_documents_chain = StuffDocumentsChain(llm_chain=llm_chain,
                                              document_variable_name="context")

  # Create the RetrievalQA chain
  chain = RetrievalQA(combine_documents_chain=stuff_documents_chain,
                      retriever=retriever)

  return chain


# Models
class Question(BaseModel):
  __root__: str


def generate_group_id():
  return secrets.token_urlsafe(8)


# Routes
@app.post("/upload", status_code=200)
async def upload_document(response: Response, file: UploadFile = File(...)):
  group_id = generate_group_id()
  # response.set_cookie(key="group_id", value=group_id, httponly=True)
  global document_processed
  # Save the uploaded file to disk
  with open(file.filename, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)

  loader = PyPDFLoader(file.filename)
  data = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=100)
  all_splits = text_splitter.split_documents(data)

  # Add group_id to each document's payload
  for doc in all_splits:
    doc.metadata['group_id'] = group_id

  global qdrant_client
  qdrant_client = Qdrant.from_documents(
      all_splits,
      embeddings,
      url=QDRANT_URL,
      api_key=QDRANT_API_KEY,
      collection_name="my_documents",
      metadata_payload_key="metadata",
      quantization_config=models.BinaryQuantization(
          binary=models.BinaryQuantizationConfig(always_ram=True, ), ),
  )

  document_processed = True
  return {"filename": file.filename, "group_id": group_id}


class ChatResponse(BaseModel):
  response: str



@app.post("/rag", response_model=ChatResponse)
async def rag_endpoint(question: Question, group_id: str):
  if group_id is None:
    raise HTTPException(status_code=403,
                        detail="Please upload a document first.")

  retriever = get_retriever(group_id)
  chain = get_chain(retriever)

  # Print the retrieved context
  retrieved_docs = chain.retriever.get_relevant_documents(question.__root__)
  print("Retrieved Documents:")
  for doc in retrieved_docs:
    print(doc.page_content)

  # # Combine the retrieved documents into a single string
  # combined_docs = chain.combine_documents_chain.run(
  #     input_documents=retrieved_docs, question=question.__root__)
  # print("Combined Documents:")
  # print(combined_docs)

  # # Get the context used for generating the response
  # context_used = chain.combine_documents_chain.input_documents
  # print("Context used for generating the response:")
  # for doc in context_used:
  #   print(doc.page_content)

  result = chain.invoke(question.__root__)

  return {"response": result['result']}


@app.get("/")
async def redirect_root_to_docs():
  return RedirectResponse("/docs")


# Run the server
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
