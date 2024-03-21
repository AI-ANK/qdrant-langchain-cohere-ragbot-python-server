# A Simple Full Stack RAG-bot for Enterprises using React, Qdrant, Langchain, Cohere and FastAPI

This backend API server is a core component of an AI-powered document chat application, designed to interpret and respond to user queries based on the content of uploaded documents. Leveraging the capabilities of LangChain, Cohere, and Qdrant, it offers a robust and scalable solution for semantic document processing.

## Features
- **Multitenancy Support**: Efficiently handles multiple users by segregating their data using unique group IDs.
- **Binary Quantization**: Utilizes Qdrant's binary quantization for optimal storage and fast retrieval of embeddings.
- **Natural Language Processing**: Empowered by Cohere's LLM, it provides accurate and context-aware responses.
- **Scalable Vector Search**: Integrates with Qdrant for scalable and efficient vector search capabilities.

## How to Use
1. **Set Up Your Environment**:
   - Ensure Python is installed.
   - Clone the repository and navigate to the project directory.
2. **Configuration**:
   - Set environment variables for Cohere API key, Qdrant URL, and Qdrant API key in a `.env` file.
3. **Installation**:
   ```bash
    pip install -r requirements.txt  ```
4. **Start the Server**:
   ```
   python main.py
   ```

