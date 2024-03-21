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
   ```
    pip install -r requirements.txt
   ```
4. **Start the Server**:
   ```
   python main.py
   ```
This will launch the FastAPI server, which will be accessible locally.

## Tools and Technologies
- LLM Orchestration: Langchain
- Vector DB and Similarity Search: Qdrant
- API Framework: FastAPI
- LLM Integration: Cohere API

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For support, please open an issue in the GitHub issue tracker.

## Live Demo
Try the demo [here](https://qdrant-langchain-cohere-ragbot-ui.vercel.app/)

## Author
Developed by [Harshad Suryawanshi](https://www.linkedin.com/in/harshadsuryawanshi/)
If you find this project useful, consider giving it a ⭐ on GitHub!
