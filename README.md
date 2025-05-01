# Open Textbook AI Chat Assistant

An intelligent chatbot system designed to help users find and learn about open textbooks. Built with Chainlit, LangChain, and Groq LLM, this assistant provides detailed information about textbooks including their subjects, links, and publication dates.

## Features

- 🤖 Intelligent conversational interface powered by Llama 3.3 70B
- 📚 Comprehensive textbook information retrieval
- 🔍 Semantic search capabilities using FAISS
- 💾 Redis caching for improved performance
- 🐳 Docker support for easy deployment
- 🔄 Conversation memory for contextual responses

## Prerequisites

- Python 3.8+
- Redis server
- Groq API key
- Docker (optional)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the root directory with:

```
GROQ_API_KEY=your_groq_api_key
```

## Data Setup

1. Prepare your textbook data CSV files in the `data/` directory:

   - textbook_details_no_dup.csv
   - textbook_details_with_disciplines.csv
   - OpenTextbookLibrary.csv

2. Generate the vector store:

```bash
python ingest.py
```

## Running the Application

### Local Development

1. Start Redis server:

```bash
redis-server
```

2. Run the Chainlit application:

```bash
chainlit run model.py
```

The application will be available at `http://localhost:8000`

### Docker Deployment

1. Build and start the containers:

```bash
docker-compose up --build
```

## Usage

1. Open your web browser and navigate to `http://localhost:8000`
2. Start chatting with the assistant
3. Ask questions about textbooks, subjects, or specific topics
4. The assistant will provide detailed responses including book information, links, and relevant metadata

## Project Structure

```
.
├── model.py              # Main application logic
├── ingest.py            # Data ingestion and vector store creation
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── data/              # CSV data files
└── vectorstore/       # FAISS vector store
```

## Technical Details

- **Embeddings**: Uses HuggingFace's 'sentence-transformers/all-MiniLM-L6-v2' model
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: Groq's Llama 3.3 70B model
- **Caching**: Redis with 1-hour expiration
- **UI Framework**: Chainlit

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
