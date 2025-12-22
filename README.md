# HRS
Health Recommendation System

A Flask-based web application that provides AI-powered health recommendations based on patient symptoms. The application supports multiple LLM providers (Google Gemini and OpenAI GPT) and uses RAG (Retrieval-Augmented Generation) to provide more accurate, context-aware medical recommendations by leveraging a curated medical knowledge base. It runs using Gunicorn as the WSGI server and can be deployed as a Docker container.

## Features

- Clean and responsive web interface
- Patient symptoms input form with multiple fields (symptoms, duration, severity, additional info)
- **RAG (Retrieval-Augmented Generation)**: Enhanced recommendations using a medical knowledge base
- **Multiple LLM provider support**: Choose between Google Gemini (default) or OpenAI GPT
- Flexible provider configuration through environment variables
- AI-powered health recommendations with configurable providers
- Context-aware responses using vector-based document retrieval
- Detailed output screen showing symptoms summary and AI-generated recommendations
- Medical disclaimer for user safety
- Production-ready with Gunicorn WSGI server
- Dockerized for easy deployment

## Prerequisites

- Python 3.13 or higher
- Conda (recommended) or pip for package management
  - **Conda**: Install from https://docs.conda.io/en/latest/miniconda.html
  - **pip**: Comes with Python
- An API key from one of the supported providers:
  - **Gemini API key** (recommended, default provider): Get from https://makersuite.google.com/app/apikey
  - **OpenAI API key** (optional): Get from https://platform.openai.com/api-keys
- Docker (for containerized deployment)

## Running Locally

### Using Conda (Recommended)

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate hrs
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and configure your preferred LLM provider:
# - Set LLM_PROVIDER to either 'gemini' (default) or 'openai'
# - Add the corresponding API key (GEMINI_API_KEY or OPENAI_API_KEY)
```

3. Run with Flask development server:
```bash
python app.py
```

4. Run with Gunicorn (Linux only):
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

5. Open your browser and navigate to `http://localhost:5000`

### Using pip

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and configure your preferred LLM provider:
# - Set LLM_PROVIDER to either 'gemini' (default) or 'openai'
# - Add the corresponding API key (GEMINI_API_KEY or OPENAI_API_KEY)
```

3. Run with Flask development server:
```bash
python app.py
```

4. Run with Gunicorn (Linux only):
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

5. Open your browser and navigate to `http://localhost:5000`

## Running with Docker

### Using Conda-based Docker Image (Recommended)

1. Build the Docker image:
```bash
docker build -f Dockerfile.conda -t hrs-flask-app .
```

2. Run the container with your LLM provider configuration:
```bash
# Using Gemini (default)
docker run -d -p 8080:5000 -e GEMINI_API_KEY=your_api_key_here --name hrs hrs-flask-app

# Or using OpenAI
docker run -d -p 8080:5000 -e LLM_PROVIDER=openai -e OPENAI_API_KEY=your_api_key_here --name hrs hrs-flask-app
```

3. Open your browser and navigate to `http://localhost:8080`

4. Stop the container:
```bash
docker stop hrs
docker rm hrs
```

### Using pip-based Docker Image

1. Build the Docker image:
```bash
docker build -t hrs-flask-app .
```

2. Run the container with your LLM provider configuration:
```bash
# Using Gemini (default)
docker run -d -p 8080:5000 -e GEMINI_API_KEY=your_api_key_here --name hrs hrs-flask-app

# Or using OpenAI
docker run -d -p 8080:5000 -e LLM_PROVIDER=openai -e OPENAI_API_KEY=your_api_key_here --name hrs hrs-flask-app
```

3. Open your browser and navigate to `http://localhost:8080`

4. Stop the container:
```bash
docker stop hrs
docker rm hrs
```

## Project Structure

```
HRS/
├── app.py                  # Flask web application (routes and web logic)
├── LLM/                    # LLM module containing all LLM-related functionality
│   ├── __init__.py         # Module initialization and exports
│   ├── llm_service.py     # LLM service module (provider factory, integration, prompts)
│   ├── llm_provider.py    # Abstract base class for LLM providers
│   ├── openai_provider.py # OpenAI provider implementation
│   ├── gemini_provider.py # Google Gemini provider implementation
│   └── llm_constants.py   # Shared constants and message templates
├── RAG/                   # RAG module for Retrieval-Augmented Generation
│   ├── __init__.py        # RAG module initialization and exports
│   ├── rag_service.py     # RAG service for document retrieval and context augmentation
│   └── medical_knowledge.py # Medical knowledge base for RAG
├── templates/
│   ├── index.html         # Patient symptoms input form
│   └── output.html        # AI recommendations display page
├── tests/                 # Unit tests
│   ├── __init__.py        # Tests module initialization
│   ├── test_llm_service.py # Unit tests for LLM service
│   ├── test_rag_service.py # Unit tests for RAG service
│   └── test_medical_knowledge.py # Unit tests for medical knowledge base
├── requirements.txt       # Python dependencies (pip)
├── environment.yml        # Conda environment specification
├── .env.example          # Environment variables template
├── Dockerfile            # Docker configuration (pip-based)
├── Dockerfile.conda      # Docker configuration (conda-based)
├── .dockerignore        # Docker ignore file
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## LLM Provider Configuration

The application supports multiple LLM providers with easy configuration:

### Supported Providers

1. **Google Gemini** (Default)
   - Model: `gemini-2.5-flash`
   - Configuration: Set `LLM_PROVIDER=gemini` (or omit, as it's the default)
   - API Key: `GEMINI_API_KEY`

2. **OpenAI GPT**
   - Model: `gpt-3.5-turbo`
   - Configuration: Set `LLM_PROVIDER=openai`
   - API Key: `OPENAI_API_KEY`

### Configuration Steps

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your preferred provider:
   ```bash
   # For Gemini (default)
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # For OpenAI
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. The application will automatically use the configured provider

### Demo Mode

If no API key is configured, the application runs in demo mode, showing example responses instead of real AI-generated recommendations.

## RAG (Retrieval-Augmented Generation) Configuration

The application uses RAG to enhance health recommendations with relevant medical knowledge from a curated knowledge base.

### How RAG Works

1. **Medical Knowledge Base**: Contains curated medical information about common conditions, symptoms, diagnostic tests, and treatment approaches
2. **Vector Embeddings**: Documents are converted to numerical vectors using sentence-transformers
3. **Semantic Search**: When a user submits symptoms, the system retrieves the most relevant medical documents
4. **Context Augmentation**: Retrieved information is added to the LLM prompt for more accurate recommendations

### RAG Configuration

RAG is enabled by default. You can configure it in your `.env` file:

```bash
# Enable/disable RAG functionality (default: true)
RAG_ENABLED=true

# ChromaDB storage path (default: ./chromadb_data)
CHROMADB_PATH=./chromadb_data

# Embedding model (default: all-MiniLM-L6-v2)
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### RAG Features

- **Automatic Initialization**: Medical knowledge base is automatically indexed on first run
- **Graceful Fallback**: If RAG fails to initialize, the system continues to work normally without RAG
- **Semantic Retrieval**: Uses vector similarity to find the most relevant medical information
- **Context-Aware Responses**: LLM receives both the user query and relevant medical context

### Embedding Models

The system supports various sentence-transformer models:
- `all-MiniLM-L6-v2` (default): Fast and efficient, good for most use cases
- `all-mpnet-base-v2`: More accurate but slower
- Any model from the sentence-transformers library

**Note**: The first time RAG initializes, it will download the embedding model from HuggingFace. This requires internet access. In offline environments, RAG will gracefully disable itself and the system will function normally.

## Testing

The project includes comprehensive unit tests using Python's `unittest` framework.

### Running Tests

Run all tests:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run a specific test file:
```bash
python -m unittest tests.test_llm_service -v
```

Run a specific test class:
```bash
python -m unittest tests.test_rag_service.TestRAGService -v
```

### Test Coverage

The test suite includes:
- **test_llm_service.py**: Tests for input sanitization, severity validation, and provider factory
- **test_rag_service.py**: Tests for RAG service initialization, context retrieval, and prompt augmentation
- **test_medical_knowledge.py**: Tests for medical knowledge base structure and content

All tests use mocks where appropriate to avoid external dependencies (ChromaDB, LLM APIs, etc.).

## Usage

1. Open the application in your browser
2. Enter your symptoms in the input form:
   - Describe your symptoms
   - Specify how long you've had them
   - Rate the severity (1-10)
   - Add any additional relevant information (optional)
3. Click "Get Health Recommendations"
4. View AI-generated health recommendations on the output page
5. Click "Enter New Symptoms" to input new symptoms

## Important Disclaimer

This application provides general health information and recommendations generated by AI. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical care.
