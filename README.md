# Enterprise AI Platform

A production-ready AI/ML platform with NLP capabilities, model management, and a Streamlit dashboard.

## Features

### Core AI Capabilities
- **NLP Capabilities**: Sentiment analysis, NER, zero-shot classification, semantic similarity
- **LLM Integration**: OpenAI (GPT-4, GPT-3.5) and Anthropic (Claude) support
- **Advanced RAG**: Multi-query retrieval, cross-encoder re-ranking, hybrid search, HyDE
- **ML Pipeline**: Train, evaluate, and deploy custom models with MLflow tracking
- **Batch Processing**: Process thousands of items asynchronously with job tracking

### Infrastructure
- **Vector Databases**: Qdrant, ChromaDB, and Pinecone support for scalable RAG
- **Caching**: Redis-based response caching for 10x faster repeated queries
- **User Management**: Role-based access control (Admin, Model Manager, Analyst, User)
- **API Keys**: Generate and manage API keys for programmatic access
- **Monitoring**: Prometheus metrics, structured logging, health checks
- **Async Tasks**: Celery workers for background processing

## Quick Start

### Using Docker Compose

```bash
# Clone and navigate to the project
cd enterprise-ai-platform

# Create environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Access the API
curl http://localhost:8000/health
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_platform
export REDIS_URL=redis://localhost:6379/0

# Run migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

### Frontend (Streamlit)

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

## API Endpoints

### Authentication
| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/auth/register` | Register new user |
| `POST /api/v1/auth/login` | Login and get tokens |

### NLP & Predictions
| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/predictions/sentiment` | Sentiment analysis |
| `POST /api/v1/predictions/ner` | Named entity recognition |
| `POST /api/v1/predictions/classify` | Zero-shot classification |
| `POST /api/v1/predictions/similarity` | Semantic similarity |

### LLM & Chat
| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/llm/chat` | Chat completion (GPT-4, Claude) |
| `POST /api/v1/llm/summarize` | Text summarization |
| `POST /api/v1/llm/translate` | Language translation |
| `POST /api/v1/llm/extract` | Structured data extraction |

### Advanced RAG
| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/rag/documents` | Add documents to knowledge base |
| `POST /api/v1/rag/query` | Query with citations & re-ranking |
| `POST /api/v1/rag/search/hybrid` | Hybrid semantic + keyword search |
| `POST /api/v1/rag/search/hyde` | HyDE search |

### Batch Processing
| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/batch/sentiment` | Batch sentiment analysis |
| `POST /api/v1/batch/predictions` | Generic batch predictions |
| `GET /api/v1/batch/jobs/{id}` | Get job status & results |

### Models & Admin
| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/models/` | List models |
| `POST /api/v1/models/{id}/train` | Train a model |
| `GET /api/v1/admin/stats` | System statistics |

## Architecture

```
├── app/
│   ├── ai/              # AI/ML engines
│   ├── api/             # FastAPI routes & middleware
│   ├── config/          # Settings & constants
│   ├── db/              # Database connection & CRUD
│   ├── models/          # SQLAlchemy models
│   ├── schemas/         # Pydantic schemas
│   ├── tasks/           # Celery tasks
│   └── tests/           # Test suite
├── frontend/            # React dashboard
├── kubernetes/          # K8s manifests
└── alembic/             # Database migrations
```

## Environment Variables

See `.env.example` for all configuration options.

## License

MIT
