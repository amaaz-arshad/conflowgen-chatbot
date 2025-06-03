# ConFlowGen: Documentation Crawler and RAG Agent

An intelligent documentation crawler and RAG (Retrieval-Augmented Generation) agent built using Pydantic AI and Supabase. The agent can crawl documentation websites, store content in a vector database, and provide intelligent answers to user questions by retrieving and analyzing relevant documentation chunks.

## Features

- Documentation website crawling and chunking
- GitHub repository content processing
- Conversation history storage and retrieval
- Vector database storage with Supabase
- Semantic search using sentence-transformers embeddings
- RAG-based question answering with DeepSeek models
- Streamlit UI with conversation management
- Support for code block preservation in chunks
- Available as both API endpoint and web interface

## Prerequisites

- Python 3.11+
- Supabase account and database
- OpenAI API key
- Streamlit (for web interface)

## Installation

1. Clone the repository

2. Install dependencies (recommended to use a Python virtual environment):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
   - Rename `.env.example` to `.env`
   - Edit `.env` with your API keys and preferences:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   LLM_MODEL=deepseek/deepseek-chat  # or your preferred model
   EMBEDDING_MODEL=sentence-transformers/multi-qa-mpnet-base-dot-v1 # or your preferred model
   TOP_K=5 # top 5 most relevant documents retrieved by the search result
   ```

## Usage

### Database Setup

Execute the SQL commands in `site_pages.sql` to:
1. Create the necessary tables
2. Enable vector similarity search
3. Set up Row Level Security policies

In Supabase, do this by going to the "SQL Editor" tab and pasting in the SQL into the editor there. Then click "Run".

### Crawl Documentation

To crawl and store documentation in the vector database:

```bash
python crawl_conflowgen_docs.py
```

### Crawl GitHub Repository

To crawl and store repository content (markdown files, notebooks, etc.):

```bash
python crawl_conflowgen_github.py
```

This will:
1. Fetch URLs from the documentation sitemap
2. Crawl each page and split into chunks
3. Generate embeddings and store in Supabase

Note: Before running the above command, make sure to place the conflowgen github repo at the root level of this project.

### Streamlit Web Interface

For an interactive web interface to query the documentation:

```bash
streamlit run streamlit_ui.py
```

The interface will be available at `http://localhost:8501`

## Configuration

### Database Schema

The Supabase database uses the following schema:
```sql
CREATE TABLE site_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT,
    chunk_number INTEGER,
    title TEXT,
    summary TEXT,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)
);
```

### Conversation System

The system maintains conversation history with:
- Persistent storage in Supabase
- Automatic title generation from first message
- Timestamp tracking
- Message serialization/deserialization

Execute `conversations.sql` to set up the conversation table before using the Streamlit UI.

### Chunking Configuration

You can configure chunking parameters in `crawl_conflowgen_docs.py`:
```python
chunk_size = 5000  # Characters per chunk
```

The chunker intelligently preserves:
- Code blocks
- Paragraph boundaries
- Sentence boundaries

## Project Structure

- `crawl_conflowgen_docs.py`: Documentation website crawler
- `crawl_conflowgen_github.py`: GitHub repo content processor  
- `pydantic_ai_rag.py`: RAG agent with tool integrations
- `streamlit_ui.py`: Interactive web interface with chat history
- `site_pages.sql`: Vector storage table setup
- `conversations.sql`: Chat history table setup  
- `.env.sample`: Environment configuration template
- `requirements.txt`: Python dependencies

## Error Handling

The system includes robust error handling for:
- Network failures during crawling
- Content conversion errors
- API rate limits and timeouts  
- Database connection issues
- Embedding generation failures
- Invalid URLs or content
- Conversation serialization errors
- Vector search failures
