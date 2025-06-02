import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from supabase import create_client, Client
from markitdown import MarkItDown, DocumentConverterResult

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
embeddingModel = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)
    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    try:
        embeddings = embeddingModel.encode(text).tolist()
        return embeddings
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 768

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    # Optionally, use get_title_and_summary if you want
    # extracted = await get_title_and_summary(chunk, url)
    # title = extracted['title']
    # summary = extracted['summary']
    title = "title"
    summary = "summary"
    embedding = await get_embedding(chunk)
    metadata = {
        "source": "conflowgen_repo",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": url
    }
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=title,
        summary=summary,
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, rel_path: str, markdown: str):
    chunks = chunk_text(markdown)
    tasks = []
    for i, chunk in enumerate(chunks):
        # Add file and chunk number header
        chunk_with_header = f"# File: {rel_path} | Chunk: {i}\n\n{chunk}"
        tasks.append(process_chunk(chunk_with_header, i, url))
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

def should_skip_file(filename):
    skip_ext = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.ico', '.spp')
    return filename.lower().endswith(skip_ext)

def github_blob_url(base_url: str, rel_path: str):
    rel_path_url = quote(rel_path.replace("\\", "/"))
    return f"{base_url}/blob/main/{rel_path_url}"

def convert_file_to_markdown(full_path):
    converter = MarkItDown()
    ext = os.path.splitext(full_path)[1].lower()

    # Special handling for .ipynb files
    if ext == '.ipynb':
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            md_cells = []
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') in ('markdown', 'code'):
                    src = ''.join(cell.get('source', []))
                    if cell.get('cell_type') == 'code':
                        src = f"```python\n{src}\n```"
                    md_cells.append(src)
            return '\n\n'.join(md_cells)
        except Exception as e:
            print(f"Failed to parse ipynb file {full_path}: {e}")
            return None

    # For other files, try MarkItDown, then fallback to plain text
    try:
        with open(full_path, 'rb') as f:
            result = converter.convert_stream(f)
        if hasattr(result, "text_content"):
            return result.text_content
        else:
            raise Exception("MarkItDown returned no text_content")
    except UnicodeDecodeError:
        # If MarkItDown fails with UnicodeDecodeError, try reading as text
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(full_path, "r", encoding="latin-1") as f:
                return f.read()
    except Exception as e:
        # If anything else fails, try as text
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(full_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                print(f"Failed to convert {full_path} after multiple attempts: {e}")
                return None
        except Exception as e:
            print(f"Failed to convert {full_path}: {e}")
            return None

async def iterate_and_process_files(root_folder: str, github_base_url: str):
    files_to_process = []
    for dirpath, dirs, files in os.walk(root_folder):
        for filename in files:
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_folder)
            if should_skip_file(filename):
                continue
            if os.path.getsize(full_path) == 0:
                continue
            files_to_process.append((rel_path, full_path))
    print(f"Found {len(files_to_process)} files to process.")

    for rel_path, full_path in files_to_process:
        url = github_blob_url(github_base_url, rel_path)
        markdown_content = convert_file_to_markdown(full_path)
        if not markdown_content:
            print(f"Skipping {full_path}: Conversion failed or empty content.")
            continue
        await process_and_store_document(url, rel_path, markdown_content)

async def main():
    root_folder = "conflowgen-repo"
    github_base_url = "https://github.com/1kastner/conflowgen"
    await iterate_and_process_files(root_folder, github_base_url)

if __name__ == "__main__":
    asyncio.run(main())
