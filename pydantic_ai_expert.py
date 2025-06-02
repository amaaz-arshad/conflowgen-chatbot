from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os

from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize embedding model
embeddingModel = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert at Conflowgen—a Python package for generating synthetic container flows at maritime container terminals, with a focus on yard operations.
You have access to all of the Conflowgen documentation, including background information, usage examples, input distribution references, API methods, and preview/analysis utilities.

Your sole job is to assist the user with Conflowgen-related questions and tasks. Do not answer unrelated questions.
When you need to answer, first look up relevant documentation (using RAG against the Supabase embeddings). Then inspect available Conflowgen documentation pages and retrieve the necessary content.
If you cannot find an answer in the documentation or on the right URL, be honest and tell the user.
"""

# Define tools without decorators
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': int(os.getenv("TOP_K", 5)),  # Default to top 5 matches
                'filter': {'source': 'conflowgen_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is conflowgen_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'conflowgen_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'conflowgen_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector using local model"""
    try:
        return embeddingModel.encode(text).tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 768  # Return zero vector on error

def get_agent(model_name: str) -> Agent:
    """Create agent with specified model and tools"""
    model = OpenAIModel(
        model_name=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL")
    )
    
    # Create tools with proper metadata
    tools = [
        Tool(
            name="retrieve_relevant_documentation",
            description="Retrieve relevant documentation based on user query",
            function=retrieve_relevant_documentation
        ),
        Tool(
            name="list_documentation_pages",
            description="List available documentation pages",
            function=list_documentation_pages
        ),
        Tool(
            name="get_page_content",
            description="Get full content of a documentation page",
            function=get_page_content
        )
    ]
    
    agent = Agent(
        model,
        system_prompt=system_prompt,
        deps_type=PydanticAIDeps,
        tools=tools,
        retries=2
    )
    
    return agent