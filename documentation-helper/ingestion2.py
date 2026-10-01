# copied ingestion.py, changed the site limiting to only few URLs and processing the chunks sequentially instead of all together in parallel.
# Still didn't worked due to gemini API credits exhausted
import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
#from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
#from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap



from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


#embeddings = OpenAIEmbeddings(    model="text-embedding-3-small",  show_progress_bar=False,  chunk_size=50, retry_min_seconds=10)
embeddings = GoogleGenerativeAIEmbeddings( model="models/gemini-embedding-001", batch_size=50)

#Incase if you want to use Chroma DB, below statement to be used 
#vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

vectorstore = PineconeVectorStore(     index_name="langchain-docs-2025", embedding=embeddings )
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(
    documents: List[Document],
    batch_size: int = 100,
):
    """Process documents sequentially to avoid Gemini API rate limits."""

    log_header("VECTOR STORAGE PHASE")

    log_info(
        f"📚 VectorStore Indexing: Preparing to add "
        f"{len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into "
        f"{len(batches)} batches of {batch_size} documents each"
    )

    successful = 0

    # ---------------------------------------------------------
    # IMPORTANT:
    # Process batches ONE AT A TIME
    # ---------------------------------------------------------
    for batch_num, batch in enumerate(batches, start=1):

        try:

            log_info(
                f"🔄 Processing batch "
                f"{batch_num}/{len(batches)} "
                f"({len(batch)} documents)"
            )

            await vectorstore.aadd_documents(batch)

            successful += 1

            log_success(
                f"VectorStore Indexing: Successfully added batch "
                f"{batch_num}/{len(batches)} "
                f"({len(batch)} documents)"
            )

            # Small delay between Gemini API operations
            if batch_num < len(batches):
                log_info(
                    "⏳ Waiting 5 seconds before next batch..."
                )

                await asyncio.sleep(5)

        except Exception as e:

            error_message = str(e)

            log_error(
                f"VectorStore Indexing: Failed to add batch "
                f"{batch_num} - {error_message}"
            )

            # Stop immediately when Gemini quota/rate limit is reached
            if (
                "RESOURCE_EXHAUSTED" in error_message
                or "429" in error_message
            ):
                log_error(
                    "❌ Gemini API quota/rate limit reached. "
                    "Stopping ingestion."
                )
                break

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    if successful == len(batches):

        log_success(
            f"VectorStore Indexing: All batches processed successfully! "
            f"({successful}/{len(batches)})"
        )

    else:

        log_warning(
            f"VectorStore Indexing: Processed "
            f"{successful}/{len(batches)} batches successfully"
        )

async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    # Crawl the documentation site
    res = tavily_crawl.invoke(
        {
            "url": "https://en.wikipedia.org/wiki/Tirumala",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # ---------------------------------------------------------
    # Take only the first 3 URLs from the crawl results
    # ---------------------------------------------------------
    crawl_results = res["results"][:3]

    log_info(
        f"📄 TavilyCrawl: Selected first {len(crawl_results)} URLs "
        f"out of {len(res['results'])} crawled URLs"
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []

    for tavily_crawl_result_item in crawl_results:

        url = tavily_crawl_result_item["url"]
        raw_content = tavily_crawl_result_item.get("raw_content")

        log_info(
            f"TavilyCrawl: Processing {url}"
        )

        # Protect against missing content
        if not raw_content:
            log_warning(
                f"⚠️ Skipping URL because no content was returned: {url}"
            )
            continue

        all_docs.append(
            Document(
                page_content=raw_content,
                metadata={"source": url},
            )
        )

    # ---------------------------------------------------------
    # Split documents into chunks
    # ---------------------------------------------------------
    log_header("DOCUMENT CHUNKING PHASE")

    log_info(
        f"✂️ Text Splitter: Processing {len(all_docs)} documents "
        f"with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )

    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks "
        f"from {len(all_docs)} documents"
    )

    # ---------------------------------------------------------
    # Process documents sequentially
    # ---------------------------------------------------------
    await index_documents_async(
        splitted_docs,
        batch_size=100,
    )

    log_header("PIPELINE COMPLETE")

    log_success(
        "🎉 Documentation ingestion pipeline finished successfully!"
    )

    log_info("📊 Summary:", Colors.BOLD)

    log_info(
        f"   • URLs selected: {len(crawl_results)}"
    )

    log_info(
        f"   • Documents extracted: {len(all_docs)}"
    )

    log_info(
        f"   • Chunks created: {len(splitted_docs)}"
    )


if __name__ == "__main__":
    asyncio.run(main())