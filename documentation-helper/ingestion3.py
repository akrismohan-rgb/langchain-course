import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_tavily import TavilyCrawl

from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# SSL CONFIGURATION
# ============================================================

ssl_context = ssl.create_default_context(
    cafile=certifi.where()
)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ============================================================
# LOCAL OLLAMA EMBEDDINGS
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "mxbai-embed-large"

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)


# ============================================================
# LOCAL CHROMA VECTOR STORE
# ============================================================

CHROMA_PERSIST_DIRECTORY = "./chroma_db"

vectorstore = Chroma(
    collection_name="documentation",
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_function=embeddings,
)


# ============================================================
# TAVILY
# ============================================================

tavily_crawl = TavilyCrawl()


# ============================================================
# VECTOR STORE INDEXING
# ============================================================

async def index_documents_async(
    documents: List[Document],
    batch_size: int = 10,
):
    """
    Add documents to Chroma using local Ollama embeddings.

    No Gemini API.
    No embedding API credits.
    No concurrent embedding requests.
    """

    log_header("VECTOR STORAGE PHASE")

    log_info(
        f"📚 VectorStore Indexing: Preparing to add "
        f"{len(documents)} documents",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into "
        f"{len(batches)} batches of {batch_size} documents each"
    )

    successful = 0

    for batch_num, batch in enumerate(batches, start=1):

        try:

            log_info(
                f"🔄 Processing batch "
                f"{batch_num}/{len(batches)} "
                f"({len(batch)} documents)"
            )

            # Chroma will call Ollama locally
            vectorstore.add_documents(batch)

            successful += 1

            log_success(
                f"VectorStore Indexing: Successfully added batch "
                f"{batch_num}/{len(batches)} "
                f"({len(batch)} documents)"
            )

        except Exception as e:

            log_error(
                f"VectorStore Indexing: Failed to add batch "
                f"{batch_num} - {e}"
            )

            break

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


# ============================================================
# MAIN
# ============================================================

async def main():

    log_header("DOCUMENTATION INGESTION PIPELINE")

    # --------------------------------------------------------
    # 1. TEST OLLAMA CONNECTION
    # --------------------------------------------------------

    log_header("OLLAMA EMBEDDING TEST")

    try:

        log_info(
            f"🔌 Connecting to Ollama at {OLLAMA_BASE_URL}"
        )

        log_info(
            f"🧠 Embedding model: {EMBEDDING_MODEL}"
        )

        test_embedding = embeddings.embed_query(
            "This is a test document."
        )

        log_success(
            f"✅ Ollama embedding successful "
            f"(vector dimensions: {len(test_embedding)})"
        )

    except Exception as e:

        log_error(
            f"❌ Could not connect to Ollama or embedding model: {e}"
        )

        log_error(
            "Make sure Ollama is running and the model is installed:"
        )

        log_error(
            "    ollama pull mxbai-embed-large"
        )

        return

    # --------------------------------------------------------
    # 2. TAVILY CRAWL
    # --------------------------------------------------------

    log_header("DOCUMENT CRAWLING PHASE")

    log_info(
        "🗺️ TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    try:

        res = tavily_crawl.invoke(
            {
                "url": "https://en.wikipedia.org/wiki/Tirumala",
                "max_depth": 2,
                "extract_depth": "advanced",
            }
        )

    except Exception as e:

        log_error(
            f"❌ Tavily crawl failed: {e}"
        )

        return

    # --------------------------------------------------------
    # 3. ONLY FIRST 3 URLS
    # --------------------------------------------------------

    all_results = res.get("results", [])

    crawl_results = all_results[:3]

    log_info(
        f"📄 TavilyCrawl: "
        f"{len(all_results)} URLs returned"
    )

    log_info(
        f"📌 Using only the first "
        f"{len(crawl_results)} URLs"
    )

    # --------------------------------------------------------
    # 4. CONVERT TO LANGCHAIN DOCUMENTS
    # --------------------------------------------------------

    all_docs = []

    for item in crawl_results:

        url = item.get("url")

        raw_content = item.get("raw_content")

        log_info(
            f"📄 Processing URL: {url}"
        )

        if not raw_content:

            log_warning(
                f"⚠️ Skipping URL because raw_content is empty: "
                f"{url}"
            )

            continue

        all_docs.append(
            Document(
                page_content=raw_content,
                metadata={
                    "source": url
                },
            )
        )

    log_success(
        f"📚 Created {len(all_docs)} LangChain documents"
    )

    # --------------------------------------------------------
    # 5. TEXT CHUNKING
    # --------------------------------------------------------

    log_header("DOCUMENT CHUNKING PHASE")

    log_info(
        f"✂️ Text Splitter: Processing "
        f"{len(all_docs)} documents",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )

    splitted_docs = text_splitter.split_documents(
        all_docs
    )

    log_success(
        f"Text Splitter: Created "
        f"{len(splitted_docs)} chunks from "
        f"{len(all_docs)} documents"
    )

    # --------------------------------------------------------
    # 6. LOCAL OLLAMA + CHROMA INDEXING
    # --------------------------------------------------------

    await index_documents_async(
        splitted_docs,
        batch_size=10,
    )

    # --------------------------------------------------------
    # 7. SUMMARY
    # --------------------------------------------------------

    log_header("PIPELINE COMPLETE")

    log_success(
        "🎉 Documentation ingestion pipeline finished!"
    )

    log_info(
        "📊 Summary:",
        Colors.BOLD,
    )

    log_info(
        f"   • URLs selected: {len(crawl_results)}"
    )

    log_info(
        f"   • Documents extracted: {len(all_docs)}"
    )

    log_info(
        f"   • Chunks created: {len(splitted_docs)}"
    )

    log_info(
        f"   • Embedding model: {EMBEDDING_MODEL}"
    )

    log_info(
        f"   • Vector database: Chroma"
    )

    log_info(
        f"   • Chroma database: {CHROMA_PERSIST_DIRECTORY}"
    )

    log_info(
        "   • Embeddings generated locally: YES"
    )

    log_info(
        "   • Gemini API used: NO"
    )

    log_info(
        "   • Pinecone API used: NO"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())