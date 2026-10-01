# this program is copied from core.py and changed the logic to chroma db and ollama llm instead of pinecose and gemini AI 
import os
from typing import Any, Dict

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

# IMPORTANT:
# This MUST be the same embedding model used by ingestion.py
EMBEDDING_MODEL = "mxbai-embed-large"

# Chat model used to generate the final answer.
#
# Change this to whatever chat model you have installed
# using: ollama list
#
# Example:
#   qwen3:4b
#   llama3.2:3b
#   gemma3:4b
#
#below model took 10 mins to get the response. 
#CHAT_MODEL = "qwen3-vl:4b"
#let's try the below model
CHAT_MODEL = "llama3.2:3b"

# ============================================================
# INITIALIZE LOCAL EMBEDDINGS
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)


# ============================================================
# INITIALIZE CHROMA VECTOR STORE
# ============================================================

CHROMA_PERSIST_DIRECTORY = "./chroma_db"

vectorstore = Chroma(
    collection_name="documentation",
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_function=embeddings,
)


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ============================================================
# INITIALIZE LOCAL OLLAMA CHAT MODEL
# ============================================================

model = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


# ============================================================
# RETRIEVAL TOOL
# ============================================================

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant documentation from the local Chroma
    vector database to help answer user questions.
    """

    # Retrieve top 4 most similar documents
    retrieved_docs = retriever.invoke(query)

    # Serialize documents for the LLM
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    # Return both:
    # 1. Text for the LLM
    # 2. Raw documents for the caller
    return serialized, retrieved_docs


# ============================================================
# RUN RAG PIPELINE
# ============================================================

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline using:
        User Query
            ↓
        Ollama Embedding
            ↓
        Chroma Retrieval
            ↓
        Ollama LLM
            ↓
        Final Answer

    Args:
        query: User's question.

    Returns:
        Dictionary containing:
            - answer: Generated answer
            - context: Retrieved documents
    """

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = (
        "You are a helpful AI assistant that answers questions "
        "using the provided documentation. "

        "You have access to a tool that retrieves relevant "
        "documentation from a local vector database. "

        "Always use the retrieval tool before answering "
        "questions about the documentation. "

        "Base your answer primarily on the retrieved documentation. "

        "Always cite the source URLs from the retrieved documentation "
        "when they are available. "

        "If the retrieved documentation does not contain enough "
        "information to answer the question, clearly say that the "
        "answer could not be found in the retrieved documentation. "
        "Do not invent information."
    )

    # --------------------------------------------------------
    # CREATE AGENT
    # --------------------------------------------------------

    agent = create_agent(
        model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": query,
        }
    ]

    # --------------------------------------------------------
    # INVOKE AGENT
    # --------------------------------------------------------

    response = agent.invoke(
        {
            "messages": messages
        }
    )

    # --------------------------------------------------------
    # EXTRACT FINAL ANSWER
    # --------------------------------------------------------

    answer = response["messages"][-1].content

    # --------------------------------------------------------
    # EXTRACT RETRIEVED DOCUMENTS
    # --------------------------------------------------------

    context_docs = []

    for message in response["messages"]:

        if isinstance(message, ToolMessage) and hasattr(
            message,
            "artifact"
        ):

            if isinstance(message.artifact, list):

                context_docs.extend(
                    message.artifact
                )

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "answer": answer,
        "context": context_docs,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = run_llm(
        query="Explain more about Amarnath temple"
    )

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n")
    print("=" * 80)
    print("RETRIEVED DOCUMENTS")
    print("=" * 80)

    for i, doc in enumerate(
        result["context"],
        start=1
    ):

        print(f"\n--- Document {i} ---")

        print(
            f"Source: "
            f"{doc.metadata.get('source', 'Unknown')}"
        )

        print(
            f"Content:\n"
            f"{doc.page_content[:500]}"
        )