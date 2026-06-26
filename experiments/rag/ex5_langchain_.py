"""
Use LangChain

- Use RecursiveCharacterTextSplitter to chunk
- use Chroma.from_documents() for chunking+storing
- vectorstore.as_retriever() to return list[Document] instead of Chroma query dicts
- LLM call still manual (Cerebras via OpenAI client)

RUN:
    uv run experiments/rag/ex5_langchain.py
    uv run experiments/rag/ex5_langchain.py --live
    uv run experiments/rag/ex5_langchain.py --query "connectome analysis"
    uv run experiments/rag/ex5_langchain.py --rebuild

Load in documents -> chunk -> store -> retrieve

load in documents:
    -> pdf docs: PyMuPDFLoader
    -> html/arxiv/github: WebPageLoader
    -> youtube/scholar: skip (should load scholar)
"""

import os
import shutil

import click
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from extract_urls import parse_urls, Url

# load data dir, chroma dir, experiments dir, resume filepath, system prompt, resume, dotenv
DATA_DIR = Path(__file__).parent / "data"
LC_CHROMA_DIR = DATA_DIR / "chroma_langchain"
EXPERIMENTS_DIR = Path(__file__).parent.parent
RESUME_FILEPATH = EXPERIMENTS_DIR / "resume.txt"
SYSTEM_PROMPT_FILEPATH = EXPERIMENTS_DIR / "system_prompt.txt"
RESUME = RESUME_FILEPATH.read_text()
SYSTEM_PROMPT = SYSTEM_PROMPT_FILEPATH.read_text()

load_dotenv()
console = Console()

if __name__ == "__main__":
    pass

# use recursive character splitter separators that map to section break, header break, sentence end patterns

# document loading

# call cerebras api