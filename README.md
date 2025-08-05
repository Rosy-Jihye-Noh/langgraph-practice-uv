# langgraph_uv

## Overview

**langgraph_uv** is a Python-based AI agent project for legal and tax Q&A, focusing on Income Tax and Real Estate Tax.  
It leverages modern LLM and vector database technologies such as LangChain, LangGraph, OpenAI, and ChromaDB.  
Each tax domain implements a Retrieval-Augmented Generation (RAG) pipeline for accurate, context-aware answers.

---

## Main Features

### 1. Income Tax Q&A (`income_tax_graph.py`)
- Uses OpenAI Embeddings to vectorize income tax documents and stores them in ChromaDB.
- Retrieves relevant documents based on user queries and generates answers using an LLM.
- Includes chains and nodes for answer relevance, helpfulness, and hallucination detection.
- Utilizes LangGraph's StateGraph to structure the Q&A flow as a graph.

### 2. Real Estate Tax Q&A (`real_estate_tax_graph.py`)
- Loads and vectorizes real estate tax documents from `real_estate_tax.txt`, storing them in ChromaDB.
- Integrates real-time web search (TavilySearchResults) and LLMs to provide up-to-date information, such as market ratios.
- Extracts and calculates formulas, deductions, and market ratios step by step for tax computation.
- Connects each processing step as nodes in a StateGraph for systematic, multi-stage Q&A.

### 3. Main Entry Point (`main.py`)
- Currently prints a simple greeting ("Hello from langgraph-uv!").
- The actual Q&A pipelines are run from the respective tax-specific Python files.

---

## Project Structure

- `income_tax_graph.py` : Income tax Q&A pipeline
- `real_estate_tax_graph.py` : Real estate tax Q&A pipeline
- `real_estate_tax.txt` : Source data for real estate tax
- `chroma_tax/`, `real_estate_tax_collection/` : ChromaDB vector store data
- `main.py` : Main entry point (example)
- `README.md` : Project documentation

---

## Installation & Usage

1. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install langchain langgraph openai chromadb python-dotenv
   ```

2. **Set Up Environment Variables**
   - Create a `.env` file and add your OpenAI API key and any other required variables.

3. **Run Examples**
   - For income tax Q&A:
     ```python
     # See example code in income_tax_graph.py
     ```
   - For real estate tax Q&A:
     ```python
     # See example code in real_estate_tax_graph.py
     ```

---

## Dependencies

- Python 3.8+
- langchain, langgraph, openai, chromadb, python-dotenv, etc.

---

## Notes

- Each pipeline can be run and tested step-by-step in a Jupyter Notebook environment.
- Vector DBs (Chroma) are stored in separate directories for each tax domain.
- For production use, additional features such as user input/session management may be required.

---
