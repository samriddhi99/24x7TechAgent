## 📘 RAG Search Assistant – Centrale Méditerranée Edition

**Author**: Anne-Laure MEALIER
**License**: GPL-3.0
**Version**: 1.3
**Last Updated**: 2024-05-14


### 🎯 Overview

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline, from **web crawling and enrichment** to **interactive question-answering** using a **local LLM** and **ChromaDB**.

It enables:

* Crawling technical documentation from any website
* Summarizing and enriching content using a local LLM via [Ollama](https://ollama.com/)
* Embedding text via SentenceTransformers (GPU supported)
* Storing embeddings in a **Chroma vector database**
* Querying this knowledge base through a **web interface** (Dash) using:

  * RAG only
  * LLM only
  * Hybrid (RAG + LLM)


## 📦 Project Structure

```bash
project-root/
├── assets/
│   └── logo_centrale.svg        # Centrale Méditerranée logo for the web app
├── logs/                        # Log files from pipeline and indexing
├── chroma_db/                   # Persistent ChromaDB store
├── enriched_pages.jsonl         # Output of RAG content enrichment
├── generate_rag.py              # Crawl + enrich + embed + store
├── vector_indexing.py           # Index JSONL to ChromaDB with weighted embeddings
├── search_engine_WebApp.py      # Dash app interface for querying
├── terminal_rag_query.py        # Terminal app for querying
├── requirements.txt             # Python packages required
└── README.md                    # You are here 🚀
```


## 🚀 Quick Start

### 1. Install Dependencies

#### 🔹 Python

This tutorial assumes you have Python 3.12+ installed.

It's recommended to use a virtual environment or a conda environment to avoid conflicts with other projects.

Install the required packages with pip:

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 <code>requirements.txt</code></summary>

```txt
dash
dash-bootstrap-components
xhtml2pdf
beautifulsoup4
requests
chromadb
sentence-transformers
numpy
torch
scikit-learn
bs4
tiktoken
asyncio
markdown2
```
</details>


#### 🔹 Ollama

*What is Ollama?*

**Ollama** Ollama is a tool that lets you run large language models like LLaMA, Mistral, or Gemma locally on your computer.
It provides a simple CLI and API interface for downloading, running, and managing models.
**Ollama is designed for ease of use, privacy, and performance without needing the cloud.**

Ollama is available for:

* macOS, Linux, or Windows (WSL supported)
* x86\_64 or Apple Silicon (M1/M2)
* At least 8GB RAM (16GB+ recommended for larger models)

Choose the installation method that suits your needs:

##### Linux using the script (privileged access required)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> On some distros you may need to install dependencies (e.g., `libssl`, `curl`, `libc++`)

##### Linux (manual install without privileged access)

```bash
# Download the latest release
wget https://ollama.com/download/ollama-linux-amd64.tgz
# Extract the tarball to your local path
tar -C /your/local/path -xzf ollama-linux-amd64.tgz
# run the server
/your/local/path/ollama serve
```

Replace `/your/local/path` with the directory where you want to install Ollama or add it to your PATH.

In case the port `11434` is already in use, you can specify a different port:

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

Take care in this case to update the python scripts to use the correct port url.

##### macOS or Windows 10+

For macOS or Windows 10+, you can install Ollama using the official installer: [Ollama website](https://ollama.com/download) and follow the instructions.

##### Docker

If you prefer to use Docker, you can run Ollama in a container:

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

For Nvidia GPUs you'll need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

See the official docker hub page for more details: [Ollama Docker Hub](https://hub.docker.com/r/ollama/ollama).


#### 🔹 Ollama CLI

**Run Your First Model**

Pull and run a model (e.g., `gemma3`):

```bash
ollama run gemma3:4b
```

This will:

* Download the model if it's not already present
* Open an interactive chat interface
* Use Ctrl + d or /bye to exit.

**Listing Available Models**

```bash
ollama list
```

**Using Ollama in Script**

You can use the `ollama` CLI programmatically:

```bash
echo "What is the capital of France?" | ollama run llama2
```

If the model is not already downloaded, it will be pulled automatically.

**API Access**

Ollama also runs a local HTTP API on port `11434`. Example:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Tell me a joke"
}'
```

**Model Management**

*Remove a Model*

```bash
ollama rm gemma3:4b
```

**Resources**

* [Ollama Website](https://ollama.com)
* [GitHub Repository](https://github.com/ollama/ollama)
* [Community Models](https://ollama.com/library)


### 2. Run the Full RAG Pipeline

```bash
python generate_rag.py https://your.website.com/
```

* Crawls all HTML pages from the base URL
* Summarizes each with an LLM
* Generates keyword lists
* Chunks content and stores enriched JSONL
* Computes vector embeddings
* Stores everything into ChromaDB

Image from the Medium website - medium.com @arunpatidar
![RAG with ChromaDB and Ollama](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*CZTdupa4MBQgsKi9_0ojJg.png)

Image from the Medium website - medium.com @jorgepit-14189 
![ChromaDB](https://miro.medium.com/v2/resize:fit:720/format:webp/0*BU1rSIzgJRqzk5Bp.png)

### 3. (Optional) Re-index with Weighted Embeddings

```bash
python vector_indexing.py enriched_pages.jsonl
```

* Loads a .jsonl enriched file containing:
    - original text chunks
    - summaries
    - keyword metadata
* Uses a weighted combination of **summary** and **keywords**
* Computes weighted vector embeddings using batching (faster with GPU)
* Choose embeddings between 
    - **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**  
    This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.
    - **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** 
    This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.


### 4. Launch the Dash Web App

```bash
python search_engine_webApp.py
```

* Open `http://127.0.0.1:8050` in your browser
  If you are running on a server, you may need to set up port forwarding:
  ```bash
  ssh -L 8050:localhost:8050 user@remote-server -N
  ```
* Ask questions in English or French
* Toggle modes: Hybrid, RAG-only, or LLM-only
* Download responses as PDFs
* View sources with similarity scores
* **Displays time taken for each response**
* **Embeddings are now generated in-memory with caching for better performance**
* **LLM and vector search work together for accurate, fast answers**

## 🧠 Architecture

### 1. **generate_rag.py**

* Crawls HTML pages from a base URL
* Extracts text, enriches with LLM (summary + keywords)
* Outputs a `.jsonl` file
* Embeds chunks and stores in ChromaDB

### 2. **vector\_indexing.py**

* (Optional) Reindexes `.jsonl` with custom weights:

  * `0.8 * summary + 0.2 * keywords`
* Uses `paraphrase-multilingual-MiniLM` with GPU acceleration

### 3. **embed\_worker.py**

* Lightweight subprocess for embedding queries
* Used by the Dash app
* Returns a vector from stdin JSON input

### 4. **search_engine_webApp.py**

* Frontend with Dash and Bootstrap (CYBORG theme)
* User asks question → Query is embedded → Search ChromaDB
* LLM answers using RAG content or own knowledge
* PDF export and source display included


## 🖼️ Web Interface

* Query interface with text area
* Modes: RAG-only, Hybrid, LLM-only
* Show source URLs and scores (sorted by relevance)
* PDF export of full Q\&A session


## ⚙️ Configuration

Modify these constants in `search_engine_webApp.py`:

```python
TOP_K = 50                # Number of top matches to retrieve
THRESHOLD_GOOD = 0.70     # Minimum score to consider a match relevant
DEFAULT_LLM_MODEL = "gemma3:4b"  # Ollama model to use
DEFAULT_QUERY_MODE = "rag_only"  # Starting mode
```


## 📌 Notes

* The system assumes Ollama is running locally at `http://localhost:11434`
* Embedding and inference prefer GPU (if available)
* All logs are stored in `/logs` with timestamps
* Files in `/assets` (like logos) are auto-served by Dash


## 🧪 Testing

To test locally:

```bash
# Crawl test site
python generate_rag.py https://doc.cc.in2p3.fr/

# Index generated file
python vector_indexing.py enriched_pages.jsonl

# Launch web app
python search_engine_webApp.py

# Launch terminal app
python terminal_rag_query.py "How run an interactive job at CC ?"
```

## 📄 License

This project is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.html)

You are free to use, modify, and distribute this software, provided that:

The source code remains open and publicly accessible under the same license.
Any derivative works or modified versions are also released under GPL-3.0.
Appropriate credit is given to the original author.


## 👩‍🔬 Author

**Anne-Laure MEALIER**
Centrale Méditerranée – 2025

Optimized for GPU acceleration and on-premise privacy
