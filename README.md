# rag-paper-pipeline
Multimodal RAG pipeline for scientific papers: text + figure extraction, LLM description, embedding and retrieval

<img width="1295" height="753" alt="Στιγμιότυπο οθόνης 2026-06-12, 18 59 26" src="https://github.com/user-attachments/assets/d9a13d92-0f22-49cc-93d7-6b429298a3a8" />

## Pipeline

1. `notebooks/01_extraction.ipynb` — PDF → per-page text + figures (`output/text`, `output/images`)
2. `notebooks/02_retrieval.ipynb` — chunk pages, embed with `sentence-transformers`, store in ChromaDB (`output/chroma_db`)
3. `notebooks/03_generation.ipynb` / `src/rag.py` — retrieve relevant chunks for a question and call an LLM API to generate a grounded, cited answer

## Generation (step 3)

`src/rag.py` retrieves context from the ChromaDB store built in step 2, builds a grounded prompt, and calls an LLM through a provider-agnostic client (`src/llm_client.py` — Anthropic or OpenAI, chosen at runtime).

Setup:

```bash
pip install -r requirements.txt
cp .env.example .env   # set LLM_PROVIDER + the matching API key
```

Use as a library:

```python
from src.rag import answer
result = answer("What is endocytosis?", top_k=3)
print(result["answer"])
```

Or from the CLI:

```bash
python -m src.rag "What is endocytosis?" --top-k 5
```
