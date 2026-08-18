# Run the Hypertension RAG Project

This project searches the supplied WHO hypertension and nutrition PDFs. It
returns relevant evidence chunks; it does not generate a medical answer.

Run every command from the project root.

## 1. Prerequisites

- Miniconda or Anaconda
- Internet access the first time the ML models are downloaded
- The `DataSet/` folder containing the PDF files

The project is configured for the Conda environment named `student_rag`.
Check that it exists:

```powershell
conda env list
conda run -n student_rag python --version
```

If it does not exist, create it and install the pinned dependencies:

```powershell
conda create -n student_rag python=3.11 -y
conda run -n student_rag python -m pip install -r requirements.txt
```

If it already exists, install or update the project dependencies with:

```powershell
conda run -n student_rag python -m pip install -r requirements.txt
```

## 2. Download the models (first run only)

The application intentionally loads models from the local `models/` cache.
Run these commands while online before attempting retrieval:

```powershell
conda run -n student_rag python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Snowflake/snowflake-arctic-embed-s', cache_folder='models', device='cpu')"
conda run -n student_rag python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', cache_folder='models', device='cpu')"
```

After this, model loading works offline as long as the `models/` directory is
kept in place.

## 3. Prepare the retrieval data

Skip any step whose output is already present and current. Run all three steps
after adding or changing PDFs.

```powershell
conda run -n student_rag python -m src.ingestion.dataset_processor
conda run -n student_rag python -m src.embedding.embed_chunks --batch-size 4
conda run -n student_rag python -m src.vector_store.qdrant_store
```

These commands create or update:

- `artifacts/chunks/` — parsed, structured PDF chunks
- `artifacts/embeddings/` — the embedding matrix and manifest
- `artifacts/qdrant_db/` — the local vector database

Use the following only when you deliberately want to erase and rebuild the
Qdrant collection:

```powershell
conda run -n student_rag python -m src.vector_store.qdrant_store --recreate
```

## 4. Start the interactive search

```powershell
conda run --no-capture-output -n student_rag python main.py
```

Type a question when prompted, for example:

```text
What are the recommendations for sodium intake?
```

Useful options:

```powershell
conda run --no-capture-output -n student_rag python main.py --top-k 10
conda run --no-capture-output -n student_rag python main.py --top-k 5 --candidate-k 30
```

`--top-k` controls the number of final evidence chunks. `--candidate-k`
controls how many candidates each dense/BM25 retrieval branch contributes
before cross-encoder reranking.

## 5. Evaluate with Cohere Rerank (optional)

Set your Cohere API key only for the current PowerShell session, then run the
ground-truth evaluation. Do not put the key in source files or commit it.

```powershell
$env:COHERE_API_KEY = "<your-cohere-api-key>"
conda run -n student_rag python -m src.evaluation.evaluate_retrieval --reranker-provider cohere --reranker-model rerank-v4.0-fast
```

The evaluation sends only the query and the retrieved candidate chunks to
Cohere. Results are recorded in `artifacts/evaluation/evaluation_runs.csv`.
If Cohere returns HTTP 429, wait for the account rate limit to reset and rerun
the command.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named ...` | Re-run `conda run -n student_rag python -m pip install -r requirements.txt`. |
| Hugging Face or `local_files_only` error | Run both model-download commands in step 2 while connected to the internet. |
| Missing embeddings, manifest, or Qdrant collection | Run all commands in step 3 in order. |
| New PDFs do not appear in results | Re-run step 3 after placing PDFs in `DataSet/`. |
| Need automated tests | `pytest` is not pinned in `requirements.txt`; install it with `conda run -n student_rag python -m pip install pytest`, then run `conda run -n student_rag python -m pytest -q`. |

For a more detailed description of the pipeline and evaluation workflow, see
`SETUP_AND_RUN.md`.
