# Setup and Run Guide

This guide explains how to run the hypertension food-guidance RAG ingestion
pipeline after cloning or pulling the project from GitHub.

## System pipeline

```mermaid
flowchart TD
    A["WHO PDF Guidelines"] --> B["Docling Parsing"]
    B --> C["Structured Chunks + Metadata"]
    C --> D["English Arctic Embeddings"]
    D --> E["Persistent Qdrant Database"]
    C --> F["BM25 Index"]

    Q["User Question"] --> G["Query Embedding"]
    G --> H["Dense Search"]
    E --> H

    Q --> I["BM25 Search"]
    F --> I

    H --> J["Combine and Deduplicate"]
    I --> J
    J --> K["Reciprocal Rank Fusion"]
    K --> L["Cross-Encoder Reranking"]
    L --> M["Final Top-K Evidence Chunks"]

    M --> N["Ground-Truth Comparison"]
    N --> O["Metrics + Evaluation CSV History"]
```

Dense search and BM25 each retrieve a configurable candidate pool. Their full
union is deduplicated by `chunk_id`, ranked with RRF, and then reordered by the
cross-encoder before the final Top-K evidence is returned.

LLM generation is not implemented yet.

## 1. Get the project

For a new copy:

```bash
git clone <repository-url>
cd Plod_presure_Hackathon
```

For an existing copy:

```bash
git pull
```

Run all remaining commands from the project root.

## 2. Use the required Conda environment

This project uses the existing environment named `student_rag`.

Verify that it exists and that Python comes from the correct environment:

```bash
conda run -n student_rag python -c "import sys; print(sys.executable)"
```

The printed path should contain:

```text
envs/student_rag/bin/python
```

Do not use the system `python` or `python3`, and do not create a virtual
environment for this project.

## 3. Install project dependencies

Install the pinned packages into `student_rag`:

```bash
conda run -n student_rag python -m pip install -r requirements.txt
```

The main dependencies are Docling, Sentence Transformers, and Qdrant Client.

## 4. Confirm the PDF dataset

The guideline PDFs must be present inside `DataSet/`. Subdirectories are
supported.

```text
DataSet/
├── Guideline for the pharmacological treatment of hypertension in adults.pdf
├── Potassium intake.pdf
└── sodium intake for adults and children.pdf
```

The pipeline discovers PDF files recursively, so individual filenames are not
hard-coded.

## 5. Download the embedding and reranker models

The model is not expected to be committed to Git because it is large. Download
`Snowflake/snowflake-arctic-embed-s` into the project-local `models/` cache:

```bash
conda run -n student_rag python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Snowflake/snowflake-arctic-embed-s', cache_folder='models', device='cpu')"
```

The model has approximately 33 million parameters and only needs to be
downloaded once.

Download the lightweight English cross-encoder used after hybrid retrieval:

```bash
conda run -n student_rag python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', cache_folder='models', device='cpu')"
```

## 6. Parse and chunk all PDFs

Run the Docling ingestion pipeline:

```bash
conda run -n student_rag python -m src.ingestion.dataset_processor
```

Generated files:

```text
artifacts/
├── parsed_markdown/
│   └── one Markdown file per PDF
└── chunks/
    └── one JSON chunk file per PDF
```

With the current three PDFs, the expected summary is:

```text
PDFs discovered: 3
Successfully processed: 3
Failed: 0
Total chunks created: 653
```

The exact chunk count may change if the PDFs, Docling version, tokenizer, or
chunking configuration changes.

## 7. Optionally export chunks as readable text

This creates simplified text files containing only the chunk ID and raw text:

```bash
conda run -n student_rag python -m src.ingestion.export_chunks_text
```

Output:

```text
artifacts/chunks_txt/
```

This step is useful for manually preparing the retrieval ground-truth dataset,
but it is not required for embedding.

## 8. Generate dense embeddings

Generate normalized embeddings from each chunk's `contextualized_text`:

```bash
conda run -n student_rag python -m src.embedding.embed_chunks --batch-size 4
```

The model runs on the CPU. A small batch size is used to reduce memory usage.

Expected result for the current dataset:

```text
Embedding matrix: (653, 384)
```

Generated files:

```text
artifacts/embeddings/
├── embeddings.npy
└── manifest.json
```

`embeddings.npy` contains the vectors. `manifest.json` maps every matrix row to
its chunk ID and source file.

## 9. Create the persistent Qdrant database

Insert the embeddings and complete chunk payloads into local Qdrant:

```bash
conda run -n student_rag python -m src.vector_store.qdrant_store
```

Expected result:

```text
Collection: hypertension_guidelines
Vector dimension: 384
Points indexed: 653
Points stored: 653
```

The persistent database is stored on disk at:

```text
artifacts/qdrant_db/
```

Running the command again updates points with the same stable IDs. To explicitly
delete and rebuild the collection, use:

```bash
conda run -n student_rag python -m src.vector_store.qdrant_store --recreate
```

Only use `--recreate` when intentionally replacing the existing collection.

## 10. Run hybrid retrieval with cross-encoder reranking

Use `--no-capture-output` so the interactive prompt receives terminal input:

```bash
conda run --no-capture-output -n student_rag python main.py
```

Enter a question when prompted. The default is Top-5. To request another number
of chunks, use `--top-k`:

```bash
conda run --no-capture-output -n student_rag python main.py --top-k 10
```

By default, dense search and BM25 each retrieve 25 candidates. Their full
deduplicated union is scored by RRF and then by the cross-encoder. You can
change the branch candidate pool independently:

```bash
conda run --no-capture-output -n student_rag python main.py --top-k 5 --candidate-k 30
```

The retriever prints the final reranked evidence, cross-encoder score, original
hybrid rank, Dense rank and BM25 rank. It does not generate an answer or call
an LLM.

## 11. Evaluate retrieval quality

Evaluate the reranked hybrid retriever against the manually reviewed ground truth:

```bash
conda run -n student_rag python -m src.evaluation.evaluate_retrieval
```

The default evaluation calculates Precision, Recall, Hit Rate, MRR and nDCG at
Top-5, Top-10 and Top-20. Every experiment is appended to:

```text
artifacts/evaluation/evaluation_runs.csv
artifacts/evaluation/evaluation_question_results.csv
```

The first CSV contains one row per run, including retrieval settings, input
fingerprints, environment information, aggregate metrics and a reproducible
command. The second contains per-question judgments for dashboards and error
analysis.

To measure the non-reranked RRF baseline with the same settings, add:

```bash
conda run -n student_rag python -m src.evaluation.evaluate_retrieval --no-reranker
```

Rerun a previous configuration using its recorded run ID:

```bash
conda run -n student_rag python -m src.evaluation.evaluate_retrieval \
  --rerun eval_YYYYMMDDTHHMMSSffffffZ
```

The evaluator warns if the ground truth, chunks, embedding manifest, source code
or Git commit changed since the original run.

## 12. Run the tests

```bash
conda run -n student_rag python -m unittest discover -s tests -v
```

## Complete pipeline command order

For a clean checkout, run these commands in order:

```bash
conda run -n student_rag python -m pip install -r requirements.txt

conda run -n student_rag python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Snowflake/snowflake-arctic-embed-s', cache_folder='models', device='cpu')"

conda run -n student_rag python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', cache_folder='models', device='cpu')"

conda run -n student_rag python -m src.ingestion.dataset_processor

conda run -n student_rag python -m src.embedding.embed_chunks --batch-size 4

conda run -n student_rag python -m src.vector_store.qdrant_store

conda run -n student_rag python -m src.evaluation.evaluate_retrieval

conda run -n student_rag python -m unittest discover -s tests -v
```

After setup, run the interactive retriever separately:

```bash
conda run --no-capture-output -n student_rag python main.py
```

## Common problems

### `ModuleNotFoundError`

Confirm that the command starts with `conda run -n student_rag`, then reinstall
the requirements:

```bash
conda run -n student_rag python -m pip install -r requirements.txt
```

### Embedding model cannot be found locally

Run both model-download commands from step 5. The embedding and reranking
pipelines use the local cache and do not silently download models at runtime.

### Qdrant says the database is already accessed

Only one local Qdrant client/process should open the database directory at a
time. Stop the other Python process and retry. Do not delete the database lock
while another process is running.

### The laptop runs out of memory during embedding

Close memory-heavy applications and reduce the batch size:

```bash
conda run -n student_rag python -m src.embedding.embed_chunks --batch-size 2
```

### Chunk IDs changed

Chunk IDs may change when PDFs or chunking settings change. Regenerate the
ground-truth evaluation dataset before calculating retrieval metrics.
