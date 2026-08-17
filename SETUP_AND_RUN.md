# Setup and Run Guide

This guide explains how to run the hypertension food-guidance RAG ingestion
pipeline after cloning or pulling the project from GitHub.

The current pipeline performs:

```text
PDF guidelines
    -> Docling parsing
    -> HybridChunker
    -> JSON chunks
    -> multilingual E5 embeddings
    -> persistent local Qdrant database
```

Retrieval and LLM generation are not implemented yet.

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

## 5. Download the embedding model

The model is not expected to be committed to Git because it is large. Download
`intfloat/multilingual-e5-small` into the project-local `models/` cache:

```bash
conda run -n student_rag python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small', cache_folder='models', device='cpu')"
```

The download is approximately 471 MB and only needs to be completed once.

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

## 10. Run the tests

```bash
conda run -n student_rag python -m unittest discover -s tests -v
```

## Complete pipeline command order

For a clean checkout, run these commands in order:

```bash
conda run -n student_rag python -m pip install -r requirements.txt

conda run -n student_rag python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small', cache_folder='models', device='cpu')"

conda run -n student_rag python -m src.ingestion.dataset_processor

conda run -n student_rag python -m src.embedding.embed_chunks --batch-size 4

conda run -n student_rag python -m src.vector_store.qdrant_store

conda run -n student_rag python -m unittest discover -s tests -v
```

## Common problems

### `ModuleNotFoundError`

Confirm that the command starts with `conda run -n student_rag`, then reinstall
the requirements:

```bash
conda run -n student_rag python -m pip install -r requirements.txt
```

### Embedding model cannot be found locally

Run the model-download command from step 5. The embedding pipeline intentionally
uses the local cache and does not silently download a model.

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
