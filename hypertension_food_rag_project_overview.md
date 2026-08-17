# Project Overview — Evidence-Based Food Guidance for Hypertension

We are building an AI-powered RAG application for a 5-day healthcare AI hackathon.

## Project Goal

The system should help adults with hypertension understand whether a specific food is generally suitable for a blood-pressure-friendly diet.

Example user questions:

- "Can I eat feta cheese if I have hypertension?"
- "Is canned tuna suitable for someone with high blood pressure?"
- "Is koshari a good choice for hypertension?"
- "What should I change in this meal to make it more suitable for a hypertension-friendly diet?"

The system must NOT behave like a doctor and must NOT diagnose, prescribe medication, or claim that a particular food can immediately treat a high blood-pressure reading.

The goal is to provide general, evidence-based dietary guidance supported by official medical and nutrition sources.

## Core Knowledge Sources

We have downloaded official documents/datasets that will form the RAG knowledge base.

The main clinical evidence should come from authoritative sources such as WHO guidelines, especially material related to:

- Sodium intake
- Potassium intake
- Healthy diet recommendations
- Lower-sodium salt substitutes
- Hypertension-related dietary guidance

Each document should be processed while preserving useful metadata such as:

- `document_name`
- `page_number`
- `section_title`
- `chunk_id`
- `source`

These metadata fields will later be used to generate traceable citations.

## Main User Flow

The expected pipeline is:

```text
User food question
        ↓
Query understanding / safety check
        ↓
Retrieve relevant evidence
        ↓
BM25 / Semantic / Hybrid Retrieval
        ↓
Optional Reranking
        ↓
Best evidence chunks
        ↓
Grounded LLM generation
        ↓
Food suitability assessment
        ↓
Explanation + safer alternatives/modifications
        ↓
Citations and supporting evidence
```

The LLM must answer using the retrieved evidence rather than relying on its internal knowledge whenever making health-related claims.

If the available evidence is insufficient, the system should explicitly say that there is not enough evidence instead of guessing.

## Expected Application Output

For a food-related question, the application should return structured information similar to:

```text
Food:
Feta cheese

Overall Assessment:
Limit / consume cautiously

Why:
Feta cheese can contain a relatively high amount of sodium.
High sodium intake is an important dietary consideration for adults
with hypertension.

What You Can Do:
- Choose a lower-sodium version when available.
- Reduce the portion size.
- Avoid combining it with several other high-sodium foods in the same meal.

Supporting Evidence:
WHO recommends limiting sodium intake in adults.

Sources:
WHO Sodium Intake Guideline
Section: ...
Page: ...

Confidence:
High
```

The exact UI can change later, but the backend should make it possible to return structured fields such as:

```text
food_name
assessment
reasoning
recommendations
supporting_evidence
citations
confidence
safety_message
```

## Food Suitability Categories

Instead of giving overly confident binary answers such as "good" or "bad", prefer categories such as:

```text
Suitable
Generally Suitable
Consume in Moderation
Limit
Insufficient Evidence
Needs Professional Guidance
```

The recommendation should include an explanation rather than only returning a category.

## Important Safety Rules

The application provides general dietary information for adults with hypertension.

It should NOT:

- Diagnose hypertension.
- Recommend changing or stopping medication.
- Provide medication doses.
- Claim that eating a specific food will immediately lower dangerously high blood pressure.
- Provide personalized medical treatment.
- Handle emergency symptoms as a normal food recommendation request.

Questions involving severe symptoms, emergencies, medication changes, kidney disease, or other conditions that substantially change dietary requirements should trigger appropriate safety handling instead of a normal recommendation.

## RAG Requirements

The system should support:

```text
Document ingestion
        ↓
Structure-aware / section-aware chunking
        ↓
Metadata preservation
        ↓
Embeddings
        ↓
Vector database
        ↓
Retriever
        ↓
Grounded generation
        ↓
Citations
```

For document chunking, we may use Docling and its `HybridChunker` because preserving document structure and section context is important.

Do not rely only on naive fixed-length chunking if document structure can be preserved.

## Retrieval Evaluation

Retrieval quality is a major part of the hackathon.

We will create an evaluation dataset containing predefined questions and ground-truth relevant chunks/sections.

Example:

```text
Question:
What does WHO recommend regarding sodium intake for adults?

Ground Truth:
WHO Sodium Guideline
Relevant section/chunk IDs: [...]
```

We should be able to evaluate experiments such as:

```text
Semantic Search
vs
BM25
vs
Hybrid Search
vs
Hybrid + Reranking
```

and measure metrics such as `Precision@K`.

Do not assume a retrieval strategy is better without measuring it.

## Citation and Grounding Requirements

Every important medical or dietary claim should be supported by retrieved evidence.

A citation should ideally identify:

```text
Document
Section
Page
Chunk ID
```

The system should distinguish between:

- retrieved evidence,
- generated explanation,
- unsupported information.

Avoid generating claims that cannot be supported by the retrieved context.

## Initial Development Goal

Do NOT try to build the entire final application immediately.

First build a minimal backend pipeline that can:

```text
1. Load the downloaded guideline documents.
2. Parse and clean them.
3. Create structure-aware chunks.
4. Preserve metadata.
5. Create embeddings.
6. Store them in a vector database.
7. Retrieve relevant chunks for a test question.
8. Display both chunk content and metadata.
```

Once retrieval works correctly, we will add:

```text
Evaluation
→ Hybrid Retrieval
→ Reranking
→ LLM Generation
→ Citations
→ Guardrails
→ UI
```

The priority is correctness, traceability, and measurable retrieval quality rather than adding many unnecessary features.

## Project Philosophy

The core principle of this project is:

**A fluent answer is not necessarily a safe answer.**

The application should behave as an evidence assistant rather than an AI doctor.

Every important recommendation should be traceable back to authoritative evidence, and the system should prefer admitting insufficient evidence over generating an unsupported answer.
