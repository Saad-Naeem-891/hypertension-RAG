import os
from dotenv import load_dotenv
from src.generation import GeminiGenerator
from src.reranking import RerankedHybridRetriever

load_dotenv()
question = "What is the recommended sodium intake?"
print(f"Testing RAG for question: '{question}'")

try:
    with RerankedHybridRetriever() as retriever:
        print("Retrieving evidence...")
        evidence = retriever.retrieve(question, top_k=3)
        print(f"Retrieved {len(evidence)} evidence chunks.")
        
        for idx, chunk in enumerate(evidence):
            print(f"Chunk {idx+1}: {chunk.chunk_id} (Score: {chunk.rerank_score:.4f})")
            
        if not evidence:
            print("No evidence retrieved. Exiting.")
            exit(0)
            
        print("\nCalling Gemini generator...")
        generator = GeminiGenerator()
        answer = generator.generate(question, evidence)
        
        print("\n=== RESPONSE ===")
        print(answer.recommendation)
        print("================\n")
except Exception as e:
    print("Error during RAG execution:", e)
