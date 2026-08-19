import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")
print(f"Using API Key starting with: {api_key[:8]}...")
print(f"Using model: {model}")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=60000),
)
try:
    interaction = client.interactions.create(
        model=model,
        input="Respond with the word: Success!",
    )
    print("Response from Gemini:", interaction.output_text)
except Exception as e:
    print("Error during generation:", e)
