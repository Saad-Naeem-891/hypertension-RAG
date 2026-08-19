import urllib.request
import json

payload = {
    "message": "What is the recommended sodium intake?",
    "top_k": 3
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/chat",
    data=json.dumps(payload).encode('utf-8'),
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print("Error calling local API:", e)
