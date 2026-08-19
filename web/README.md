# Pulse Evidence web app

The Next.js interface provides a RAG technology/evaluation dashboard and a
chat workspace whose history is stored in the browser.

## Run locally

Start the Python API from the repository root:

```powershell
conda run -n student_rag python -m pip install -r requirements.txt
conda run -n student_rag python -m uvicorn src.api.app:app --reload
```

In a second terminal, start Next.js:

```powershell
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. Put `GEMINI_API_KEY` in the root `.env` file to
enable generated answers. The API remains on `http://127.0.0.1:8000` by default;
copy `.env.local.example` to `.env.local` to override it.
