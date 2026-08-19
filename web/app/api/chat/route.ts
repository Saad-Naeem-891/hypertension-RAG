import { NextResponse } from "next/server";

const apiUrl = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    const response = await fetch(`${apiUrl}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "RAG API is offline. Start it with: python -m uvicorn src.api.app:app --reload" }, { status: 503 });
  }
}
