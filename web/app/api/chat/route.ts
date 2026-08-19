import { NextResponse } from "next/server";

const apiUrl = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";
const requestTimeoutMs = 75_000;
const rateLimitWindowMs = 60_000;
const rateLimitMaximum = 20;
const requestsByClient = new Map<string, number[]>();

function clientAddress(request: Request): string {
  return request.headers.get("x-forwarded-for")?.split(",")[0]?.trim()
    || request.headers.get("x-real-ip")
    || "local";
}

function isRateLimited(clientId: string): boolean {
  const now = Date.now();
  const cutoff = now - rateLimitWindowMs;
  const recent = (requestsByClient.get(clientId) ?? []).filter(
    (timestamp) => timestamp > cutoff,
  );
  if (recent.length >= rateLimitMaximum) {
    requestsByClient.set(clientId, recent);
    return true;
  }
  recent.push(now);
  requestsByClient.set(clientId, recent);
  return false;
}

export async function POST(request: Request) {
  const apiToken = process.env.RAG_API_TOKEN;
  if (!apiToken) {
    return NextResponse.json(
      { detail: "The chat service is not configured." },
      { status: 503 },
    );
  }

  if (isRateLimited(clientAddress(request))) {
    return NextResponse.json(
      { detail: "Too many chat requests. Please try again in one minute." },
      { status: 429 },
    );
  }

  try {
    const payload = await request.json();
    const response = await fetch(`${apiUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-RAG-API-Token": apiToken,
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(requestTimeoutMs),
    });
    const responseText = await response.text();
    let responseBody: unknown;
    try {
      responseBody = JSON.parse(responseText);
    } catch {
      responseBody = { detail: "The RAG API returned an invalid response." };
    }
    return NextResponse.json(responseBody, { status: response.status });
  } catch {
    return NextResponse.json(
      { detail: "The RAG API is unavailable or timed out." },
      { status: 503 },
    );
  }
}
