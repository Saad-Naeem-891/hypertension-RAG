"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

type Citation = {
  chunk_id: string;
  text: string;
  document_name: string | null;
  section_title: string | null;
  page_start: number | null;
  page_end: number | null;
};

type SupportingEvidence = {
  statement: string;
  chunk_ids: string[];
};

type Message = {
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  supportingEvidence?: SupportingEvidence[];
  confidence?: string;
  safetyMessage?: string;
  evidenceConfidencePercentage?: number | null;
  evidenceConfidenceThreshold?: number | null;
};

const starter: Message = {
  role: "assistant",
  text: "Ask about the WHO hypertension, sodium, or potassium guidelines. I will answer only from retrieved evidence and show the source citations.",
};

function isMessage(value: unknown): value is Message {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<Message>;
  return (
    (candidate.role === "user" || candidate.role === "assistant") &&
    typeof candidate.text === "string"
  );
}

function loadHistory(): Message[] {
  const saved = localStorage.getItem("pulse-rag-history");
  if (!saved) return [starter];
  try {
    const parsed: unknown = JSON.parse(saved);
    return Array.isArray(parsed) && parsed.every(isMessage) ? parsed : [starter];
  } catch {
    localStorage.removeItem("pulse-rag-history");
    return [starter];
  }
}

function citationReferences(
  chunkIds: string[],
  citations: Citation[] = [],
): number[] {
  return chunkIds
    .map((chunkId) => citations.findIndex((citation) => citation.chunk_id === chunkId) + 1)
    .filter((reference) => reference > 0);
}

function pages(citation: Citation): string {
  if (citation.page_start === null && citation.page_end === null) return "Not available";
  if (citation.page_start === citation.page_end || citation.page_end === null) {
    return String(citation.page_start ?? citation.page_end);
  }
  return `${citation.page_start ?? "?"} - ${citation.page_end}`;
}

function StructuredResponse({ message }: { message: Message }) {
  const citations = message.citations ?? [];

  return (
    <div className="rag-response">
      <section className="response-section recommendation-section">
        <h2>Recommendation</h2>
        <p>{message.text}</p>
      </section>

      <section className="response-section supporting-evidence">
        <h2>Supporting Evidence</h2>
        {message.supportingEvidence && message.supportingEvidence.length > 0 ? (
          <ul>
            {message.supportingEvidence.map((item, evidenceIndex) => {
              const references = citationReferences(item.chunk_ids, citations);
              return (
                <li key={`${evidenceIndex}-${item.chunk_ids.join("-")}`}>
                  <span>{item.statement}</span>
                  {references.length > 0 && (
                    <span className="evidence-references" aria-label="Citation references">
                      {references.map((reference) => `[${reference}]`).join(" ")}
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="not-available">No supporting evidence was provided.</p>
        )}
      </section>

      <section className="response-section citations-section">
        <h2>Citations</h2>
        {citations.length > 0 ? (
          <div className="citation-list">
            {citations.map((citation, citationIndex) => (
              <article className="citation" key={citation.chunk_id}>
                <b className="citation-number">[{citationIndex + 1}]</b>
                <dl>
                  <div><dt>Document</dt><dd>{citation.document_name || "WHO guideline"}</dd></div>
                  <div><dt>Page</dt><dd>{pages(citation)}</dd></div>
                </dl>
                <details>
                  <summary>More Evidence</summary>
                  <dl>
                    <div><dt>Section</dt><dd>{citation.section_title || "Not available"}</dd></div>
                    <div><dt>Chunk ID</dt><dd>{citation.chunk_id}</dd></div>
                  </dl>
                  <b>Evidence:</b>
                  <p className="chunk-text">
                    {citation.text || "Chunk text is unavailable in this saved response. Ask the question again to reload it."}
                  </p>
                </details>
              </article>
            ))}
          </div>
        ) : (
          <p className="not-available">No citations were provided.</p>
        )}
      </section>

      <section className="response-section confidence-section">
        <h2>Confidence</h2>
        <p>{message.confidence || "Not available"}</p>
        {message.evidenceConfidencePercentage !== null &&
          message.evidenceConfidencePercentage !== undefined && (
            <small className="evidence-confidence">
              Evidence confidence: {message.evidenceConfidencePercentage.toFixed(1)}%
              {message.evidenceConfidenceThreshold !== null &&
                message.evidenceConfidenceThreshold !== undefined
                ? ` · minimum ${message.evidenceConfidenceThreshold.toFixed(1)}%`
                : ""}
            </small>
          )}
      </section>

    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([starter]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setMessages(loadHistory());
    setHistoryLoaded(true);
  }, []);

  useEffect(() => {
    if (historyLoaded) {
      localStorage.setItem("pulse-rag-history", JSON.stringify(messages));
    }
  }, [historyLoaded, messages]);

  async function send(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message || loading) return;

    setInput("");
    setMessages((items) => [...items, { role: "user", text: message }]);
    setLoading(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "The RAG service is unavailable.");
      }
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          text: data.answer,
          citations: data.citations,
          supportingEvidence: data.supporting_evidence,
          confidence: data.confidence,
          safetyMessage: data.safety_message,
          evidenceConfidencePercentage: data.evidence_confidence_percentage,
          evidenceConfidenceThreshold: data.evidence_confidence_threshold,
        },
      ]);
    } catch (error) {
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          text:
            error instanceof Error
              ? error.message
              : "Unable to reach the RAG service.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function newConversation() {
    setMessages([starter]);
    localStorage.removeItem("pulse-rag-history");
  }

  return (
    <main className="chat-shell">
      <aside>
        <Link href="/" className="brand"><span>✦</span> Pulse Evidence</Link>
        <p className="eyebrow">CHAT HISTORY</p>
        <button className="new-chat" onClick={newConversation}>＋ New conversation</button>
        <p className="side-note">History is saved locally in this browser. Use New conversation to clear it.</p>
        <Link className="back" href="/">← Dashboard</Link>
      </aside>

      <section className="chat-panel">
        <header>
          <div>
            <p className="eyebrow">GROUNDED GUIDELINE ASSISTANT</p>
            <h1>Clinical evidence chat</h1>
          </div>
          <span className="live"><i /> Sources required</span>
        </header>

        <div className="messages">
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={index}>
              <span className="avatar">{message.role === "user" ? "You" : "PE"}</span>
              <div>
                {message.role === "assistant" && (
                  message.supportingEvidence || message.citations || message.confidence || message.safetyMessage
                ) ? (
                  <StructuredResponse message={message} />
                ) : (
                  <p>{message.text}</p>
                )}
              </div>
            </article>
          ))}

          {loading && (
            <article className="message assistant">
              <span className="avatar">PE</span>
              <p className="typing">Retrieving guideline evidence…</p>
            </article>
          )}
        </div>

        <form onSubmit={send}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            maxLength={2000}
            placeholder="Ask a question about hypertension guidelines…"
          />
          <button disabled={loading} aria-label="Send message">↑</button>
        </form>
        <p className="disclaimer">For educational use only. Not individualized medical advice.</p>
      </section>
    </main>
  );
}
