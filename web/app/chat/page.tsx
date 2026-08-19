"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

type Citation = { chunk_id: string; document_name: string | null; section_title: string | null; page_start: number | null; page_end: number | null };
type Message = { role: "user" | "assistant"; text: string; citations?: Citation[]; confidence?: string };
const starter: Message = { role: "assistant", text: "Ask about the WHO hypertension, sodium, or potassium guidelines. I will answer only from retrieved evidence and show the source citations." };

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([starter]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => { const saved = localStorage.getItem("pulse-rag-history"); if (saved) setMessages(JSON.parse(saved)); }, []);
  useEffect(() => { localStorage.setItem("pulse-rag-history", JSON.stringify(messages)); }, [messages]);

  async function send(event: FormEvent) {
    event.preventDefault(); const message = input.trim(); if (!message || loading) return;
    setInput(""); setMessages((items) => [...items, { role: "user", text: message }]); setLoading(true);
    try {
      const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The RAG service is unavailable.");
      setMessages((items) => [...items, { role: "assistant", text: data.answer, citations: data.citations, confidence: data.confidence }]);
    } catch (error) { setMessages((items) => [...items, { role: "assistant", text: error instanceof Error ? error.message : "Unable to reach the RAG service." }]); }
    finally { setLoading(false); }
  }

  return <main className="chat-shell"><aside><Link href="/" className="brand"><span>✦</span> Pulse Evidence</Link><p className="eyebrow">CHAT HISTORY</p><button className="new-chat" onClick={() => setMessages([starter])}>＋ New conversation</button><p className="side-note">History is saved locally in this browser.</p><Link className="back" href="/">← Dashboard</Link></aside><section className="chat-panel"><header><div><p className="eyebrow">GROUNDED GUIDELINE ASSISTANT</p><h1>Clinical evidence chat</h1></div><span className="live"><i /> Sources required</span></header><div className="messages">{messages.map((message, index) => <article className={`message ${message.role}`} key={index}><span className="avatar">{message.role === "user" ? "You" : "PE"}</span><div><p>{message.text}</p>{message.confidence && <small className="confidence">Confidence: {message.confidence}</small>}{message.citations?.map((citation) => <div className="citation" key={citation.chunk_id}><b>{citation.document_name || "WHO guideline"}</b><span>{citation.section_title || "Source section"} · p. {citation.page_start ?? "?"}{citation.page_end && citation.page_end !== citation.page_start ? `–${citation.page_end}` : ""}</span></div>)}</div></article>)}{loading && <article className="message assistant"><span className="avatar">PE</span><p className="typing">Retrieving guideline evidence…</p></article>}</div><form onSubmit={send}><input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask a question about hypertension guidelines…" /><button disabled={loading} aria-label="Send message">↑</button></form><p className="disclaimer">For educational use only. Not individualized medical advice.</p></section></main>;
}
