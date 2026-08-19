import Link from "next/link";

const technologies = [
  ["Document intelligence", "Docling", "Parses WHO guideline PDFs into structured text and metadata.", "Ready"],
  ["Embeddings", "Snowflake Arctic Embed S", "Creates 384-dimensional dense representations for semantic search.", "Ready"],
  ["Keyword retrieval", "BM25", "Finds exact clinical terms, units, and recommendations.", "Ready"],
  ["Vector search", "Qdrant", "Persistent local vector index for dense retrieval.", "Ready"],
  ["Fusion", "Reciprocal Rank Fusion", "Combines dense and keyword candidates before reranking.", "Ready"],
  ["Reranking", "Cross-Encoder MiniLM", "Reorders evidence chunks by question-to-passage relevance.", "Ready"],
  ["Generation", "Gemini / Grok", "Produces constrained answers with validated citations.", "Configure key"]
];

const evaluations = [
  ["Hybrid + Cross-Encoder", "Top 5", "0.76", "0.82", "0.74"],
  ["Hybrid + Cross-Encoder", "Top 10", "0.85", "0.83", "0.78"],
  ["Hybrid + Cross-Encoder", "Top 20", "0.92", "0.83", "0.81"]
];

export default function Dashboard() {
  return <main className="shell">
    <nav><Link href="/" className="brand"><span>✦</span> Pulse Evidence</Link><Link href="/chat" className="nav-button">Open RAG chat →</Link></nav>
    <header className="hero">
      <p className="eyebrow">HYPERTENSION GUIDELINES · RAG OPERATIONS</p>
      <h1>Evidence retrieval you can inspect.</h1>
      <p>Track the clinical knowledge pipeline and evaluate what reaches each answer.</p>
      <Link href="/chat" className="primary">Ask the guideline assistant <span>→</span></Link>
    </header>
    <section className="stats"><div><b>653</b><span>Indexed evidence chunks</span></div><div><b>3</b><span>WHO source documents</span></div><div><b>384</b><span>Embedding dimensions</span></div><div><b>20</b><span>Reviewed evaluation questions</span></div></section>
    <section className="section"><div className="section-heading"><div><p className="eyebrow">PIPELINE</p><h2>Technologies in use</h2></div><span className="live"><i /> System ready</span></div><div className="technology-grid">{technologies.map(([label, name, detail, state]) => <article className="technology" key={name}><span className="tech-label">{label}</span><h3>{name}</h3><p>{detail}</p><small className={state === "Ready" ? "ready" : "setup"}>{state}</small></article>)}</div></section>
    <section className="section evaluation"><div className="section-heading"><div><p className="eyebrow">OFFLINE BENCHMARK</p><h2>Retrieval evaluation</h2></div><span>Potassium ground-truth set</span></div><div className="table-wrap"><table><thead><tr><th>Configuration</th><th>Cutoff</th><th>Recall</th><th>MRR</th><th>nDCG</th></tr></thead><tbody>{evaluations.map((row) => <tr key={row[1]}>{row.map((cell) => <td key={cell}>{cell}</td>)}</tr>)}</tbody></table></div><p className="muted">Metrics are read from the latest successful local evaluation. Run the evaluator after changing models or chunking.</p></section>
  </main>;
}
