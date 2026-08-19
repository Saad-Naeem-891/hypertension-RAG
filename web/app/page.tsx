import Link from "next/link";
import EvaluationCharts, {
  EvaluationQuestionMetric,
  EvaluationRun,
} from "./components/evaluation_charts";

const technologies = [
  ["Document intelligence", "Docling", "Parses WHO guideline PDFs into structured text and metadata.", "Ready"],
  ["Embeddings", "Snowflake Arctic Embed S", "Creates 384-dimensional dense representations for semantic search.", "Ready"],
  ["Keyword retrieval", "BM25", "Finds exact clinical terms, units, and recommendations.", "Ready"],
  ["Vector search", "Qdrant", "Persistent local vector index for dense retrieval.", "Ready"],
  ["Fusion", "Reciprocal Rank Fusion", "Combines dense and keyword candidates before reranking.", "Ready"],
  ["Reranking", "Cross-Encoder MiniLM", "Reorders evidence chunks by question-to-passage relevance.", "Ready"],
  ["Generation", "Gemini", "Produces constrained answers with validated citations.", "Configure key"],
];

async function loadEvaluationData<T>(endpoint: string): Promise<T[]> {
  const apiUrl = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";
  try {
    const response = await fetch(`${apiUrl}/${endpoint}`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    if (!response.ok) return [];
    const payload: unknown = await response.json();
    return Array.isArray(payload) ? payload as T[] : [];
  } catch {
    return [];
  }
}

function formatScore(score: number): string {
  return score.toFixed(4);
}

export default async function Dashboard() {
  const [evaluationRuns, questionMetrics] = await Promise.all([
    loadEvaluationData<EvaluationRun>("evaluations"),
    loadEvaluationData<EvaluationQuestionMetric>("evaluation-question-results"),
  ]);
  const latest = evaluationRuns.at(-1);
  const configuration = latest?.reranker_enabled
    ? "Hybrid + Cross-Encoder"
    : "Hybrid Retrieval";

  return (
    <main className="shell">
      <nav>
        <Link href="/" className="brand"><span>✦</span> Pulse Evidence</Link>
        <Link href="/chat" className="nav-button">Open RAG chat →</Link>
      </nav>

      <header className="hero">
        <p className="eyebrow">HYPERTENSION GUIDELINES · RAG OPERATIONS</p>
        <h1>Evidence retrieval you can inspect.</h1>
        <p>Track the clinical knowledge pipeline and evaluate what reaches each answer.</p>
        <Link href="/chat" className="primary">Ask the guideline assistant <span>→</span></Link>
      </header>

      <section className="stats">
        <div><b>{latest?.chunk_count ?? "—"}</b><span>Indexed evidence chunks</span></div>
        <div><b>3</b><span>WHO source documents</span></div>
        <div><b>{latest?.embedding_dimension ?? "—"}</b><span>Embedding dimensions</span></div>
        <div><b>{latest?.question_count ?? "—"}</b><span>Evaluation questions</span></div>
      </section>

      <section className="section">
        <div className="section-heading">
          <div><p className="eyebrow">PIPELINE</p><h2>Technologies in use</h2></div>
          <span className="live"><i /> System configured</span>
        </div>
        <div className="technology-grid">
          {technologies.map(([label, name, detail, state]) => (
            <article className="technology" key={name}>
              <span className="tech-label">{label}</span>
              <h3>{name}</h3>
              <p>{detail}</p>
              <small className={state === "Ready" ? "ready" : "setup"}>{state}</small>
            </article>
          ))}
        </div>
      </section>

      <section className="section evaluation">
        <div className="section-heading">
          <div><p className="eyebrow">OFFLINE BENCHMARK</p><h2>Retrieval evaluation</h2></div>
          <span>{latest?.ground_truth_name ?? "No evaluation loaded"}</span>
        </div>

        {latest && latest.metrics.length > 0 ? (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Configuration</th><th>Cutoff</th><th>Precision</th>
                    <th>Recall</th><th>Hit Rate</th><th>MRR</th><th>nDCG</th>
                  </tr>
                </thead>
                <tbody>
                  {latest.metrics.map((metric) => (
                    <tr key={metric.cutoff}>
                      <td>{configuration}</td>
                      <td>Top {metric.cutoff}</td>
                      <td>{formatScore(metric.precision)}</td>
                      <td>{formatScore(metric.recall)}</td>
                      <td>{formatScore(metric.hit_rate)}</td>
                      <td>{formatScore(metric.mrr)}</td>
                      <td>{formatScore(metric.ndcg)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted">
              Latest successful run: {latest.run_id}. Metrics are loaded from the local evaluation history.
            </p>
          </>
        ) : (
          <p className="empty-state">
            No successful evaluation is available. Start the Python API and run the retrieval evaluator.
          </p>
        )}
      </section>

      <EvaluationCharts runs={evaluationRuns} questionMetrics={questionMetrics} />
    </main>
  );
}
