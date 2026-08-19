"use client";

import { useMemo, useState } from "react";

export type EvaluationMetric = {
  cutoff: number;
  precision: number;
  recall: number;
  hit_rate: number;
  mrr: number;
  ndcg: number;
};

export type EvaluationRun = {
  run_id: string;
  started_at_utc: string;
  finished_at_utc: string;
  duration_seconds: number | null;
  embedding_model: string;
  retriever_type: string;
  reranker_enabled: boolean;
  reranker_model: string | null;
  candidate_k: number | null;
  question_count: number | null;
  chunk_count: number | null;
  embedding_dimension: number | null;
  ground_truth_name: string | null;
  metrics: EvaluationMetric[];
};

export type EvaluationQuestionMetric = {
  run_id: string;
  question_index: number;
  question: string;
  cutoff: number;
  ndcg: number;
};

const metricDefinitions = [
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "hit_rate", label: "Hit Rate" },
  { key: "mrr", label: "MRR" },
  { key: "ndcg", label: "nDCG" },
] as const;

type MetricKey = typeof metricDefinitions[number]["key"];

const runColors = [
  "#0d7b68", "#e58d50", "#3975a8", "#8b5fbf", "#c84c63", "#6d8b3d",
  "#2f9ca8", "#a96939", "#5e6fd6", "#b04f9c", "#73808a", "#d4a62a",
];

const metricColors: Record<MetricKey, string> = {
  precision: "#3975a8",
  recall: "#0d7b68",
  hit_rate: "#6d8b3d",
  mrr: "#8b5fbf",
  ndcg: "#e58d50",
};

function shortModelName(model: string | null): string {
  if (!model) return "No model";
  return model.split("/").at(-1) || model;
}

function runLabel(run: EvaluationRun): string {
  const model = shortModelName(run.embedding_model);
  const reranker = run.reranker_model
    ? ` + ${shortModelName(run.reranker_model)}`
    : "";
  const timestamp = run.started_at_utc
    ? run.started_at_utc.replace("T", " ").replace(/\.\d+\+00:00$/, " UTC")
    : run.run_id;
  return `${model}${reranker} · ${timestamp}`;
}

function scoreAt(
  run: EvaluationRun,
  cutoff: number,
  metric: MetricKey,
): number | null {
  const values = run.metrics.find((item) => item.cutoff === cutoff);
  return values ? values[metric] : null;
}

function formatScore(value: number): string {
  return value.toFixed(3);
}

function MetricLineChart({
  metric,
  label,
  runs,
  cutoffs,
}: {
  metric: MetricKey;
  label: string;
  runs: EvaluationRun[];
  cutoffs: number[];
}) {
  const width = 460;
  const height = 235;
  const margin = { left: 42, right: 14, top: 14, bottom: 34 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (cutoff: number) => {
    const index = cutoffs.indexOf(cutoff);
    return margin.left + (index / Math.max(cutoffs.length - 1, 1)) * plotWidth;
  };
  const y = (value: number) => margin.top + (1 - value) * plotHeight;

  return (
    <article className="metric-chart-card">
      <h3>{label}</h3>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} across K`}>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line className="chart-grid-line" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
            <text className="chart-axis-label" x={margin.left - 8} y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text>
          </g>
        ))}
        {cutoffs.map((cutoff) => (
          <text className="chart-axis-label" key={cutoff} x={x(cutoff)} y={height - 10} textAnchor="middle">K={cutoff}</text>
        ))}
        {runs.map((run, runIndex) => {
          const points = cutoffs
            .map((cutoff) => ({ cutoff, value: scoreAt(run, cutoff, metric) }))
            .filter((point): point is { cutoff: number; value: number } => point.value !== null);
          if (points.length === 0) return null;
          const color = runColors[runIndex % runColors.length];
          return (
            <g key={run.run_id}>
              <polyline
                fill="none"
                points={points.map((point) => `${x(point.cutoff)},${y(point.value)}`).join(" ")}
                stroke={color}
                strokeWidth="2.5"
              />
              {points.map((point) => (
                <circle key={point.cutoff} cx={x(point.cutoff)} cy={y(point.value)} fill={color} r="4">
                  <title>{`${runLabel(run)} — ${label}@${point.cutoff}: ${point.value.toFixed(4)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
    </article>
  );
}

function RunLegend({ runs }: { runs: EvaluationRun[] }) {
  return (
    <div className="run-legend">
      {runs.map((run, index) => (
        <div key={run.run_id}>
          <i style={{ background: runColors[index % runColors.length] }} />
          <span title={run.run_id}>{runLabel(run)}</span>
        </div>
      ))}
    </div>
  );
}

function ExperimentBars({ runs, cutoff }: { runs: EvaluationRun[]; cutoff: number }) {
  return (
    <div className="experiment-bars">
      {runs.map((run) => (
        <article key={run.run_id} className="experiment-bar-row">
          <h4 title={run.run_id}>{runLabel(run)}</h4>
          <div>
            {metricDefinitions.map(({ key, label }) => {
              const value = scoreAt(run, cutoff, key);
              return (
                <div className="metric-bar" key={key}>
                  <span>{label}</span>
                  <div><i style={{ background: metricColors[key], width: `${(value ?? 0) * 100}%` }} /></div>
                  <b>{value === null ? "—" : formatScore(value)}</b>
                </div>
              );
            })}
          </div>
        </article>
      ))}
    </div>
  );
}

function BestRuns({ runs, cutoff }: { runs: EvaluationRun[]; cutoff: number }) {
  return (
    <div className="best-run-grid">
      {metricDefinitions.map(({ key, label }) => {
        const ranked = runs
          .map((run) => ({ run, score: scoreAt(run, cutoff, key) }))
          .filter((item): item is { run: EvaluationRun; score: number } => item.score !== null)
          .sort((left, right) => right.score - left.score);
        const best = ranked[0];
        return (
          <article key={key}>
            <span>{label}@{cutoff}</span>
            <b>{best ? best.score.toFixed(4) : "—"}</b>
            <small>{best ? runLabel(best.run) : "No result"}</small>
          </article>
        );
      })}
    </div>
  );
}

function NdcgHeatmap({
  runs,
  questionMetrics,
  cutoff,
}: {
  runs: EvaluationRun[];
  questionMetrics: EvaluationQuestionMetric[];
  cutoff: number;
}) {
  const selectedRows = questionMetrics.filter((row) => row.cutoff === cutoff);
  const questions = Array.from(
    new Map(selectedRows.map((row) => [row.question_index, row.question])).entries(),
  ).sort((left, right) => left[0] - right[0]);
  const scoreMap = new Map(
    selectedRows.map((row) => [`${row.run_id}:${row.question_index}`, row.ndcg]),
  );

  if (questions.length === 0) {
    return <p className="chart-empty">No per-question results are available at K={cutoff}.</p>;
  }

  return (
    <div className="heatmap-scroll">
      <table className="ndcg-heatmap">
        <thead>
          <tr>
            <th>Question</th>
            {runs.map((run) => <th key={run.run_id} title={runLabel(run)}>{shortModelName(run.reranker_model || run.embedding_model)}</th>)}
          </tr>
        </thead>
        <tbody>
          {questions.map(([questionIndex, question]) => (
            <tr key={questionIndex}>
              <th title={question}>Q{questionIndex}: {question}</th>
              {runs.map((run) => {
                const value = scoreMap.get(`${run.run_id}:${questionIndex}`);
                const background = value === undefined
                  ? "#edf0ee"
                  : `hsl(${value * 120} 58% 82%)`;
                return (
                  <td key={run.run_id} style={{ background }} title={`${runLabel(run)} — ${value?.toFixed(4) ?? "No result"}`}>
                    {value === undefined ? "—" : value.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RuntimeScatter({ runs, cutoff }: { runs: EvaluationRun[]; cutoff: number }) {
  const points = runs
    .map((run, index) => ({ run, index, duration: run.duration_seconds, ndcg: scoreAt(run, cutoff, "ndcg") }))
    .filter((point): point is { run: EvaluationRun; index: number; duration: number; ndcg: number } => (
      point.duration !== null && point.ndcg !== null
    ));
  if (points.length === 0) return <p className="chart-empty">No runtime data is available.</p>;

  const width = 780;
  const height = 330;
  const margin = { left: 58, right: 24, top: 20, bottom: 44 };
  const durations = points.map((point) => point.duration);
  const minDuration = Math.min(...durations);
  const maxDuration = Math.max(...durations);
  const durationRange = Math.max(maxDuration - minDuration, 1);
  const x = (duration: number) => margin.left + ((duration - minDuration) / durationRange) * (width - margin.left - margin.right);
  const y = (ndcg: number) => margin.top + (1 - ndcg) * (height - margin.top - margin.bottom);

  return (
    <svg className="scatter-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Runtime versus nDCG at ${cutoff}`}>
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={tick}>
          <line className="chart-grid-line" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
          <text className="chart-axis-label" x={margin.left - 9} y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text>
        </g>
      ))}
      <text className="chart-axis-title" x={width / 2} y={height - 8} textAnchor="middle">Evaluation duration (seconds)</text>
      <text className="chart-axis-title" transform={`translate(15 ${height / 2}) rotate(-90)`} textAnchor="middle">Mean nDCG@{cutoff}</text>
      <text className="chart-axis-label" x={margin.left} y={height - 25} textAnchor="middle">{minDuration.toFixed(1)}</text>
      <text className="chart-axis-label" x={width - margin.right} y={height - 25} textAnchor="middle">{maxDuration.toFixed(1)}</text>
      {points.map((point) => (
        <circle
          key={point.run.run_id}
          cx={x(point.duration)}
          cy={y(point.ndcg)}
          fill={runColors[point.index % runColors.length]}
          r="7"
          stroke="#fff"
          strokeWidth="2"
        >
          <title>{`${runLabel(point.run)} — ${point.duration.toFixed(3)}s, nDCG@${cutoff} ${point.ndcg.toFixed(4)}`}</title>
        </circle>
      ))}
    </svg>
  );
}

export default function EvaluationCharts({
  runs,
  questionMetrics,
}: {
  runs: EvaluationRun[];
  questionMetrics: EvaluationQuestionMetric[];
}) {
  const cutoffs = useMemo(
    () => Array.from(new Set(runs.flatMap((run) => run.metrics.map((metric) => metric.cutoff)))).sort((a, b) => a - b),
    [runs],
  );
  const [selectedCutoff, setSelectedCutoff] = useState(
    cutoffs.includes(5) ? 5 : cutoffs[0] ?? 5,
  );

  if (runs.length === 0 || cutoffs.length === 0) return null;

  return (
    <section className="section evaluation-charts-section">
      <div className="section-heading">
        <div><p className="eyebrow">EXPERIMENT COMPARISON</p><h2>Evaluation charts</h2></div>
        <span>{runs.length} successful runs</span>
      </div>

      <div className="cutoff-selector" aria-label="Select evaluation cutoff">
        <span>Compare at:</span>
        {cutoffs.map((cutoff) => (
          <button
            className={selectedCutoff === cutoff ? "active" : ""}
            key={cutoff}
            onClick={() => setSelectedCutoff(cutoff)}
            type="button"
          >
            Top-{cutoff}
          </button>
        ))}
      </div>

      <div className="chart-block">
        <div className="chart-heading"><h3>Metrics across K</h3><p>Each line represents one saved evaluation run.</p></div>
        <div className="metric-chart-grid">
          {metricDefinitions.map(({ key, label }) => (
            <MetricLineChart key={key} metric={key} label={label} runs={runs} cutoffs={cutoffs} />
          ))}
        </div>
        <RunLegend runs={runs} />
      </div>

      <div className="chart-block">
        <div className="chart-heading"><h3>All experiments at K={selectedCutoff}</h3><p>Direct metric comparison using the selected cutoff.</p></div>
        <ExperimentBars runs={runs} cutoff={selectedCutoff} />
      </div>

      <div className="chart-block">
        <div className="chart-heading"><h3>Best run by metric</h3><p>Highest mean score at K={selectedCutoff}.</p></div>
        <BestRuns runs={runs} cutoff={selectedCutoff} />
      </div>

      <div className="chart-block">
        <div className="chart-heading"><h3>Per-question nDCG@{selectedCutoff}</h3><p>Green cells indicate stronger ranking quality for that question.</p></div>
        <NdcgHeatmap runs={runs} questionMetrics={questionMetrics} cutoff={selectedCutoff} />
      </div>

      <div className="chart-block">
        <div className="chart-heading"><h3>Runtime versus nDCG@{selectedCutoff}</h3><p>Hover over a point to inspect the experiment.</p></div>
        <RuntimeScatter runs={runs} cutoff={selectedCutoff} />
      </div>
    </section>
  );
}
