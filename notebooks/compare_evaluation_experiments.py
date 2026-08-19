"""Compare all saved retrieval-evaluation experiments in a Jupyter notebook.

Run from a notebook cell with:

    %run notebooks/compare_evaluation_experiments.py

The script only reads the evaluation CSV files. Set ``SAVE_CHARTS`` to ``True``
if you also want PNG copies in ``artifacts/evaluation/charts``.
"""

# %% [markdown]
# # Retrieval experiment comparison
#
# This notebook-style script compares every successful evaluation run saved in:
#
# - `artifacts/evaluation/evaluation_runs.csv`
# - `artifacts/evaluation/evaluation_question_results.csv`
#
# If Matplotlib is missing from the active notebook kernel, install it into the
# `student_rag` environment before starting Jupyter:
#
# `conda run -n student_rag python -m pip install matplotlib`

'''Run this in a new notebook by" 
%cd /Plod_presure_Hackathon
%run notebooks/compare_evaluation_experiments.py'''

# %% Imports and dashboard settings
from pathlib import Path
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(value: object) -> None:
        """Print tables when the script is run outside Jupyter."""
        if isinstance(value, pd.DataFrame):
            print(value.to_string(index=False))
        else:
            print(value)


# Change these values and rerun the notebook to explore different views.
SELECTED_K = 5
SHOW_LATEST_RUN_PER_CONFIGURATION = False
SAVE_CHARTS = False

METRICS = {
    "Precision": ("precision", "precision"),
    "Recall": ("recall", "recall"),
    "Hit Rate": ("hit_rate", "hit"),
    "MRR": ("mrr", "reciprocal_rank"),
    "nDCG": ("ndcg", "ndcg"),
}


# %% Locate and load the evaluation data
def find_project_root(start: Path) -> Path:
    """Find the repository root from either the root or a notebook directory."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        runs_file = candidate / "artifacts" / "evaluation" / "evaluation_runs.csv"
        if runs_file.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find artifacts/evaluation/evaluation_runs.csv. "
        "Start Jupyter from inside the project directory."
    )


PROJECT_ROOT = find_project_root(Path.cwd())
EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
RUNS_PATH = EVALUATION_DIR / "evaluation_runs.csv"
QUESTION_RESULTS_PATH = EVALUATION_DIR / "evaluation_question_results.csv"
CHARTS_DIR = EVALUATION_DIR / "charts"

runs = pd.read_csv(RUNS_PATH)
question_results = pd.read_csv(QUESTION_RESULTS_PATH)

runs["started_at_utc"] = pd.to_datetime(runs["started_at_utc"], utc=True, errors="coerce")
runs["finished_at_utc"] = pd.to_datetime(
    runs["finished_at_utc"], utc=True, errors="coerce"
)

successful_runs = runs.loc[runs["status"].eq("success")].copy()
successful_runs = successful_runs.sort_values("started_at_utc").reset_index(drop=True)

if successful_runs.empty:
    raise ValueError(f"No successful evaluation runs were found in {RUNS_PATH}")


def short_model_name(model_name: str) -> str:
    """Shorten a Hugging Face model identifier for chart labels."""
    return str(model_name).split("/")[-1]


def make_configuration_key(row: pd.Series) -> str:
    """Identify reruns that used the same retrieval configuration."""
    fields = [
        "embedding_model",
        "retriever_type",
        "reranker_model",
        "reranker_batch_size",
        "candidate_k",
        "rrf_k",
        "bm25_k1",
        "bm25_b",
        "ground_truth_sha256",
        "chunks_fingerprint",
    ]
    return "|".join(str(row.get(field, "")) for field in fields)


successful_runs["configuration_key"] = successful_runs.apply(
    make_configuration_key, axis=1
)
def make_run_label(row: pd.Series) -> str:
    label = short_model_name(row["embedding_model"])
    reranker_model = row.get("reranker_model")
    if pd.notna(reranker_model) and str(reranker_model).strip():
        label += f" + {short_model_name(reranker_model)}"
    return f"{label}\n{row['started_at_utc'].strftime('%Y-%m-%d %H:%M:%S')}"


successful_runs["run_label"] = successful_runs.apply(make_run_label, axis=1)

if SHOW_LATEST_RUN_PER_CONFIGURATION:
    plotted_runs = successful_runs.drop_duplicates(
        subset="configuration_key", keep="last"
    ).copy()
else:
    plotted_runs = successful_runs.copy()

plotted_run_ids = plotted_runs["run_id"].tolist()
plotted_question_results = question_results.loc[
    question_results["run_id"].isin(plotted_run_ids)
].copy()

print(f"Project root: {PROJECT_ROOT}")
print(f"Successful runs loaded: {len(successful_runs)}")
print(f"Runs included in charts: {len(plotted_runs)}")


# %% Experiment summary table
summary_columns = [
    "run_id",
    "started_at_utc",
    "embedding_model",
    "retriever_type",
    "reranker_model",
    "reranker_batch_size",
    "candidate_k",
    "rrf_k",
    "bm25_k1",
    "bm25_b",
    "question_count",
    "duration_seconds",
]
summary_columns = [column for column in summary_columns if column in plotted_runs]

summary = plotted_runs.loc[:, summary_columns].copy()
summary["duration_seconds"] = pd.to_numeric(
    summary["duration_seconds"], errors="coerce"
).round(3)
display(summary)


# %% Convert run metrics into a tidy table
def available_k_values(run_frame: pd.DataFrame) -> list[int]:
    """Read every K value from flattened columns or the metrics JSON field."""
    values = set()
    for column in run_frame.columns:
        if column.startswith("precision_at_"):
            suffix = column.removeprefix("precision_at_")
            if suffix.isdigit():
                values.add(int(suffix))
    if "metrics_json" in run_frame.columns:
        for raw_metrics in run_frame["metrics_json"].dropna():
            values.update(int(k) for k in json.loads(raw_metrics))
    return sorted(values)


K_VALUES = available_k_values(plotted_runs)

metric_records = []
for _, run in plotted_runs.iterrows():
    run_metrics = json.loads(run["metrics_json"]) if pd.notna(run["metrics_json"]) else {}
    for k in K_VALUES:
        metrics_at_k = run_metrics.get(str(k), {})
        for display_name, (column_prefix, json_name) in METRICS.items():
            column = f"{column_prefix}_at_{k}"
            if json_name in metrics_at_k:
                score = metrics_at_k[json_name]
            elif column in plotted_runs.columns and pd.notna(run[column]):
                score = run[column]
            else:
                continue
            metric_records.append(
                {
                    "run_id": run["run_id"],
                    "run_label": run["run_label"],
                    "embedding_model": run["embedding_model"],
                    "k": k,
                    "metric": display_name,
                    "score": float(score),
                }
            )

metrics_long = pd.DataFrame(metric_records)

if metrics_long.empty:
    raise ValueError("No metric columns such as precision_at_5 were found.")

display(metrics_long.head())


# %% Helpers shared by the charts
plt.style.use("seaborn-v0_8-whitegrid")


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Optionally save a high-resolution PNG without changing the CSV files."""
    if not SAVE_CHARTS:
        return
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHARTS_DIR / filename
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved chart: {output_path}")


colors = plt.cm.tab10(np.linspace(0, 1, max(len(plotted_runs), 2)))
color_by_run = dict(zip(plotted_runs["run_id"], colors))


# %% Chart 1: every metric across K for every experiment
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for axis, metric_name in zip(axes, METRICS):
    metric_data = metrics_long.loc[metrics_long["metric"].eq(metric_name)]
    for _, run in plotted_runs.iterrows():
        run_data = metric_data.loc[metric_data["run_id"].eq(run["run_id"])]
        axis.plot(
            run_data["k"],
            run_data["score"],
            marker="o",
            linewidth=2,
            color=color_by_run[run["run_id"]],
            label=run["run_label"],
        )
    axis.set_title(metric_name)
    axis.set_xlabel("K")
    axis.set_xticks(K_VALUES)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Mean score")

axes[-1].axis("off")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.08))
fig.suptitle("Retrieval metrics across all evaluation experiments", fontsize=16)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
save_figure(fig, "all_metrics_by_k.png")
plt.show()


# %% Chart 2: direct comparison of all experiments at one K
if SELECTED_K not in K_VALUES:
    raise ValueError(f"SELECTED_K={SELECTED_K} is unavailable. Choose one of {K_VALUES}.")

selected_metrics = metrics_long.loc[metrics_long["k"].eq(SELECTED_K)].pivot(
    index="run_label", columns="metric", values="score"
)
selected_metrics = selected_metrics.reindex(columns=list(METRICS))

fig, axis = plt.subplots(figsize=(14, max(5, 0.8 * len(selected_metrics))))
selected_metrics.plot(kind="barh", ax=axis, width=0.78)
axis.set_title(f"All experiments at K={SELECTED_K}")
axis.set_xlabel("Mean score")
axis.set_ylabel("Experiment")
axis.set_xlim(0, 1.05)
axis.legend(loc="lower right", ncols=3)
fig.tight_layout()
save_figure(fig, f"experiment_comparison_at_{SELECTED_K}.png")
plt.show()


# %% Best run for every metric and K
best_rows = (
    metrics_long.sort_values("score", ascending=False)
    .groupby(["k", "metric"], as_index=False)
    .first()
)
best_runs = best_rows[
    ["k", "metric", "score", "embedding_model", "run_id"]
].sort_values(["k", "metric"])
best_runs["score"] = best_runs["score"].round(4)
display(best_runs)


# %% Chart 3: per-question nDCG heatmap at the selected K
heatmap_data = plotted_question_results.loc[
    plotted_question_results["k"].eq(SELECTED_K)
].copy()

if heatmap_data.empty:
    print(f"No per-question rows were found for K={SELECTED_K}.")
else:
    label_by_run = plotted_runs.set_index("run_id")["run_label"].to_dict()
    heatmap_data["run_label"] = heatmap_data["run_id"].map(label_by_run)
    heatmap = heatmap_data.pivot(
        index="question_index", columns="run_label", values="ndcg"
    )
    heatmap = heatmap.reindex(
        columns=[label_by_run[run_id] for run_id in plotted_run_ids]
    )

    question_by_index = (
        heatmap_data.drop_duplicates("question_index")
        .set_index("question_index")["question"]
        .to_dict()
    )
    question_labels = [
        f"Q{question_index}: "
        + textwrap.shorten(
            str(question_by_index.get(question_index, "")),
            width=72,
            placeholder="...",
        )
        for question_index in heatmap.index
    ]

    fig, axis = plt.subplots(
        figsize=(max(10, 2.3 * len(heatmap.columns)), max(6, 0.7 * len(heatmap)))
    )
    image = axis.imshow(heatmap.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(heatmap.index)), question_labels)
    axis.set_title(f"Per-question nDCG@{SELECTED_K}")

    for row_index in range(heatmap.shape[0]):
        for column_index in range(heatmap.shape[1]):
            value = heatmap.iloc[row_index, column_index]
            if pd.notna(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black" if 0.25 < value < 0.85 else "white",
                    fontsize=9,
                )

    fig.colorbar(image, ax=axis, label=f"nDCG@{SELECTED_K}")
    fig.tight_layout()
    save_figure(fig, f"question_ndcg_heatmap_at_{SELECTED_K}.png")
    plt.show()


# %% Chart 4: runtime versus retrieval quality
quality_column = f"ndcg_at_{SELECTED_K}"
runtime_data = plotted_runs.dropna(subset=["duration_seconds", quality_column]).copy()

fig, axis = plt.subplots(figsize=(11, 6))
for _, run in runtime_data.iterrows():
    axis.scatter(
        float(run["duration_seconds"]),
        float(run[quality_column]),
        s=110,
        color=color_by_run[run["run_id"]],
    )
    axis.annotate(
        short_model_name(run["embedding_model"]),
        (float(run["duration_seconds"]), float(run[quality_column])),
        xytext=(6, 6),
        textcoords="offset points",
    )

axis.set_title(f"Evaluation runtime versus nDCG@{SELECTED_K}")
axis.set_xlabel("Evaluation duration (seconds)")
axis.set_ylabel(f"Mean nDCG@{SELECTED_K}")
axis.set_ylim(0, 1.05)
fig.tight_layout()
save_figure(fig, f"runtime_vs_ndcg_at_{SELECTED_K}.png")
plt.show()


# %% Failed-run audit
failed_runs = runs.loc[~runs["status"].eq("success"), ["run_id", "status", "error"]]
if failed_runs.empty:
    print("No failed evaluation runs are recorded.")
else:
    print("Failed evaluation runs:")
    display(failed_runs)
