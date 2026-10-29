"""
Generate 7 publication-grade figures (Vector PDF + 300 DPI PNG)
strictly adhering to Luís Cruz's "16 Guidelines for Effective Data Visualizations
in Academic Papers" (https://luiscruz.github.io/2021/03/01/effective-visualizations.html).

Key Principles Applied:
- Guideline 1: Self-explanatory figures with clear titles and annotations.
- Guideline 2: Turn off the box around figures (remove top & right spines).
- Guideline 4: Visual variables (colors) used strictly for data variation, not decoration.
- Guideline 6: Consistent colors representing the same conditions across all panels.
- Guideline 7: Direct labeling without collisions; dedicated colorbar axes.
- Guideline 8: Horizontal text labels (no rotated or vertical text).
- Guideline 9 & 10: Clear visual hierarchy without chartjunk or redundant callout boxes.
- Guideline 16: High-contrast, colorblind-safe palettes (Okabe-Ito / Blues & Slate).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Global typography and styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

df_summary = pd.read_csv("outputs/evaluation/h100_5models_5regimes_summary.csv")
df_qtype_era = pd.read_csv("outputs/evaluation/h100_breakdown_by_qtype_and_era.csv")
df_persona = pd.read_csv("outputs/evaluation/h100_persona_breakdown.csv")

# Standardized model identifiers, concise display labels, and colorblind-safe palette
models = [
    ("qwen3_4b", "Qwen3-4B", "#2b5c8f", "o"),
    ("deepseek_r1_7b", "DeepSeek-R1-7B", "#d95f02", "s"),
    ("gpt_oss_20b", "GPT-OSS-20B", "#1b9e77", "^"),
    ("qwen3_5_27b", "Qwen3.5-27B", "#7570b3", "D"),
    ("gemma4_31b", "Gemma-4-31B", "#e7298a", "v"),
]
model_display_names = [m[1] for m in models]

# -------------------------------------------------------------------------
# Figure 1: Experimental Architecture & Pipeline Overview
# -------------------------------------------------------------------------
def make_fig1():
    fig, ax = plt.subplots(figsize=(10.5, 3.8), dpi=300)
    ax.set_xlim(-0.1, 10.85)
    ax.set_ylim(0.05, 4.55)
    ax.axis("off")

    # Stage 1: Corpus & Chunks
    p1 = patches.FancyBboxPatch((0.25, 2.4), 2.35, 1.8, boxstyle="round,pad=0.12", ec="#1b365d", fc="#f8fafc", lw=1.4)
    ax.add_patch(p1)
    ax.text(1.425, 3.85, "Stage 1: Document Corpus", ha="center", va="center", weight="bold", color="#1b365d", fontsize=9.0)
    ax.text(1.425, 3.3, "518 Longitudinal Reports\n(2010–2026: GRI, TCFD, CSRD)", ha="center", va="center", fontsize=7.8, color="#334155")
    ax.text(1.425, 2.75, "279,862 Passages (1k char)\nQwen3-Embed-4B Index", ha="center", va="center", fontsize=7.6, color="#64748b")

    # Stage 2: Stratified Query Benchmark
    p2 = patches.FancyBboxPatch((0.25, 0.3), 2.35, 1.7, boxstyle="round,pad=0.12", ec="#1b365d", fc="#f8fafc", lw=1.4)
    ax.add_patch(p2)
    ax.text(1.425, 1.65, "Stage 2: Audit Benchmark", ha="center", va="center", weight="bold", color="#1b365d", fontsize=9.0)
    ax.text(1.425, 1.15, "120 Stratified Audit Queries\n4 Question Types × 3 Eras", ha="center", va="center", fontsize=7.8, color="#334155")
    ax.text(1.425, 0.65, "Balanced E / S / G Profiles", ha="center", va="center", fontsize=7.6, color="#64748b")

    # Stage 3: Controlled Regimes
    p3 = patches.FancyBboxPatch((3.15, 0.3), 2.35, 3.9, boxstyle="round,pad=0.12", ec="#0369a1", fc="#f0f9ff", lw=1.4)
    ax.add_patch(p3)
    ax.text(4.325, 3.85, "Stage 3: 5 Regimes", ha="center", va="center", weight="bold", color="#0369a1", fontsize=9.0)
    regimes_text = [
        "1. Top-k=1 (Single Passage)",
        "2. Top-k=3 (Multi-Passage)",
        "3. Oracle (Gold Context)",
        "4. Near-Miss Standard",
        "5. Near-Miss Strict"
    ]
    for i, t in enumerate(regimes_text):
        ax.text(3.32, 3.25 - i*0.62, t, ha="left", va="center", fontsize=7.8, color="#0c4a6e")

    # Stage 4: Open-Weight LLMs
    p4 = patches.FancyBboxPatch((6.05, 0.8), 2.05, 3.4, boxstyle="round,pad=0.12", ec="#0f766e", fc="#f0fdfa", lw=1.4)
    ax.add_patch(p4)
    ax.text(7.075, 3.85, "Stage 4: 5 Evaluators", ha="center", va="center", weight="bold", color="#0f766e", fontsize=9.0)
    gen_text = "• Qwen3-4B\n• DeepSeek-R1-7B\n• GPT-OSS-20B\n• Qwen3.5-27B\n• Gemma-4-31B"
    ax.text(7.075, 2.7, gen_text, ha="center", va="center", fontsize=8.0, color="#134e4a")
    ax.text(7.075, 1.3, "NVIDIA H100 80GB\n3,000 Completions", ha="center", va="center", weight="semibold", fontsize=7.8, color="#0f766e")

    # Stage 5: Independent Phi-4 Judge
    p5 = patches.FancyBboxPatch((8.65, 0.8), 1.9, 3.4, boxstyle="round,pad=0.12", ec="#4338ca", fc="#eef2ff", lw=1.4)
    ax.add_patch(p5)
    ax.text(9.6, 3.85, "Stage 5: Audit Judge", ha="center", va="center", weight="bold", color="#4338ca", fontsize=9.0)
    ax.text(9.6, 3.2, "Phi-4-14B\n(14B Dense Referee)\nZero Family Bias", ha="center", va="center", fontsize=7.8, color="#312e81")
    ax.text(9.6, 2.05, "Audit Metrics:\n• Context Precision\n• Context Recall\n• Faithfulness\n• Answer Relevancy\n• Factual Correctness", ha="center", va="center", fontsize=7.4, color="#1e1b4b")
    ax.text(9.6, 1.15, "Strict Abstention\nScoring", ha="center", va="center", weight="semibold", fontsize=7.6, color="#4338ca")

    # Connective arrows with clear hierarchy
    arrow_props = dict(arrowstyle="->", lw=1.3, color="#475569")
    ax.annotate("", xy=(3.01, 3.3), xytext=(2.74, 3.3), arrowprops=arrow_props)
    ax.annotate("", xy=(3.01, 1.15), xytext=(2.74, 1.15), arrowprops=arrow_props)
    ax.annotate("", xy=(5.91, 2.5), xytext=(5.64, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(8.51, 2.5), xytext=(8.24, 2.5), arrowprops=arrow_props)

    plt.tight_layout()
    fig.savefig("outputs/figures/fig1_rag_architecture.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig("outputs/figures/fig1_rag_architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    print("Created Fig 1: fig1_rag_architecture")

# -------------------------------------------------------------------------
# Figure 2: Longitudinal Corpus Distribution & Protocol
# (Fixes text collision in Panel a and tick collision in Panel b)
# -------------------------------------------------------------------------
def make_fig2():
    fig = plt.figure(figsize=(10.5, 3.8), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel A: Clean Linear Flow (Left-to-Right + Downward Fold)
    ax1.set_xlim(-0.2, 10.6)
    ax1.set_ylim(0.1, 4.45)
    ax1.axis("off")
    ax1.set_title("(a) Stratified Benchmark Construction Protocol", loc="left", weight="bold", fontsize=10)

    box_ec = "#334155"
    box_fc = "#f8fafc"
    
    # Top Row with generous box width (2.85) and comfortable font size (7.4)
    steps_row1 = [
        (1.65, 3.3, "1. Load Corpus\n518 ESG Reports\n(2010–2026)"),
        (5.20, 3.3, "2. Chunk & Index\n279,862 Passages\nQwen3-Embed-4B"),
        (8.75, 3.3, "3. Stratify 120-Q\n4 Question Types\n3 Reporting Eras"),
    ]
    for x, y, text in steps_row1:
        patch = patches.FancyBboxPatch((x-1.4, y-0.65), 2.8, 1.3, boxstyle="round,pad=0.1", ec=box_ec, fc=box_fc, lw=1.2)
        ax1.add_patch(patch)
        ax1.text(x, y, text, ha="center", va="center", fontsize=7.5, color="#0f172a")

    # Connectors Row 1
    ax1.annotate("", xy=(3.68, 3.3), xytext=(3.17, 3.3), arrowprops=dict(arrowstyle="->", lw=1.3, color="#64748b"))
    ax1.annotate("", xy=(7.23, 3.3), xytext=(6.72, 3.3), arrowprops=dict(arrowstyle="->", lw=1.3, color="#64748b"))

    # Turn down to Row 2
    ax1.annotate("", xy=(8.75, 1.9), xytext=(8.75, 2.5), arrowprops=dict(arrowstyle="->", lw=1.3, color="#64748b"))

    # Bottom Row (Step 4 on right, Step 5 on left, arrow flows left)
    steps_row2 = [
        (8.75, 1.1, "4. Attach Regimes\nk=1, k=3, Oracle\nNear-Miss Traps"),
        (3.60, 1.1, "5. Evaluation (Phi-4-14B)\n3,000 Completions on H100\nZero Family Bias"),
    ]
    # Box 4
    patch4 = patches.FancyBboxPatch((8.75-1.4, 1.1-0.65), 2.8, 1.3, boxstyle="round,pad=0.1", ec=box_ec, fc=box_fc, lw=1.2)
    ax1.add_patch(patch4)
    ax1.text(8.75, 1.1, steps_row2[0][2], ha="center", va="center", fontsize=7.5, color="#0f172a")

    # Box 5 (generous width with short balanced lines)
    patch5 = patches.FancyBboxPatch((1.2, 1.1-0.65), 4.8, 1.3, boxstyle="round,pad=0.1", ec="#1e3a8a", fc="#eff6ff", lw=1.3)
    ax1.add_patch(patch5)
    ax1.text(3.60, 1.1, steps_row2[1][2], ha="center", va="center", fontsize=7.5, color="#1e3a8a", weight="semibold")

    # Flow from Step 4 to Step 5 (ample clearance, no collision)
    ax1.annotate("", xy=(6.15, 1.1), xytext=(7.25, 1.1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#64748b"))

    # Panel B: Temporal Distribution of Reports
    years = list(range(2010, 2027))
    report_counts = [33, 32, 34, 33, 34, 33, 33, 34, 34, 33, 34, 33, 34, 34, 30, 14, 6]

    # Meaningful sequential blues (Guideline 4 & 6)
    colors = []
    for y in years:
        if y < 2018:
            colors.append("#93c5fd") # 2010–2017: Voluntary GRI
        elif y <= 2024:
            colors.append("#3b82f6") # 2018–2024: TCFD/NFRD Transition
        else:
            colors.append("#1d4ed8") # 2025–2026: Mandatory CSRD/ESRS

    ax2.bar(years, report_counts, color=colors, edgecolor="#1e3a8a", lw=0.7, width=0.75)
    ax2.set_xlabel("Publication Year (2010–2026)")
    ax2.set_ylabel("Number of Full ESG Reports")
    ax2.set_title("(b) Longitudinal Corpus Distribution (N=518)", loc="left", weight="bold", fontsize=10)
    
    # Guideline 8 & 2: Regular even-year spacing eliminates tick collision ("20252026")
    ax2.set_xticks([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026])
    ax2.set_ylim(0, 42)
    ax2.xaxis.grid(False) # Turn off vertical grid lines to maximize data-ink ratio

    p1 = patches.Patch(facecolor="#93c5fd", edgecolor="#1e3a8a", label="2010–2017: Voluntary GRI")
    p2 = patches.Patch(facecolor="#3b82f6", edgecolor="#1e3a8a", label="2018–2024: TCFD/NFRD")
    p3 = patches.Patch(facecolor="#1d4ed8", edgecolor="#1e3a8a", label="2025–2026: CSRD/ESRS")
    ax2.legend(handles=[p1, p2, p3], loc="upper right", framealpha=0.95, edgecolor="#cbd5e1")

    plt.tight_layout()
    fig.savefig("outputs/figures/fig2_corpus_and_protocol.pdf", bbox_inches="tight")
    fig.savefig("outputs/figures/fig2_corpus_and_protocol.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Created Fig 2: fig2_corpus_and_protocol")

# -------------------------------------------------------------------------
# Figure 3: Master Performance Heatmap Across 5 Models & Metrics
# (Fixes colorbar overlapping Panel b FactCorr column and rotates labels)
# -------------------------------------------------------------------------
def make_fig3():
    # Dedicated GridSpec allocating an explicit column for the colorbar
    fig = plt.figure(figsize=(10, 3.6), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.04], wspace=0.22)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy_when_answered", "factual_correctness_all"]
    # Clean abbreviations that fit horizontally without rotation (Guideline 8)
    metric_labels = ["CtxPrec", "CtxRec", "Faith", "AnsRel", "FactCorr"]

    mat_k1 = []
    for slug, _, _, _ in models:
        row = df_summary[(df_summary["model_slug"] == slug) & (df_summary["regime"] == "k1")][metrics].values[0]
        mat_k1.append(row)
    mat_k1 = np.array(mat_k1)

    mat_k3 = []
    for slug, _, _, _ in models:
        row = df_summary[(df_summary["model_slug"] == slug) & (df_summary["regime"] == "k3")][metrics].values[0]
        mat_k3.append(row)
    mat_k3 = np.array(mat_k3)

    cmap = plt.cm.YlGnBu

    # Panel A: Top-k=1
    im1 = ax1.imshow(mat_k1, cmap=cmap, vmin=0.35, vmax=0.95, aspect="auto")
    ax1.set_title("(a) Top-k=1 Minimal Retrieval", loc="left", weight="bold", fontsize=10)
    ax1.set_xticks(range(len(metric_labels)))
    ax1.set_xticklabels(metric_labels, rotation=0, fontsize=8.5) # Horizontal labels
    ax1.set_yticks(range(len(model_display_names)))
    ax1.set_yticklabels(model_display_names, fontsize=8.5)

    for i in range(len(model_display_names)):
        for j in range(len(metric_labels)):
            val = mat_k1[i, j]
            tc = "white" if val > 0.75 else "black"
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center", color=tc, fontsize=8.2, weight="semibold")

    # Panel B: Top-k=3
    im2 = ax2.imshow(mat_k3, cmap=cmap, vmin=0.35, vmax=0.95, aspect="auto")
    ax2.set_title("(b) Top-k=3 Multi-Chunk Fusion", loc="left", weight="bold", fontsize=10)
    ax2.set_xticks(range(len(metric_labels)))
    ax2.set_xticklabels(metric_labels, rotation=0, fontsize=8.5) # Horizontal labels
    ax2.set_yticks(range(len(model_display_names)))
    ax2.set_yticklabels([]) # Hide duplicate y-labels cleanly

    for i in range(len(model_display_names)):
        for j in range(len(metric_labels)):
            val = mat_k3[i, j]
            tc = "white" if val > 0.75 else "black"
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", color=tc, fontsize=8.2, weight="semibold")

    # Colorbar in its own dedicated axis (zero overlap with FactCorr!)
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("RAGAS Metric Score (Phi-4 Judge)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8.0)

    fig.savefig("outputs/figures/fig3_master_heatmap.pdf", bbox_inches="tight")
    fig.savefig("outputs/figures/fig3_master_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Created Fig 3: fig3_master_heatmap")

# -------------------------------------------------------------------------
# Figure 4: Retrieval Scaling & Ceiling (k=1 -> k=3 -> Oracle)
# -------------------------------------------------------------------------
def make_fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)

    regimes = ["k1", "k3", "oracle"]
    regime_labels = ["Top-k=1\n(Minimal)", "Top-k=3\n(Multi-Chunk)", "Oracle\n(Gold Chunk)"]

    for slug, label, color, marker in models:
        sub = df_summary[df_summary["model_slug"] == slug].set_index("regime")
        fc_vals = [sub.loc[r, "factual_correctness_all"] for r in regimes]
        faith_vals = [sub.loc[r, "faithfulness"] for r in regimes]

        ax1.plot(range(3), fc_vals, label=label, color=color, marker=marker, markersize=6.5)
        ax2.plot(range(3), faith_vals, label=label, color=color, marker=marker, markersize=6.5)

    # Panel A: Factual Correctness
    ax1.set_title("(a) Factual Correctness (F1) Scaling", loc="left", weight="bold", fontsize=10)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(regime_labels, fontsize=8.5)
    ax1.set_ylabel("Factual Correctness (F1)")
    ax1.set_ylim(0.35, 0.78)
    ax1.axhline(0.5, color="#94a3b8", linestyle=":", alpha=0.7)
    ax1.xaxis.grid(False)

    # Panel B: Faithfulness
    ax2.set_title("(b) Faithfulness Scaling", loc="left", weight="bold", fontsize=10)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(regime_labels, fontsize=8.5)
    ax2.set_ylabel("Faithfulness Score")
    ax2.set_ylim(0.60, 0.98)
    ax2.legend(loc="lower right", framealpha=0.95, edgecolor="#cbd5e1", fontsize=8.0)
    ax2.xaxis.grid(False)

    plt.tight_layout()
    fig.savefig("outputs/figures/fig4_retrieval_scaling_ceiling.pdf", bbox_inches="tight")
    fig.savefig("outputs/figures/fig4_retrieval_scaling_ceiling.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Created Fig 4: fig4_retrieval_scaling_ceiling")

# -------------------------------------------------------------------------
# Figure 5: Near-Miss Adversarial Abstention vs Hallucination
# (Fixes color inconsistency, eliminates label collisions, horizontal layout)
# -------------------------------------------------------------------------
def make_fig5():
    # Guideline 6: Consistent colors representing the same condition in both panels!
    # "Standard RAG Prompt" = Steel Blue (#93c5fd), "Strict Abstention Prompt" = Deep Navy (#1d4ed8)
    c_std = "#93c5fd"
    c_strict = "#1d4ed8"
    ec = "#1e3a8a"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)

    y_pos = np.arange(len(models))
    bar_height = 0.35

    std_abstain, strict_abstain = [], []
    std_halluc, strict_halluc = [], []

    for slug, _, _, _ in models:
        sub = df_summary[df_summary["model_slug"] == slug].set_index("regime")
        std_abstain.append(sub.loc["nearmiss_standard", "faithful_abstention_rate"] * 100)
        strict_abstain.append(sub.loc["nearmiss_strict", "faithful_abstention_rate"] * 100)
        std_halluc.append(sub.loc["nearmiss_standard", "hallucination_or_substitution_rate"] * 100)
        strict_halluc.append(sub.loc["nearmiss_strict", "hallucination_or_substitution_rate"] * 100)

    # Panel A: Faithful Abstention Rate (Higher is Better)
    rects1 = ax1.barh(y_pos - bar_height/2, std_abstain, bar_height, label="Standard RAG Prompt", color=c_std, edgecolor=ec, lw=0.8)
    rects2 = ax1.barh(y_pos + bar_height/2, strict_abstain, bar_height, label="Strict Abstention Prompt", color=c_strict, edgecolor=ec, lw=0.8)

    ax1.set_xlabel("Faithful Abstention Rate (%)")
    ax1.set_title("(a) Faithful Abstention on Near-Miss Contexts", loc="left", weight="bold", fontsize=10)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_display_names, fontsize=8.5)
    ax1.set_xlim(0, 105)
    ax1.invert_yaxis() # Top-to-bottom: Qwen3-4B down to Gemma-4-31B
    ax1.yaxis.grid(False)

    # Clean direct data labels with sufficient clearance
    for r in rects1:
        w = r.get_width()
        ax1.annotate(f"{w:.1f}%", xy=(w, r.get_y() + r.get_height()/2), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5)
    for r in rects2:
        w = r.get_width()
        ax1.annotate(f"{w:.1f}%", xy=(w, r.get_y() + r.get_height()/2), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, weight="bold")

    # Panel B: Unfaithful Hallucination Rate (Lower is Better) - Identical Color Semantics!
    rects3 = ax2.barh(y_pos - bar_height/2, std_halluc, bar_height, label="Standard RAG Prompt", color=c_std, edgecolor=ec, lw=0.8)
    rects4 = ax2.barh(y_pos + bar_height/2, strict_halluc, bar_height, label="Strict Abstention Prompt", color=c_strict, edgecolor=ec, lw=0.8)

    ax2.set_xlabel("Hallucination / Substitution Rate (%)")
    ax2.set_title("(b) Unfaithful Hallucination Rate (Lower is Better)", loc="left", weight="bold", fontsize=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([]) # Hide duplicate y-labels
    ax2.set_xlim(0, 58)
    ax2.invert_yaxis()
    ax2.yaxis.grid(False)

    for r in rects3:
        w = r.get_width()
        ax2.annotate(f"{w:.1f}%", xy=(w, r.get_y() + r.get_height()/2), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5)
    for r in rects4:
        w = r.get_width()
        ax2.annotate(f"{w:.1f}%", xy=(w, r.get_y() + r.get_height()/2), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, weight="bold")

    # Single unified legend placed cleanly in upper right of Panel (b) where empty space is vast
    ax2.legend(handles=[rects1, rects2], loc="upper right", framealpha=0.95, edgecolor="#cbd5e1", fontsize=8.0)

    plt.tight_layout()
    fig.savefig("outputs/figures/fig5_nearmiss_abstention.pdf", bbox_inches="tight")
    fig.savefig("outputs/figures/fig5_nearmiss_abstention.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Created Fig 5: fig5_nearmiss_abstention")

# -------------------------------------------------------------------------
# Figure 6: Breakdown by Question Type & Regulatory Era
# -------------------------------------------------------------------------
def make_fig6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)

    x = np.arange(len(models))
    width = 0.35

    # Panel A: Hard KPI vs Policy Boilerplate (k=3)
    kpi_scores, boiler_scores = [], []
    for slug, _, _, _ in models:
        sub = df_summary[(df_summary["model_slug"] == slug) & (df_summary["regime"] == "k3")]
        kpi_scores.append(sub["hard_kpi_factual_correctness"].values[0])
        boiler_scores.append(sub["boilerplate_factual_correctness"].values[0])

    r1 = ax1.bar(x - width/2, kpi_scores, width, label="Hard Numerical KPIs", color="#0284c7", edgecolor="#0369a1", lw=0.8)
    r2 = ax1.bar(x + width/2, boiler_scores, width, label="Policy Boilerplate Text", color="#f59e0b", edgecolor="#d97706", lw=0.8)

    ax1.set_ylabel("Factual Correctness (F1)")
    ax1.set_title("(a) Hard KPI Extraction vs. Policy Boilerplate (k=3)", loc="left", weight="bold", fontsize=9.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Qwen-4B", "R1-7B", "GPT-20B", "Qwen-27B", "Gemma-31B"], rotation=0, fontsize=8.5)
    ax1.set_ylim(0, 0.76)
    ax1.xaxis.grid(False)
    ax1.legend(loc="upper left", framealpha=0.95, edgecolor="#cbd5e1", fontsize=8.0)

    for r in r1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.2)
    for r in r2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.2, weight="bold")

    # Panel B: Era Breakdown (2010-2024 Voluntary vs 2025-2026 CSRD) in Oracle
    sub_q_vol = df_qtype_era[(df_qtype_era["regime"] == "oracle") & (df_qtype_era["era"] == "2010_2024_voluntary")]
    sub_q_csrd = df_qtype_era[(df_qtype_era["regime"] == "oracle") & (df_qtype_era["era"] == "2025_2026_csrd")]

    vol_scores, csrd_scores = [], []
    for slug, _, _, _ in models:
        v = sub_q_vol[sub_q_vol["model_slug"] == slug]["factual_correctness_all"].mean()
        c = sub_q_csrd[sub_q_csrd["model_slug"] == slug]["factual_correctness_all"].mean()
        vol_scores.append(v)
        csrd_scores.append(c)

    r_vol = ax2.bar(x - width/2, vol_scores, width, label="2010–2024 Voluntary (N=84)", color="#818cf8", edgecolor="#4338ca", lw=0.8)
    r_csrd = ax2.bar(x + width/2, csrd_scores, width, label="2025–2026 CSRD (N=36)", color="#4338ca", edgecolor="#1e1b4b", lw=0.8)

    ax2.set_ylabel("Factual Correctness (F1, Oracle)")
    ax2.set_title("(b) Performance Across Regulatory Eras (Oracle)", loc="left", weight="bold", fontsize=9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Qwen-4B", "R1-7B", "GPT-20B", "Qwen-27B", "Gemma-31B"], rotation=0, fontsize=8.5)
    ax2.set_ylim(0, 0.98) # Ample space to prevent collision with upper boundary
    ax2.xaxis.grid(False)
    ax2.legend(loc="upper left", framealpha=0.95, edgecolor="#cbd5e1", fontsize=8.0)

    for r in r_vol:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.2)
    for r in r_csrd:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.2, weight="bold")

    plt.tight_layout()
    fig.savefig("outputs/figures/fig6_qtype_and_era_breakdown.pdf", bbox_inches="tight")
    fig.savefig("outputs/figures/fig6_qtype_and_era_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Created Fig 6: fig6_qtype_and_era_breakdown")

# -------------------------------------------------------------------------
# Figure 7: ESG Pillar Breakdown & Pareto Decision Frontier
# -------------------------------------------------------------------------
def make_fig7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)

    personas = ["Environmental KPI Extractor", "Social KPI Extractor", "Governance KPI Extractor"]
    x = np.arange(len(models))
    w = 0.25

    e_scores, s_scores, g_scores = [], [], []
    for slug, _, _, _ in models:
        sub_m = df_persona[(df_persona["model_slug"] == slug) & (df_persona["regime"] == "k3")]
        e = sub_m[sub_m["persona"] == personas[0]]["factual_correctness_all"].mean()
        s = sub_m[sub_m["persona"] == personas[1]]["factual_correctness_all"].mean()
        g = sub_m[sub_m["persona"] == personas[2]]["factual_correctness_all"].mean()
        e_scores.append(e)
        s_scores.append(s)
        g_scores.append(g)

    # Panel A: Pillar Breakdown
    ax1.bar(x - w, e_scores, w, label="Environmental (E)", color="#10b981", edgecolor="#047857", lw=0.8)
    ax1.bar(x, s_scores, w, label="Social (S)", color="#3b82f6", edgecolor="#1d4ed8", lw=0.8)
    ax1.bar(x + w, g_scores, w, label="Governance (G)", color="#8b5cf6", edgecolor="#6d28d9", lw=0.8)

    ax1.set_ylabel("Factual Correctness (F1, k=3)")
    ax1.set_title("(a) Accuracy Across ESG Pillars (Top-k=3)", loc="left", weight="bold", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Qwen-4B", "R1-7B", "GPT-20B", "Qwen-27B", "Gemma-31B"], rotation=0, fontsize=8.5)
    ax1.set_ylim(0, 0.74)
    ax1.xaxis.grid(False)
    ax1.legend(loc="upper left", framealpha=0.95, edgecolor="#cbd5e1", fontsize=8.0)

    # Panel B: 2D Pareto Decision Frontier
    param_sizes = [4.0, 7.6, 20.9, 27.0, 30.7]

    # Deterministic offsets to eliminate label overlaps and keep well inside axes
    offsets = [
        (8, -16),   # Qwen3-4B (below right of circle)
        (8, 4),     # DeepSeek-R1-7B
        (-28, -24), # GPT-OSS-20B (centered below circle, well inside right edge)
        (8, 6),     # Qwen3.5-27B
        (-64, 8),   # Gemma-4-31B (cleanly upper-left of circle)
    ]

    for i, (slug, label, color, marker) in enumerate(models):
        sub = df_summary[df_summary["model_slug"] == slug].set_index("regime")
        x_val = sub.loc["oracle", "factual_correctness_all"]
        y_val = sub.loc["nearmiss_strict", "faithful_abstention_rate"] * 100
        size = param_sizes[i] * 16

        ax2.scatter(x_val, y_val, s=size, color=color, alpha=0.8, edgecolors="#1e293b", lw=1.1, label=f"{label} ({param_sizes[i]}B)")
        dx, dy = offsets[i]
        ax2.annotate(f"{label}\n({x_val:.2f}, {y_val:.1f}%)", xy=(x_val, y_val), xytext=(dx, dy), textcoords="offset points", fontsize=7.6)

    ax2.set_xlabel("Extraction Ceiling (Oracle Factual Correctness F1)")
    ax2.set_ylabel("Adversarial Faithful Abstention Rate (%)")
    ax2.set_title("(b) Pareto Frontier: Accuracy vs. Abstention", loc="left", weight="bold", fontsize=10)
    ax2.set_xlim(0.54, 0.78)
    ax2.set_ylim(60, 95)
    ax2.axvline(0.70, color="#94a3b8", linestyle=":", alpha=0.6)
    ax2.axhline(75, color="#94a3b8", linestyle=":", alpha=0.6)

    # Clean non-intrusive quadrant indicator without heavy box
    ax2.text(0.705, 91, "High Accuracy &\nHigh Abstention", ha="left", va="center", fontsize=7.6, color="#047857", weight="bold")

    plt.tight_layout()
    fig.savefig("outputs/figures/fig7_pillar_pareto_tradeoff.pdf", bbox_inches="tight")
    fig.savefig("outputs/figures/fig7_pillar_pareto_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Created Fig 7: fig7_pillar_pareto_tradeoff")

if __name__ == "__main__":
    import shutil, glob
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    os.makedirs("paper/figures", exist_ok=True)
    for f in glob.glob("outputs/figures/*"):
        shutil.copy2(f, "paper/figures/")
    print("All 7 figures generated and copied to paper/figures/ successfully!")
