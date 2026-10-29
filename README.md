# Auditing the Auditor: Extraction Accuracy, Hallucination, and Abstention in Open-Weight LLMs for ESG Reporting

**A Longitudinal Empirical Benchmark Across 518 Corporate Disclosures (2010–2026)**

**Authors:** Kate Markin, Derek Amuna, Felix Coker, Redeemer Salami Okekale  
**Institution:** University of Ghana  
**Repository:** [https://github.com/Sredeemer/ESG-RAG-benchmark-ragas](https://github.com/Sredeemer/ESG-RAG-benchmark-ragas)

---

## Abstract Overview

Corporate sustainability auditing increasingly relies on Large Language Models (LLMs) to extract verified quantitative metrics from lengthy Environmental, Social, and Governance (ESG) reports. This repository provides the complete, self-contained replication artifact for our empirical evaluation of five leading open-weight LLMs across 518 full-length corporate sustainability reports published between 2010 and 2026 (279,862 text passages).

Evaluated Open-Weight Models:
- **Qwen3-4B** (`Qwen/Qwen3-4B-Instruct`, 4.0B dense)
- **DeepSeek-R1-7B** (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, 7.6B reasoning)
- **GPT-OSS-20B** (`openai/gpt-oss-20b`, 20.9B total / 3.6B active MoE)
- **Qwen3.5-27B** (`Qwen/Qwen3.5-27B`, 27.0B dense)
- **Gemma-4-31B** (`google/gemma-4-31B-it`, 30.7B dense)

Independent AI Referee:
- **Phi-4-14B** (`microsoft/phi-4`, 14.0B dense, independent from all generator families to eliminate self-enhancement bias)

Five Evaluated Evidence Regimes (3,000 Total Model Completions):
1. **Top-$k=1$ (Single Passage):** Minimal search context
2. **Top-$k=3$ (Multi-Passage):** Combined search context
3. **Oracle (Gold Source Passage):** Search errors isolated to establish reading ceiling
4. **Near-Miss Standard:** Unanswerable distractor passage with standard helpful prompt
5. **Near-Miss Strict:** Unanswerable distractor passage with strict compliance guardrail prompt

---

## Repository Structure

```
ESG-RAG-benchmark-ragas/
├── code/
│   ├── modal_esg_benchmark.py          # Complete end-to-end benchmark on Modal (H100 GPU)
│   ├── 00_select_documents.py          # Longitudinal document selection and stratification
│   ├── 00a_ingest_pdfs.py              # PyMuPDF structured text extraction
│   ├── 00b_download_2025_2026_pdfs.py  # Acquisition of 2025-2026 CSRD/ESRS filings
│   ├── 01_generate_testset.py          # Stratified 120-Q testset generation with balanced personas
│   ├── 02a_compute_embeddings.py       # Corpus indexing with Qwen3-Embed-4B
│   ├── 02_run_rag_evaluation.py        # Local evaluation runner
│   ├── 03_analyze_results.py           # Metric aggregation and breakdown analysis
│   ├── generate_publication_tables.py  # Generates Tables 1-5 (LaTeX)
│   └── generate_publication_figures.py # Generates Figures 1-7 adhering to Cruz guidelines
├── data/
│   ├── raw/                            # 518 extracted markdown filings (2010-2026)
│   └── documents/                      # Stratified document corpus for benchmarking
├── outputs/
│   ├── evaluation/                     # Full evaluation JSONs and aggregated CSV summaries
│   │   ├── h100_5models_5regimes_summary.csv
│   │   ├── h100_breakdown_by_qtype_and_era.csv
│   │   ├── h100_persona_breakdown.csv
│   │   └── h100_qualitative_audit_cases.json
│   ├── figures/                        # Publication figures (PDF & high-res PNG)
│   └── tables/                         # Publication tables (LaTeX)
├── paper/                              # Complete LaTeX manuscript
│   ├── main.tex                        # Self-contained academic paper
│   ├── main.bib                        # Comprehensive bibliography
│   ├── main.pdf                        # Compiled 20-page camera-ready PDF
│   ├── figures/                        # Vector PDF figures
│   └── tables/                         # Formatted LaTeX tables
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

---

## Quickstart & Replication

### 1. Environment Setup

```bash
git clone https://github.com/Sredeemer/ESG-RAG-benchmark-ragas.git
cd ESG-RAG-benchmark-ragas

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Regenerate Publication Tables & Figures

To regenerate all LaTeX tables and vector figures directly from the verified evaluation data:

```bash
# Generate Tables 1 through 5
python3 code/generate_publication_tables.py

# Generate Figures 1 through 7 (Vector PDF + 300 DPI PNG)
python3 code/generate_publication_figures.py
```

### 3. Compile Manuscript

The paper is written in standard XeLaTeX / LuaLaTeX using the `XCharter` typography and compiles cleanly via `tectonic` or `latexmk`:

```bash
cd paper
tectonic -r 2 main.tex
# or using latexmk
latexmk -pdf main.tex
```

### 4. Running the Full 3,000-Completion Evaluation on Cloud GPU (Modal)

The full evaluation runs 5 models $\times$ 5 regimes $\times$ 120 questions across an NVIDIA H100 80GB GPU. We provide the complete standalone runner in `code/modal_esg_benchmark.py`:

```bash
pip install modal
modal setup
modal run code/modal_esg_benchmark.py
```

---

## Citation

If you use this benchmark, code, or findings in your research, please cite:

```bibtex
@article{markin2026auditing,
  title={Auditing the Auditor: Extraction Accuracy, Hallucination, and Abstention in Open-Weight LLMs for ESG Reporting},
  author={Markin, Kate and Amuna, Derek and Coker, Felix and Okekale, Redeemer Salami},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the Apache 2.0 License. All evaluation data and code are provided for academic research and sustainability assurance benchmarking.
