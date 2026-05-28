# SESG RAGAS Project

This project uses RAGAS (Retrieval-Augmented Generation Assessment) to evaluate RAG systems on Environmental, Social, and Governance (ESG) documents.

## Overview

The project consists of four main steps:
1. **Data Generation** (`01_generate_testset.py`) - Generates test questions from ESG documents
2. **Embedding Computation** (`02a_compute_embeddings.py`) - Embeds all corpus chunks once and saves to disk (run once after step 1)
3. **RAG Evaluation** (`02_run_rag_evaluation.py`) - Runs RAG across 5 LLMs, loading embeddings from file
4. **Analysis** (`03_analyze_results.py`) - Evaluates RAG responses using RAGAS metrics

## Setup

### Prerequisites
- Python 3.8+
- Ollama installed and running
- Required Ollama models:
  - `qwen3:4b-instruct` (for generation/evaluation)
  - `qwen3-embedding:4b` (for embeddings)
  - Various small models for testing (see `02_run_rag_evaluation.py`)

### Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pull required Ollama models:
```bash
ollama pull qwen3:4b-instruct
ollama pull qwen3-embedding:4b
```

## Usage

### 1. Generate Test Data
```bash
python code/01_generate_testset.py
```
Creates:
- `outputs/01_esg_testset/01_esg_knowledge_graph.pkl` - Knowledge graph
- `outputs/01_esg_testset/01_esg_testset.csv` - Test questions and references

### 2a. Compute Embeddings (once)
```bash
python code/02a_compute_embeddings.py
# or via runner:
bash run_experiment.sh step2a
```
Embeds all corpus chunks with `qwen3-embedding:4b` and saves to `outputs/01_esg_testset/doc_embeddings.npy`. Run once — step 2 loads from this file on every run.

### 2. Run RAG Evaluation
```bash
python code/02_run_rag_evaluation.py
# or via runner:
bash run_experiment.sh step2
```
Loads pre-computed embeddings, runs RAG across 5 models, saves responses to `outputs/02_rag_responses/`.

### 3. Evaluate Performance
```bash
python code/03_analyze_results.py
# or via runner:
bash run_experiment.sh step3
```
Evaluates RAG responses using RAGAS metrics, saves results to `outputs/03_evaluation_results/`.

## Project Structure

```
sesg-ragas/
├── code/
│   ├── 00_select_documents.py    # Document selection
│   ├── 01_generate_testset.py    # Test data generation
│   ├── 02a_compute_embeddings.py # Embedding computation (run once)
│   ├── 02_run_rag_evaluation.py  # RAG evaluation across models
│   └── 03_analyze_results.py     # Evaluation
├── data/                  # ESG document corpus
│   ├── raw/               # Source ESG documents (498 documents)
│   └── documents/         # Selected documents (copied by 00_select_documents.py)
├── outputs/               # All generated files
│   ├── 01_esg_knowledge_graph.pkl  # Knowledge graph
│   ├── 01_esg_testset.csv          # Test questions
│   ├── 02_rag_responses/          # RAG responses for each model
│   └── 03_evaluation_results/     # Evaluation results for each model
├── requirements.txt        # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Evaluation Metrics

The project uses the following RAGAS metrics:
- **Context Precision** - Relevance of retrieved context
- **Context Recall** - Completeness of retrieved context
- **Context Relevance** - How well context matches the query
- **Faithfulness** - Factual consistency of the answer
- **Factual Correctness** - Accuracy of the answer
- **Answer Relevancy** - How well the answer addresses the query

## Experimental Design

### Document Dataset
- **Dataset Size**: 100 ESG documents selected from the full corpus of 498 documents
- **Selection Strategy**: Representative sampling across years (2010-2024) and ESG topics (Environmental, Social, Governance)
- **Purpose**: Statistically significant evaluation while maintaining computational feasibility

### Model Selection Strategy

#### Size Range Justification
**Focus Range**: 4B-14B parameter models

**Selection Criteria:**
- **Practical Relevance**: 4B-14B range represents the sweet spot for real-world applications balancing performance and computational efficiency
- **Computational Feasibility**: Models in this range can be evaluated within reasonable timeframes for academic research
- **Research Impact**: Provides meaningful insights into size scaling without prohibitive resource requirements
- **Reproducibility**: Selected sizes are accessible to most research institutions for validation

#### Model Families Evaluated

**Final Model Selection** (5 models - one per family)

**Qwen Series** (1 model - Balanced Representative)
- qwen3:4b-instruct
- **Rationale**: 4B parameter model provides optimal balance of performance and efficiency from Alibaba's comprehensive family

**Gemma Series** (1 model - Google Latest Generation)
- gemma3:4b
- **Rationale**: Google's latest generation 4B model representing cutting-edge architecture from major tech company

**Ministral Series** (1 model - European Representative)
- ministral-3:8b
- **Rationale**: 8B parameter model from leading European AI company, representing larger model performance

**Nemotron Series** (1 model - NVIDIA Enterprise)
- nemotron-3-nano:4b
- **Rationale**: NVIDIA's optimized 4B model for enterprise applications with strong engineering focus

**GLM Series** (1 model - Quantized Efficiency)
- glm-4.7-flash:q4_K_M
- **Rationale**: Quantized 4B model representing efficiency-focused approaches from Zhipu AI

**Selection Strategy:**
- **Final Implementation**: 5 models across 5 families for comprehensive evaluation
- **Size Range**: 4B-8B parameters covering practical deployment scenarios
- **Family Diversity**: Each model represents a major AI research organization/geographic region
- **Evaluation Time**: ~9.5 hours total for complete 498-document evaluation
- **Maintains**: Architectural diversity while ensuring computational feasibility

#### Research Design Benefits

**Size Range Evaluated**:
- 4B parameters: qwen3:4b-instruct, gemma3:4b, nemotron-3-nano:4b, glm-4.7-flash:q4_K_M (quantized)
- 8B parameters: ministral-3:8b

**Family Comparison**:
- **Architectural Diversity**: Different training methodologies and model architectures
- **Geographic Diversity**: Models from US, Europe, and China
- **Training Paradigms**: Various approaches to LLM development

**Sample Size**:
- **5 models** across **5 families** for cross-family comparison
- **100 synthetic QA pairs** (Environmental: 97 / Social: 94 / Governance: 93 in the 284-pair generation pool; first 100 used for evaluation)
- **500 model-query evaluations** in total

**Computational Footprint**:
- **Total evaluation time**: ~9.5 hours for the full 498-document setup across all 5 models
- **Resource requirements**: Manageable for academic research settings (RTX 6000 Ada, 48 GB VRAM)

**Total Models**: 5 open-source models across 5 families.

## Output Files

- `outputs/01_esg_knowledge_graph.pkl` - Knowledge graph of ESG documents
- `outputs/01_esg_testset.csv` - Generated test questions and references
- `outputs/02_rag_responses/` - RAG responses for each model
- `outputs/03_evaluation_results/` - Evaluation results for each model

## Notes

- Ensure Ollama is running before executing the scripts
- The evaluation process can be time-consuming due to multiple models
- Results are saved in JSON format for further analysis
