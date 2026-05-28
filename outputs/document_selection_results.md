# Document Selection Results for ESG RAG Experiments

## Overview
This document summarizes the document selection process for the ESG RAG evaluation experiments. A strategic sampling approach was used to select 100 documents from the full corpus of 498 ESG documents.

## Selection Strategy

### Corpus Analysis
- **Total available documents**: 498
- **Year range**: 2010-2024 (15 years)
- **Original size distribution**: 
  - Mean: 425,897 bytes
  - Median: 295,596 bytes
  - Min: 2,396 bytes
  - Max: 2,788,503 bytes

### Selection Method
1. **Stratified Sampling**: Documents were sampled to ensure balanced representation across years
2. **Size Diversity**: Documents were selected from different size quartiles to ensure variety in document complexity
3. **Temporal Coverage**: Equal representation across the 15-year period
4. **Random Selection**: Within each stratum, random selection ensured unbiased sampling

## Results

### Selected Dataset
- **Documents selected**: 100
- **Dataset location**: `markdown_docs_100/`
- **Selection info**: `outputs/document_selection_info.csv`

### Temporal Distribution
| Year | Documents Selected | Available in Corpus |
|------|-------------------|---------------------|
| 2010 | 7 | 16 |
| 2011 | 7 | 20 |
| 2012 | 7 | 17 |
| 2013 | 7 | 22 |
| 2014 | 7 | 24 |
| 2015 | 7 | 25 |
| 2016 | 7 | 29 |
| 2017 | 7 | 28 |
| 2018 | 7 | 32 |
| 2019 | 7 | 32 |
| 2020 | 6 | 43 |
| 2021 | 6 | 40 |
| 2022 | 6 | 39 |
| 2023 | 6 | 57 |
| 2024 | 6 | 74 |
| **Total** | **100** | **498** |

### Size Distribution (Selected Documents)
- **Mean**: 282,119 bytes
- **Median**: 230,348 bytes  
- **Min**: 11,137 bytes
- **Max**: 1,482,863 bytes

## Selection Benefits

### Statistical Significance
- **Sample size**: 100 documents provides statistically significant results for paper publication
- **Confidence interval**: Adequate for reliable performance evaluation across models

### Computational Feasibility
- **Processing time**: Manageable for multiple model evaluations
- **Resource requirements**: Balanced between comprehensive evaluation and practical constraints

### Research Quality
- **Temporal coverage**: Spans 15 years of ESG reporting evolution
- **Size diversity**: Tests model performance across different document complexities
- **Representative sampling**: Ensures results generalize to the full corpus

## Next Steps

1. **Update Data Generation**: Modify `01_generate_testset.py` to use `markdown_docs_100/` directory
2. **Model Evaluation**: Update `02_run_rag_evaluation.py` with 12 selected models in 4B-14B range
3. **Run Experiments**: Execute full evaluation pipeline with expanded dataset and model set

## Files Generated

- `markdown_docs_100/` - Selected documents for experiments
- `outputs/document_selection_info.csv` - Detailed selection information
- `outputs/selected_documents.json` - Selection metadata in JSON format
- `outputs/document_selection_results.md` - This summary document

---
*Generated on: 2025-04-15*  
*Purpose: ESG RAG evaluation experiments for academic publication*
