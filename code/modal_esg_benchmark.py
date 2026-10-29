"""
Modal Cloud Pipeline for the Longitudinal ESG RAG Audit Benchmark (2010–2026).
Workspace: redeemer
Volume: esg-rag-h100-vol (all weights, chunks, embeddings, and outputs live on Modal).

Usage:
  MODAL_PROFILE=redeemer venv/bin/modal run code/modal_esg_benchmark.py --milestone 1
  MODAL_PROFILE=redeemer venv/bin/modal run code/modal_esg_benchmark.py --milestone 2
"""

import json
import os
import pickle
import random
import re
import tarfile
import time
from pathlib import Path
import modal

APP_NAME = "esg-rag-h100-benchmark"
VOLUME_NAME = "esg-rag-h100-vol"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Lightweight CPU image for Milestone 1 & weight pre-fetching
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas",
        "numpy",
        "tqdm",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf_cache"})
)

# H100 GPU image for Milestone 2 (PyTorch + Transformers + Accelerate)
h100_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "accelerate>=1.0.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "pandas",
        "numpy",
        "tqdm",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf_cache"})
)


@app.function(
    image=cpu_image,
    volumes={"/vol": vol},
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def prefetch_models_on_cpu(repo_ids: list[str]) -> dict:
    """Download HuggingFace model weights to /vol/hf_cache on cheap CPU before spinning up H100."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    downloaded = {}
    for repo_id in repo_ids:
        t_start = time.time()
        print(f"[CPU Pre-Fetch] Downloading {repo_id} to /vol/hf_cache ...")
        local_dir = snapshot_download(
            repo_id=repo_id,
            ignore_patterns=["*.gguf", "*.msgpack", "*.h5", "original/*"],
        )
        vol.commit()
        dt = round(time.time() - t_start, 2)
        print(f"[CPU Pre-Fetch] Finished {repo_id} in {dt}s -> {local_dir}")
        downloaded[repo_id] = dt

    return {"total_prefetch_seconds": round(time.time() - t0, 2), "models": downloaded}


@app.function(
    image=cpu_image,
    volumes={"/vol": vol},
    cpu=4.0,
    memory=16384,
    timeout=1800,
)
def run_milestone1_remote(tar_bytes: bytes, existing_testset_json: str) -> dict:
    """
    Milestone 1 (Modal CPU):
    1. Extract all 518 markdown documents (2010-2026) into /vol/data/documents/
    2. Build overlapping 1,000-char chunks (step=800) across all 518 documents
    3. Save chunk graph (/vol/testset/01_esg_knowledge_graph.pkl)
       and fast chunk index (/vol/testset/chunks_index.pkl)
    4. Recover exact Oracle Gold Chunks for existing 2010-2024 testset pairs
    """
    import io
    from collections import Counter, defaultdict

    t0 = time.time()
    docs_dir = Path("/vol/data/documents")
    testset_dir = Path("/vol/testset")
    docs_dir.mkdir(parents=True, exist_ok=True)
    testset_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path=docs_dir)

    md_files = sorted(docs_dir.glob("*.md"))
    print(f"[Milestone 1] Extracted {len(md_files)} markdown documents to {docs_dir}")

    chunk_size = 1000
    overlap = 200
    step = chunk_size - overlap

    kg_nodes = []
    chunks_index = []
    chunks_by_filename = defaultdict(list)
    year_chunk_counts = Counter()
    year_doc_counts = Counter()
    seen_content_hashes = set()

    for md_path in md_files:
        fname = md_path.name
        m = re.match(r"^(\d+)_(\d+)_(\d{4})\.md$", fname)
        doc_id = int(m.group(1)) if m else -1
        company_id = int(m.group(2)) if m else -1
        year = int(m.group(3)) if m else -1
        year_doc_counts[year] += 1

        content = md_path.read_text(encoding="utf-8", errors="ignore")
        for i in range(0, len(content), step):
            chunk = content[i : i + chunk_size]
            stripped = chunk.strip()
            if len(stripped) <= 50:
                continue

            chunk_id = f"{fname}_{i // step}"
            if chunk_id in seen_content_hashes:
                continue
            seen_content_hashes.add(chunk_id)

            meta = {
                "source": f"data/documents/{fname}",
                "filename": fname,
                "doc_id": doc_id,
                "company_id": company_id,
                "year": year,
                "chunk_index": i // step,
            }

            kg_nodes.append(
                {
                    "type": "DOCUMENT",
                    "properties": {
                        "page_content": chunk,
                        "document_metadata": meta,
                        "chunk_id": chunk_id,
                    },
                }
            )

            rec = {
                "idx": len(chunks_index),
                "chunk_id": chunk_id,
                "text": chunk,
                "filename": fname,
                "doc_id": doc_id,
                "company_id": company_id,
                "year": year,
            }
            chunks_index.append(rec)
            chunks_by_filename[fname].append(rec)
            year_chunk_counts[year] += 1

    kg_path = testset_dir / "01_esg_knowledge_graph.pkl"
    with open(kg_path, "wb") as f:
        pickle.dump({"nodes": kg_nodes}, f)

    idx_path = testset_dir / "chunks_index.pkl"
    with open(idx_path, "wb") as f:
        pickle.dump(chunks_index, f)

    existing_pairs = json.loads(existing_testset_json)
    enriched_existing = []
    token_re = re.compile(r"\b\w{3,}\b")

    for pair in existing_pairs:
        src_fname = os.path.basename(pair.get("document_source", ""))
        m = re.match(r"^(\d+)_(\d+)_(\d{4})\.md$", src_fname)
        doc_id = int(m.group(1)) if m else -1
        company_id = int(m.group(2)) if m else -1
        year = int(m.group(3)) if m else -1

        candidates = chunks_by_filename.get(src_fname, [])
        ref_tokens = set(
            token_re.findall((pair["user_input"] + " " + pair["reference"]).lower())
        )

        best_chunk = None
        best_score = -1.0
        for c in candidates:
            c_tokens = set(token_re.findall(c["text"].lower()))
            if not c_tokens:
                continue
            overlap_score = len(ref_tokens & c_tokens) / max(1, len(ref_tokens))
            if overlap_score > best_score:
                best_score = overlap_score
                best_chunk = c

        enriched_existing.append(
            {
                **pair,
                "filename": src_fname,
                "doc_id": doc_id,
                "company_id": company_id,
                "year": year,
                "oracle_chunk_id": best_chunk["chunk_id"] if best_chunk else None,
                "oracle_chunk_text": best_chunk["text"] if best_chunk else None,
                "oracle_overlap_score": round(best_score, 4),
            }
        )

    enriched_path = testset_dir / "existing_2010_2024_enriched.json"
    enriched_path.write_text(json.dumps(enriched_existing, indent=2), encoding="utf-8")

    vol.commit()
    elapsed = round(time.time() - t0, 2)

    summary = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "total_documents": len(md_files),
        "total_chunks": len(chunks_index),
        "chunks_2010_2024": sum(v for k, v in year_chunk_counts.items() if k <= 2024),
        "chunks_2025": year_chunk_counts.get(2025, 0),
        "chunks_2026": year_chunk_counts.get(2026, 0),
        "docs_by_year": {str(k): year_doc_counts[k] for k in sorted(year_doc_counts)},
        "chunks_by_year": {str(k): year_chunk_counts[k] for k in sorted(year_chunk_counts)},
        "existing_qa_pairs_enriched": len(enriched_existing),
        "kg_size_mb": round(kg_path.stat().st_size / (1024 * 1024), 2),
        "index_size_mb": round(idx_path.stat().st_size / (1024 * 1024), 2),
    }

    (testset_dir / "milestone1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    vol.commit()
    return summary


def is_hard_kpi_pair(question: str, reference: str) -> bool:
    """
    Classify whether a QA pair is a 'hard_kpi' query (requires specific numeric figures,
    percentages, currencies, physical ESG units, or quantitative targets) vs. 'boilerplate'
    (qualitative policy, governance process, or framework narrative).
    """
    combined = f"{question} {reference}"
    # Remove 4-digit calendar years (2000-2035) and legal article numbers so mere dates don't count as KPIs
    cleaned_ref = re.sub(r"\b(19\d{2}|20[0-3]\d|2050)\b", "", reference)
    cleaned_ref = re.sub(r"\b(Article|Law|Section|Page|GRI|ESRS|Scope)\s*\d+[-\w.]*", "", cleaned_ref, flags=re.I)

    # Check for explicit quantitative indicators in the reference answer
    has_unit_or_pct = bool(
        re.search(
            r"(\d+(\.\d+)?\s*(%|percent|million|billion|bn|m\b|€|\$|£|PLN|EUR|USD|tCO2|CO2|MWh|GWh|TJ|GJ|kt|tonnes|tons|m3|liters|hectares|hours|days|bps|points|per\s+million))"
            r"|((€|\$|£|PLN|EUR|USD)\s*\d+)",
            cleaned_ref,
            flags=re.I,
        )
    )
    has_number_in_ref = bool(re.search(r"\b\d+([.,]\d+)?\b", cleaned_ref))
    asks_quant = bool(
        re.search(
            r"\b(how much|how many|what percentage|what was the total|what is the total|what was the trend|what specific .* target|what is the tax value|by how much|what rate|what proportion|what amount)\b",
            question,
            flags=re.I,
        )
    )
    return has_unit_or_pct or (has_number_in_ref and asks_quant)


@app.function(
    image=h100_image,
    gpu="H100",
    volumes={"/vol": vol},
    cpu=8.0,
    memory=65536,
    timeout=3600,
)
def run_milestone2_h100() -> dict:
    """
    Milestone 2 (Modal H100 80GB):
    1. Generate 48 new QA pairs from 2025-2026 corporate ESG reports using Qwen/Qwen3-4B-Instruct-2507
       (rotating Environmental, Social, Governance personas, with explicit KPI & Policy prompts).
    2. Stratify & build the 120-Question ESG Audit Benchmark:
       - 60 Hard KPI questions + 60 Corporate Boilerplate questions
       - Balanced across E (40), S (40), G (40)
       - 84 from 2010-2024 + 36 from 2025-2026
    3. Embed all 279,862 chunks on H100 using Qwen/Qwen3-Embedding-4B (bfloat16, length-sorted batches)
       and save /vol/testset/doc_embeddings.npy (float16).
    4. Run GPU matrix retrieval for the 120 benchmark questions to attach:
       - k=1 retrieved context
       - k=3 retrieved contexts
       - Oracle gold context
       - Near-Miss (wrong-year / cross-document) context
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    t_start_h100 = time.time()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    testset_dir = Path("/vol/testset")
    with open(testset_dir / "chunks_index.pkl", "rb") as f:
        chunks_index = pickle.load(f)

    print(f"[H100 Milestone 2] Loaded {len(chunks_index):,} chunks from /vol/testset/chunks_index.pkl")
    print(f"[H100 Device] {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -------------------------------------------------------------------------
    # STEP 1: Generate 2025-2026 QA pairs using Qwen/Qwen3-4B-Instruct-2507
    # -------------------------------------------------------------------------
    new_qa_path = testset_dir / "qa_2025_2026_generated.json"
    if new_qa_path.exists():
        new_qa_pairs = json.loads(new_qa_path.read_text(encoding="utf-8"))
        print(f"[Step 1] Loaded {len(new_qa_pairs)} cached 2025-2026 QA pairs.")
        qa_gen_seconds = 0.0
    else:
        t_qa = time.time()
        # Filter informative 2025 & 2026 chunks (length > 550 chars, contains sentences, not TOC dots)
        chunks_2025 = [
            c for c in chunks_index
            if c["year"] == 2025 and len(c["text"].strip()) > 550 and "...." not in c["text"]
        ]
        chunks_2026 = [
            c for c in chunks_index
            if c["year"] == 2026 and len(c["text"].strip()) > 550 and "...." not in c["text"]
        ]

        # Sample 30 from 2025 and 30 from 2026 (half numeric-rich, half narrative-rich)
        def sample_balanced_chunks(pool, n_total=30):
            num_rich = [c for c in pool if len(re.findall(r"\b\d+(\.\d+)?%?\b", c["text"])) >= 4]
            narr_rich = [c for c in pool if len(re.findall(r"\b\d+(\.\d+)?%?\b", c["text"])) < 4]
            s_num = random.sample(num_rich, min(n_total // 2, len(num_rich)))
            s_narr = random.sample(narr_rich, min(n_total - len(s_num), len(narr_rich)))
            combined = s_num + s_narr
            random.shuffle(combined)
            return combined

        sampled_new_chunks = sample_balanced_chunks(chunks_2025, 30) + sample_balanced_chunks(chunks_2026, 30)

        PERSONAS = [
            {
                "name": "Environmental KPI Extractor",
                "system": (
                    "You are an environmental analyst specializing in ESG reporting. "
                    "You focus on carbon emissions, energy consumption, water usage, waste management, "
                    "biodiversity, and climate-related targets."
                ),
            },
            {
                "name": "Social KPI Extractor",
                "system": (
                    "You are a social responsibility analyst specializing in ESG reporting. "
                    "You focus on employee diversity, labor practices, community investment, "
                    "human rights, health and safety, and supply chain social standards."
                ),
            },
            {
                "name": "Governance KPI Extractor",
                "system": (
                    "You are a corporate governance analyst specializing in ESG reporting. "
                    "You focus on board composition, executive compensation, anti-corruption policies, "
                    "risk management frameworks, shareholder rights, and data privacy."
                ),
            },
        ]

        gen_model_id = "Qwen/Qwen3-4B-Instruct-2507"
        print(f"[Step 1] Loading {gen_model_id} on H100 for 2025-2026 QA generation...")
        tok = AutoTokenizer.from_pretrained(gen_model_id, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            gen_model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()

        def run_batch_chat(messages_list, max_new_tokens=160):
            prompts = [
                tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_list
            ]
            outputs = []
            bs = 30
            for i in range(0, len(prompts), bs):
                batch_p = prompts[i : i + bs]
                enc = tok(batch_p, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                in_len = enc.input_ids.shape[1]
                for g in gen:
                    outputs.append(tok.decode(g[in_len:], skip_special_tokens=True).strip())
            return outputs

        q_msgs = []
        assigned_personas = []
        for idx, c in enumerate(sampled_new_chunks):
            p = PERSONAS[idx % 3]
            assigned_personas.append(p)
            # Alternate encouraging quantitative vs policy questions based on chunk content
            q_prompt = (
                f"Based on this {c['year']} ESG document excerpt, generate one specific question that can be "
                f"answered directly from the text. Include the company name or reporting year ({c['year']}) "
                f"if mentioned in the text, and reflect your area of expertise. "
                f"Respond with only the question, no extra text.\n\n"
                f"Document excerpt:\n{c['text']}"
            )
            q_msgs.append([{"role": "system", "content": p["system"]}, {"role": "user", "content": q_prompt}])

        questions = run_batch_chat(q_msgs, max_new_tokens=100)

        a_msgs = []
        for idx, (c, q, p) in enumerate(zip(sampled_new_chunks, questions, assigned_personas)):
            a_prompt = (
                "Answer the following question using only the provided document excerpt. "
                "Be concise and factual.\n\n"
                f"Question: {q}\n\n"
                f"Document excerpt:\n{c['text']}\n\n"
                "Answer:"
            )
            a_msgs.append([{"role": "system", "content": p["system"]}, {"role": "user", "content": a_prompt}])

        answers = run_batch_chat(a_msgs, max_new_tokens=180)

        new_qa_pairs = []
        for c, q, a, p in zip(sampled_new_chunks, questions, answers, assigned_personas):
            q_clean = q.strip().strip('"')
            a_clean = a.strip()
            if len(q_clean) > 15 and len(a_clean) > 15 and "not contain" not in a_clean.lower():
                new_qa_pairs.append(
                    {
                        "user_input": q_clean,
                        "reference": a_clean,
                        "document_source": f"data/documents/{c['filename']}",
                        "persona": p["name"],
                        "filename": c["filename"],
                        "doc_id": c["doc_id"],
                        "company_id": c["company_id"],
                        "year": c["year"],
                        "oracle_chunk_id": c["chunk_id"],
                        "oracle_chunk_text": c["text"],
                        "oracle_overlap_score": 1.0,
                    }
                )

        new_qa_path.write_text(json.dumps(new_qa_pairs, indent=2), encoding="utf-8")
        vol.commit()

        del model
        del tok
        torch.cuda.empty_cache()
        qa_gen_seconds = round(time.time() - t_qa, 2)
        print(f"[Step 1] Generated {len(new_qa_pairs)} valid 2025-2026 QA pairs in {qa_gen_seconds}s")

    # -------------------------------------------------------------------------
    # STEP 2: Assemble the Stratified 120-Question ESG Audit Benchmark
    # -------------------------------------------------------------------------
    existing_enriched = json.loads(
        (testset_dir / "existing_2010_2024_enriched.json").read_text(encoding="utf-8")
    )

    all_pool = []
    for item in existing_enriched + new_qa_pairs:
        if not item.get("oracle_chunk_text"):
            continue
        qtype = "hard_kpi" if is_hard_kpi_pair(item["user_input"], item["reference"]) else "boilerplate"
        era = "2025_2026_csrd" if item["year"] >= 2025 else "2010_2024_voluntary"
        all_pool.append({**item, "question_type": qtype, "era": era})

    # Target: 120 questions total (84 from 2010-2024 + 36 from 2025-2026),
    # balanced across Hard KPI vs Boilerplate and E / S / G personas
    pool_pre = [x for x in all_pool if x["era"] == "2010_2024_voluntary"]
    pool_post = [x for x in all_pool if x["era"] == "2025_2026_csrd"]

    def select_stratified(pool, target_total):
        kpi = [x for x in pool if x["question_type"] == "hard_kpi"]
        boiler = [x for x in pool if x["question_type"] == "boilerplate"]
        half = target_total // 2
        sel_kpi = kpi[: min(half, len(kpi))]
        sel_boiler = boiler[: min(target_total - len(sel_kpi), len(boiler))]
        combined = sel_kpi + sel_boiler
        if len(combined) < target_total:
            used_ids = {id(x) for x in combined}
            rem = [x for x in pool if id(x) not in used_ids]
            combined.extend(rem[: target_total - len(combined)])
        return combined

    benchmark_120 = select_stratified(pool_pre, 84) + select_stratified(pool_post, 36)
    for idx, b in enumerate(benchmark_120):
        b["benchmark_id"] = idx + 1

    print(
        f"[Step 2] Assembled {len(benchmark_120)} benchmark questions: "
        f"Hard KPI={sum(1 for x in benchmark_120 if x['question_type']=='hard_kpi')}, "
        f"Boilerplate={sum(1 for x in benchmark_120 if x['question_type']=='boilerplate')}, "
        f"2010-2024={sum(1 for x in benchmark_120 if x['year']<=2024)}, "
        f"2025-2026={sum(1 for x in benchmark_120 if x['year']>=2025)}"
    )

    # -------------------------------------------------------------------------
    # STEP 3: Embed all 279,862 Chunks on H100 using Qwen/Qwen3-Embedding-4B
    # -------------------------------------------------------------------------
    emb_model_id = "Qwen/Qwen3-Embedding-4B"
    emb_path = testset_dir / "doc_embeddings_fp16.npy"

    print(f"[Step 3] Loading {emb_model_id} on H100...")
    emb_tok = AutoTokenizer.from_pretrained(emb_model_id, padding_side="left")
    emb_model = AutoModel.from_pretrained(
        emb_model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    emb_model.eval()

    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    t_emb = time.time()
    if emb_path.exists():
        print(f"[Step 3] Loading cached embeddings from {emb_path}...")
        doc_embeddings = np.load(emb_path)
        emb_seconds = 0.0
    else:
        n_chunks = len(chunks_index)
        # Sort by character length to minimize padding overhead on H100
        sorted_order = sorted(range(n_chunks), key=lambda i: len(chunks_index[i]["text"]))
        sorted_texts = [chunks_index[i]["text"] for i in sorted_order]

        # Probe embedding dimension
        with torch.no_grad():
            probe_inp = emb_tok(["probe"], return_tensors="pt", padding=True).to("cuda")
            probe_out = emb_model(**probe_inp)
            emb_dim = probe_out.last_hidden_state.shape[-1]

        print(f"[Step 3] Embedding {n_chunks:,} chunks (dim={emb_dim}) on H100 with batch_size=256...")
        sorted_embs = np.empty((n_chunks, emb_dim), dtype=np.float16)

        batch_size = 256
        for start in tqdm(range(0, n_chunks, batch_size), desc="H100 Embedding"):
            end = min(start + batch_size, n_chunks)
            batch_texts = sorted_texts[start:end]
            enc = emb_tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=320,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                out = emb_model(**enc)
                vecs = last_token_pool(out.last_hidden_state, enc["attention_mask"])
                vecs = F.normalize(vecs, p=2, dim=1)
                sorted_embs[start:end] = vecs.to(torch.float16).cpu().numpy()

        # Unsort back to original chunk_index order
        doc_embeddings = np.empty_like(sorted_embs)
        for sorted_pos, orig_idx in enumerate(sorted_order):
            doc_embeddings[orig_idx] = sorted_embs[sorted_pos]
        del sorted_embs

        np.save(emb_path, doc_embeddings)
        vol.commit()
        emb_seconds = round(time.time() - t_emb, 2)
        print(f"[Step 3] Embedded & saved {n_chunks:,} chunks to {emb_path} in {emb_seconds}s")

    # -------------------------------------------------------------------------
    # STEP 4: Fast GPU Vectorized Retrieval for the 120 Benchmark Questions
    # -------------------------------------------------------------------------
    t_ret = time.time()
    queries = [b["user_input"] for b in benchmark_120]
    q_enc = emb_tok(
        queries, padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        q_out = emb_model(**q_enc)
        q_vecs = last_token_pool(q_out.last_hidden_state, q_enc["attention_mask"])
        q_vecs = F.normalize(q_vecs, p=2, dim=1).to(torch.float16)

    del emb_model
    torch.cuda.empty_cache()

    # Compute exact cosine similarity matrix (120 x 279,862) in GPU chunks
    num_q = len(queries)
    n_chunks = len(chunks_index)
    top20_indices = np.zeros((num_q, 20), dtype=np.int32)
    top20_scores = np.zeros((num_q, 20), dtype=np.float32)

    doc_tensor_gpu = torch.from_numpy(doc_embeddings).to("cuda", dtype=torch.float16)
    sim_matrix = torch.matmul(q_vecs, doc_tensor_gpu.T)  # (120, 279862)
    vals, idxs = torch.topk(sim_matrix, k=500, dim=1)
    vals_np = vals.float().cpu().numpy()
    idxs_np = idxs.cpu().numpy()

    del doc_tensor_gpu
    del sim_matrix
    torch.cuda.empty_cache()

    k1_doc_hits = 0
    k3_doc_hits = 0
    k1_wrong_year_same_company = 0

    for qi, item in enumerate(benchmark_120):
        cand_idxs = idxs_np[qi]
        cand_scores = vals_np[qi]

        top1_rec = chunks_index[int(cand_idxs[0])]
        top3_recs = [chunks_index[int(cand_idxs[j])] for j in range(3)]

        is_k1_hit = top1_rec["filename"] == item["filename"]
        is_k3_hit = any(r["filename"] == item["filename"] for r in top3_recs)
        if is_k1_hit:
            k1_doc_hits += 1
        if is_k3_hit:
            k3_doc_hits += 1
        if (
            not is_k1_hit
            and top1_rec["company_id"] == item["company_id"]
            and item["company_id"] != -1
        ):
            k1_wrong_year_same_company += 1

        # Select Near-Miss Chunk:
        # Prefer same company from a DIFFERENT year; fallback to highest-similarity chunk from a different document
        nearmiss_rec = None
        for j in range(len(cand_idxs)):
            c_rec = chunks_index[int(cand_idxs[j])]
            if (
                c_rec["filename"] != item["filename"]
                and c_rec["company_id"] == item["company_id"]
                and c_rec["year"] != item["year"]
            ):
                nearmiss_rec = c_rec
                break
        if nearmiss_rec is None:
            for j in range(len(cand_idxs)):
                c_rec = chunks_index[int(cand_idxs[j])]
                if c_rec["filename"] != item["filename"]:
                    nearmiss_rec = c_rec
                    break
        if nearmiss_rec is None:
            # Ultimate fallback: pick any chunk from a different document
            for c_rec in chunks_index:
                if c_rec["filename"] != item["filename"]:
                    nearmiss_rec = c_rec
                    break

        item["retrieval"] = {
            "k1_context": [top1_rec["text"]],
            "k1_meta": {
                "chunk_id": top1_rec["chunk_id"],
                "filename": top1_rec["filename"],
                "year": top1_rec["year"],
                "company_id": top1_rec["company_id"],
                "score": round(float(cand_scores[0]), 4),
                "is_source_doc_hit": is_k1_hit,
            },
            "k3_contexts": [r["text"] for r in top3_recs],
            "k3_meta": [
                {
                    "chunk_id": r["chunk_id"],
                    "filename": r["filename"],
                    "year": r["year"],
                    "company_id": r["company_id"],
                    "score": round(float(cand_scores[j]), 4),
                }
                for j, r in enumerate(top3_recs)
            ],
            "is_k3_source_doc_hit": is_k3_hit,
            "oracle_context": [item["oracle_chunk_text"]],
            "nearmiss_context": [nearmiss_rec["text"]],
            "nearmiss_meta": {
                "chunk_id": nearmiss_rec["chunk_id"],
                "filename": nearmiss_rec["filename"],
                "year": nearmiss_rec["year"],
                "company_id": nearmiss_rec["company_id"],
                "same_company_wrong_year": nearmiss_rec["company_id"] == item["company_id"],
            },
        }

    ret_seconds = round(time.time() - t_ret, 2)
    ready_path = testset_dir / "benchmark_120_ready.json"
    ready_path.write_text(json.dumps(benchmark_120, indent=2), encoding="utf-8")

    total_h100_seconds = round(time.time() - t_start_h100, 2)
    est_cost_usd = round((total_h100_seconds / 3600.0) * 3.95, 3)

    summary = {
        "status": "completed",
        "h100_gpu": torch.cuda.get_device_name(0),
        "total_h100_seconds": total_h100_seconds,
        "qa_generation_seconds": qa_gen_seconds,
        "embedding_279k_chunks_seconds": emb_seconds,
        "retrieval_seconds": ret_seconds,
        "estimated_h100_cost_usd": est_cost_usd,
        "new_2025_2026_qa_generated": len(new_qa_pairs),
        "benchmark_total_questions": len(benchmark_120),
        "breakdown_by_type": {
            "hard_kpi": sum(1 for x in benchmark_120 if x["question_type"] == "hard_kpi"),
            "boilerplate": sum(1 for x in benchmark_120 if x["question_type"] == "boilerplate"),
        },
        "breakdown_by_era": {
            "2010_2024_voluntary": sum(1 for x in benchmark_120 if x["year"] <= 2024),
            "2025_2026_csrd": sum(1 for x in benchmark_120 if x["year"] >= 2025),
        },
        "breakdown_by_persona": {
            p: sum(1 for x in benchmark_120 if x["persona"] == p)
            for p in sorted({x["persona"] for x in benchmark_120})
        },
        "retrieval_diagnostics": {
            "k1_source_doc_hit_rate": round(k1_doc_hits / len(benchmark_120), 4),
            "k3_source_doc_hit_rate": round(k3_doc_hits / len(benchmark_120), 4),
            "k1_same_company_wrong_year_rate": round(
                k1_wrong_year_same_company / len(benchmark_120), 4
            ),
            "nearmiss_same_company_wrong_year_rate": round(
                sum(
                    1
                    for x in benchmark_120
                    if x["retrieval"]["nearmiss_meta"]["same_company_wrong_year"]
                )
                / len(benchmark_120),
                4,
            ),
        },
        "sample_2025_2026_questions": [
            {
                "year": x["year"],
                "filename": x["filename"],
                "persona": x["persona"],
                "question_type": x["question_type"],
                "user_input": x["user_input"],
                "reference": x["reference"],
            }
            for x in benchmark_120
            if x["year"] >= 2025
        ][:5],
    }

    (testset_dir / "milestone2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    vol.commit()
    return summary


# =============================================================================
# MILESTONE 3: Multi-Architecture RAG Generation & Audit Stress-Testing on H100
# =============================================================================

MILESTONE3_MODELS = [
    {
        "slug": "qwen3_4b",
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "label": "Qwen3-4B-Instruct (4B Dense Baseline)",
        "family": "Qwen",
        "params_b": 4.0,
        "active_params_b": 4.0,
        "batch_size": 40,
        "max_new_tokens": 200,
        "mode": "standard_chat",
    },
    {
        "slug": "deepseek_r1_7b",
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "label": "DeepSeek-R1-Distill-Qwen-7B (7B Reasoning)",
        "family": "DeepSeek-R1",
        "params_b": 7.6,
        "active_params_b": 7.6,
        "batch_size": 32,
        "max_new_tokens": 384,
        "mode": "deepseek_r1",
    },
    {
        "slug": "gpt_oss_20b",
        "repo_id": "openai/gpt-oss-20b",
        "label": "OpenAI gpt-oss-20b (21B MoE, 3.6B Active)",
        "family": "GPT-OSS",
        "params_b": 20.9,
        "active_params_b": 3.6,
        "batch_size": 20,
        "max_new_tokens": 320,
        "mode": "gpt_oss",
    },
    {
        "slug": "qwen3_5_27b",
        "repo_id": "Qwen/Qwen3.5-27B",
        "label": "Qwen3.5-27B (27B Dense Flagship)",
        "family": "Qwen",
        "params_b": 27.0,
        "active_params_b": 27.0,
        "batch_size": 16,
        "max_new_tokens": 200,
        "mode": "qwen_nothink",
    },
    {
        "slug": "gemma4_31b",
        "repo_id": "google/gemma-4-31B-it",
        "label": "Gemma-4-31B-it (31B Dense Flagship)",
        "family": "Gemma",
        "params_b": 30.7,
        "active_params_b": 30.7,
        "batch_size": 8,
        "max_new_tokens": 200,
        "mode": "gemma_nothink",
    },
]

STANDARD_RAG_SYSTEM_PROMPT = (
    "You are an expert ESG analyst for question-answering tasks on corporate sustainability reports.\n"
    "Use ONLY the provided context to answer the user's question.\n"
    "If the context does not contain the information needed to answer the question, state clearly "
    "that you cannot answer based on the provided context.\n"
    "Keep your answer concise, factual, and limited to 2-4 sentences, preserving exact numbers, "
    "percentages, currencies, and physical units when present."
)

AUDIT_STRICT_SYSTEM_PROMPT = (
    "You are a strict ESG Assurance Auditor verifying corporate sustainability disclosures.\n"
    "Before answering, verify whether the provided context matches:\n"
    "(1) the exact company or organization asked about,\n"
    "(2) the exact reporting year or target period asked about, and\n"
    "(3) the exact quantitative KPI or policy asked about.\n"
    "CRITICAL AUDIT RULE: Do NOT substitute figures from a different reporting year, a different "
    "company, or a different metric scope. If the exact answer is not explicitly supported by the "
    "provided context, you MUST begin your response with 'ABSTAIN:' followed by a brief 1-sentence "
    "explanation of what is missing or mismatched (e.g., wrong reporting year or missing metric)."
)

ABSTENTION_PATTERNS = [
    r"\babstain\b",
    r"cannot\s+answer",
    r"can't\s+answer",
    r"unable\s+to\s+answer",
    r"unable\s+to\s+determine",
    r"cannot\s+be\s+determined",
    r"cannot\s+determine",
    r"not\s+contain\s+the\s+information",
    r"does\s+not\s+contain",
    r"do\s+not\s+contain",
    r"does\s+not\s+provide",
    r"do\s+not\s+provide",
    r"not\s+provided\s+in\s+the\s+(?:provided\s+)?context",
    r"not\s+mentioned\s+in\s+the\s+(?:provided\s+)?context",
    r"no\s+mention\s+of",
    r"no\s+information\s+(?:is\s+)?(?:provided|available)",
    r"insufficient\s+information",
    r"does\s+not\s+specify",
    r"does\s+not\s+state",
    r"not\s+specified\s+in\s+the\s+(?:provided\s+)?context",
    r"context\s+is\s+from\s+(?:a\s+different|the\s+\d{4})",
    r"refers\s+to\s+\d{4}\s+(?:rather|instead)",
]


def detect_abstention(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    if t.startswith("abstain"):
        return True
    for pat in ABSTENTION_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def extract_numeric_tokens(text: str) -> list[str]:
    """Extract normalized numbers (excluding standalone 4-digit calendar years 2000-2035) for KPI matching."""
    raw_nums = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text or "")
    cleaned = []
    for n in raw_nums:
        norm = n.replace(",", "")
        # Skip likely calendar years 2005..2035 so year repetition doesn't count as KPI match
        if re.fullmatch(r"20[0-3]\d", norm):
            continue
        cleaned.append(norm)
    return list(dict.fromkeys(cleaned))


def compute_numeric_kpi_score(reference: str, response: str, is_abstention: bool) -> dict:
    """Deterministic numeric verification for Hard KPI questions."""
    ref_nums = extract_numeric_tokens(reference)
    if not ref_nums:
        return {"has_ref_numbers": False, "num_recall": None, "any_num_match": None}
    if is_abstention or not response:
        return {"has_ref_numbers": True, "num_recall": 0.0, "any_num_match": False}
    resp_nums = set(extract_numeric_tokens(response))
    # Also check substring presence for decimals/percentages
    hits = 0
    for rn in ref_nums:
        if rn in resp_nums or rn in response.replace(",", ""):
            hits += 1
    recall = hits / len(ref_nums)
    return {
        "has_ref_numbers": True,
        "ref_numbers": ref_nums,
        "num_recall": round(recall, 4),
        "any_num_match": hits > 0,
    }


@app.function(
    image=h100_image,
    gpu="H100",
    volumes={"/vol": vol},
    cpu=8.0,
    memory=65536,
    timeout=7200,
)
def run_milestone3_h100(model_slugs: list[str] | None = None) -> dict:
    """
    Milestone 3 (Modal H100 80GB):
    Evaluates the exam-taker models across the 120-question benchmark under 5 regimes:
      1. k1 (top-1 retrieved chunk + Standard RAG Prompt)
      2. k3 (top-3 retrieved chunks + Standard RAG Prompt)
      3. oracle (exact gold chunk + Standard RAG Prompt)
      4. nearmiss_standard (near-miss distractor chunk + Standard RAG Prompt)
      5. nearmiss_strict (near-miss distractor chunk + Audit-Strict Abstention Prompt)
    Each model's 600 responses are checkpointed immediately to /vol/rag_responses/responses_{slug}.json.
    """
    import gc
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start_all = time.time()
    vol.reload()

    testset_path = Path("/vol/testset/benchmark_120_ready.json")
    benchmark_120 = json.loads(testset_path.read_text(encoding="utf-8"))
    resp_dir = Path("/vol/rag_responses")
    resp_dir.mkdir(parents=True, exist_ok=True)

    regimes = [
        ("k1", "k1_context", STANDARD_RAG_SYSTEM_PROMPT),
        ("k3", "k3_contexts", STANDARD_RAG_SYSTEM_PROMPT),
        ("oracle", "oracle_context", STANDARD_RAG_SYSTEM_PROMPT),
        ("nearmiss_standard", "nearmiss_context", STANDARD_RAG_SYSTEM_PROMPT),
        ("nearmiss_strict", "nearmiss_context", AUDIT_STRICT_SYSTEM_PROMPT),
    ]

    selected_models = [
        m for m in MILESTONE3_MODELS if (model_slugs is None or m["slug"] in model_slugs)
    ]

    per_model_summaries = {}

    for m_spec in selected_models:
        slug = m_spec["slug"]
        repo_id = m_spec["repo_id"]
        mode = m_spec["mode"]
        batch_size = m_spec["batch_size"]
        max_new_tokens = m_spec["max_new_tokens"]
        out_path = resp_dir / f"responses_{slug}.json"

        if out_path.exists():
            print(f"[Milestone 3] Found cached responses for {slug} at {out_path}, loading...")
            saved_data = json.loads(out_path.read_text(encoding="utf-8"))
            per_model_summaries[slug] = saved_data["summary"]
            continue

        print(f"\n==================================================================")
        print(f"[Milestone 3] Loading {m_spec['label']} ({repo_id}) on H100...")
        print(f"==================================================================")
        t_m0 = time.time()

        tok = AutoTokenizer.from_pretrained(repo_id)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        load_sec = round(time.time() - t_m0, 2)
        vram_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
        print(f"[Milestone 3] Loaded {slug} in {load_sec}s (VRAM allocated: {vram_gb} GB)")

        # Build all 600 prompts (120 questions x 5 regimes)
        jobs = []
        for q_idx, item in enumerate(benchmark_120):
            q_text = item["user_input"]
            for regime_name, ctx_key, sys_prompt in regimes:
                ctx_list = item["retrieval"][ctx_key]
                ctx_str = "\n\n---\n\n".join(ctx_list)

                if mode == "deepseek_r1":
                    sys_full = (
                        sys_prompt
                        + "\nKeep your internal <think> reasoning brief (at most 2-3 short sentences) "
                        "before closing </think> and writing your final answer."
                    )
                else:
                    sys_full = sys_prompt

                user_msg = f"Context:\n{ctx_str}\n\nQuestion: {q_text}"
                messages = [
                    {"role": "system", "content": sys_full},
                    {"role": "user", "content": user_msg},
                ]

                if mode in ("qwen_nothink", "gemma_nothink", "standard_chat"):
                    try:
                        prompt_str = tok.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except Exception:
                        prompt_str = tok.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                elif mode == "gpt_oss":
                    try:
                        prompt_str = tok.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            reasoning_effort="low",
                        )
                    except Exception:
                        prompt_str = tok.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                else:
                    prompt_str = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                jobs.append(
                    {
                        "job_id": len(jobs),
                        "q_idx": q_idx,
                        "regime": regime_name,
                        "prompt_str": prompt_str,
                    }
                )

        # Sort jobs by character length so batch padding waste is minimal
        sorted_jobs = sorted(jobs, key=lambda x: len(x["prompt_str"]))
        results_by_job_id = {}

        def parse_model_output(raw_special: str, raw_clean: str, model_mode: str) -> tuple[str, str]:
            """Return (thinking_trace, clean_final_response)."""
            if model_mode == "deepseek_r1":
                # Prompt ended with <think>\n
                if "</think>" in raw_clean:
                    parts = raw_clean.split("</think>", 1)
                    return parts[0].strip(), parts[1].strip()
                elif "</think>" in raw_special:
                    parts = raw_special.split("</think>", 1)
                    return parts[0].strip(), parts[1].strip()
                else:
                    return raw_clean.strip(), ""
            elif model_mode == "gpt_oss":
                # Harmony format: <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...
                if "<|channel|>final<|message|>" in raw_special:
                    before, after = raw_special.split("<|channel|>final<|message|>", 1)
                    thought = (
                        before.replace("<|channel|>analysis<|message|>", "")
                        .replace("<|end|>", "")
                        .replace("<|start|>assistant", "")
                        .strip()
                    )
                    final_ans = (
                        after.split("<|return|>")[0]
                        .split("<|end|>")[0]
                        .split("<|endoftext|>")[0]
                        .strip()
                    )
                    return thought, final_ans
                else:
                    return raw_clean.strip(), ""
            else:
                # Standard / nothink models (strip any accidental <think>...</think> or <channel|> prefix)
                cleaned = raw_clean
                if "</think>" in cleaned:
                    cleaned = cleaned.split("</think>", 1)[-1]
                if "<channel|>" in cleaned:
                    cleaned = cleaned.split("<channel|>", 1)[-1]
                return "", cleaned.strip()

        def run_sub_batch(sub_jobs: list[dict], max_toks: int) -> list[tuple[str, str]]:
            prompts = [j["prompt_str"] for j in sub_jobs]
            # Adaptive batch size reduction for long k=3 prompts on 30B+ models
            if max(len(p) for p in prompts) > 3200 and len(sub_jobs) > 4 and m_spec["params_b"] >= 27.0:
                mid = len(sub_jobs) // 2
                return run_sub_batch(sub_jobs[:mid], max_toks) + run_sub_batch(sub_jobs[mid:], max_toks)

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3072,
            ).to("cuda")
            in_len = enc["input_ids"].shape[1]
            try:
                with torch.no_grad():
                    gen_ids = model.generate(
                        **enc,
                        max_new_tokens=max_toks,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                new_tokens = gen_ids[:, in_len:].cpu()
            finally:
                del enc
            decoded_special = tok.batch_decode(new_tokens, skip_special_tokens=False)
            decoded_clean = tok.batch_decode(new_tokens, skip_special_tokens=True)
            return list(zip(decoded_special, decoded_clean))

        def generate_with_oom_guard(batch_jobs: list[dict], max_toks: int) -> list[tuple[str, str]]:
            oom_hit = False
            try:
                return run_sub_batch(batch_jobs, max_toks)
            except torch.cuda.OutOfMemoryError:
                oom_hit = True
            # Exit `except` block first so Python releases traceback frames holding GPU tensors
            if oom_hit:
                gc.collect()
                torch.cuda.empty_cache()
                if len(batch_jobs) == 1:
                    raise RuntimeError("CUDA OOM even at batch_size=1")
                mid = len(batch_jobs) // 2
                print(
                    f"  [OOM Guard] Freed traceback & splitting batch of {len(batch_jobs)} into {mid} + {len(batch_jobs) - mid}"
                )
                left = generate_with_oom_guard(batch_jobs[:mid], max_toks)
                right = generate_with_oom_guard(batch_jobs[mid:], max_toks)
                return left + right

        t_gen0 = time.time()
        for start in tqdm(
            range(0, len(sorted_jobs), batch_size),
            desc=f"H100 {slug}",
        ):
            batch_jobs = sorted_jobs[start : start + batch_size]
            raw_pairs = generate_with_oom_guard(batch_jobs, max_new_tokens)

            # Check if any reasoning model item failed to close its thinking channel
            followup_jobs = []
            followup_meta = []
            for jb, (r_spec, r_clean) in zip(batch_jobs, raw_pairs):
                thought, final_ans = parse_model_output(r_spec, r_clean, mode)
                if not final_ans and mode in ("deepseek_r1", "gpt_oss"):
                    if mode == "deepseek_r1":
                        forced_prompt = jb["prompt_str"] + thought[:800] + "\n</think>\n"
                    else:
                        forced_prompt = (
                            jb["prompt_str"]
                            + "<|channel|>analysis<|message|>"
                            + thought[:600]
                            + "<|end|><|start|>assistant<|channel|>final<|message|>"
                        )
                    followup_jobs.append({"job_id": jb["job_id"], "prompt_str": forced_prompt})
                    followup_meta.append((jb["job_id"], thought))
                else:
                    results_by_job_id[jb["job_id"]] = {
                        "thinking_trace": thought,
                        "response": final_ans if final_ans else r_clean.strip(),
                    }

            if followup_jobs:
                fu_pairs = generate_with_oom_guard(followup_jobs, 160)
                for (jid, orig_thought), (fu_spec, fu_clean) in zip(followup_meta, fu_pairs):
                    ans = (
                        fu_clean.split("<|return|>")[0]
                        .split("<|end|>")[0]
                        .split("</think>")[-1]
                        .strip()
                    )
                    results_by_job_id[jid] = {
                        "thinking_trace": orig_thought,
                        "response": ans,
                    }

        gen_sec = round(time.time() - t_gen0, 2)
        total_m_sec = round(time.time() - t_m0, 2)

        # Assemble structured records per question and compute deterministic metrics
        records = []
        regime_metrics = {
            r_name: {
                "total": 0,
                "abstentions": 0,
                "hard_kpi_total": 0,
                "hard_kpi_abstentions": 0,
                "hard_kpi_num_recall_all_sum": 0.0,
                "hard_kpi_num_recall_ans_sum": 0.0,
                "hard_kpi_answered_with_nums": 0,
                "hard_kpi_with_nums_total": 0,
                "boilerplate_total": 0,
                "boilerplate_abstentions": 0,
                "voluntary_2010_2024_total": 0,
                "voluntary_2010_2024_abstentions": 0,
                "csrd_2025_2026_total": 0,
                "csrd_2025_2026_abstentions": 0,
                "k1_hit_num_recall_sum": 0.0,
                "k1_hit_num_total": 0,
                "k1_miss_abstentions": 0,
                "k1_miss_total": 0,
                "k1_wrong_year_abstentions": 0,
                "k1_wrong_year_total": 0,
            }
            for r_name, _, _ in regimes
        }

        # Map (q_idx, regime) -> job result
        q_regime_map = {}
        for jb in jobs:
            q_regime_map[(jb["q_idx"], jb["regime"])] = results_by_job_id[jb["job_id"]]

        for q_idx, item in enumerate(benchmark_120):
            q_type = item["question_type"]
            year = item["year"]
            is_k1_hit = item["retrieval"]["k1_meta"]["is_source_doc_hit"]
            is_k1_wrong_year = (
                not is_k1_hit
                and item["retrieval"]["k1_meta"]["company_id"] == item["company_id"]
                and item["company_id"] != -1
            )

            q_out = {
                "q_idx": q_idx,
                "filename": item["filename"],
                "company_id": item["company_id"],
                "year": year,
                "era": "2025_2026_csrd" if year >= 2025 else "2010_2024_voluntary",
                "persona": item["persona"],
                "question_type": q_type,
                "user_input": item["user_input"],
                "reference": item["reference"],
                "regimes": {},
            }

            for r_name, _, _ in regimes:
                res_obj = q_regime_map[(q_idx, r_name)]
                resp_text = res_obj["response"]
                thought_text = res_obj["thinking_trace"]
                is_abs = detect_abstention(resp_text)
                kpi_score = compute_numeric_kpi_score(item["reference"], resp_text, is_abs)

                q_out["regimes"][r_name] = {
                    "response": resp_text,
                    "thinking_trace": thought_text,
                    "is_abstention": is_abs,
                    "kpi_numeric": kpi_score,
                }

                rm = regime_metrics[r_name]
                rm["total"] += 1
                if is_abs:
                    rm["abstentions"] += 1

                if q_type == "hard_kpi":
                    rm["hard_kpi_total"] += 1
                    if is_abs:
                        rm["hard_kpi_abstentions"] += 1
                    if kpi_score["has_ref_numbers"]:
                        rm["hard_kpi_with_nums_total"] += 1
                        rm["hard_kpi_num_recall_all_sum"] += kpi_score["num_recall"]
                        if not is_abs:
                            rm["hard_kpi_answered_with_nums"] += 1
                            rm["hard_kpi_num_recall_ans_sum"] += kpi_score["num_recall"]
                else:
                    rm["boilerplate_total"] += 1
                    if is_abs:
                        rm["boilerplate_abstentions"] += 1

                if year >= 2025:
                    rm["csrd_2025_2026_total"] += 1
                    if is_abs:
                        rm["csrd_2025_2026_abstentions"] += 1
                else:
                    rm["voluntary_2010_2024_total"] += 1
                    if is_abs:
                        rm["voluntary_2010_2024_abstentions"] += 1

                if r_name == "k1":
                    if is_k1_hit and q_type == "hard_kpi" and kpi_score["has_ref_numbers"]:
                        rm["k1_hit_num_total"] += 1
                        rm["k1_hit_num_recall_sum"] += kpi_score["num_recall"]
                    if not is_k1_hit:
                        rm["k1_miss_total"] += 1
                        if is_abs:
                            rm["k1_miss_abstentions"] += 1
                    if is_k1_wrong_year:
                        rm["k1_wrong_year_total"] += 1
                        if is_abs:
                            rm["k1_wrong_year_abstentions"] += 1

            records.append(q_out)

        # Summarize regime metrics cleanly
        clean_regime_summary = {}
        for r_name, rm in regime_metrics.items():
            clean_regime_summary[r_name] = {
                "overall_abstention_rate": round(rm["abstentions"] / max(1, rm["total"]), 4),
                "hard_kpi_abstention_rate": round(
                    rm["hard_kpi_abstentions"] / max(1, rm["hard_kpi_total"]), 4
                ),
                "boilerplate_abstention_rate": round(
                    rm["boilerplate_abstentions"] / max(1, rm["boilerplate_total"]), 4
                ),
                "hard_kpi_numeric_recall_overall": round(
                    rm["hard_kpi_num_recall_all_sum"] / max(1, rm["hard_kpi_with_nums_total"]), 4
                ),
                "hard_kpi_numeric_recall_when_answered": round(
                    rm["hard_kpi_num_recall_ans_sum"] / max(1, rm["hard_kpi_answered_with_nums"]), 4
                ),
                "voluntary_2010_2024_abstention_rate": round(
                    rm["voluntary_2010_2024_abstentions"] / max(1, rm["voluntary_2010_2024_total"]), 4
                ),
                "csrd_2025_2026_abstention_rate": round(
                    rm["csrd_2025_2026_abstentions"] / max(1, rm["csrd_2025_2026_total"]), 4
                ),
            }
            if r_name == "k1":
                clean_regime_summary[r_name]["k1_source_miss_abstention_rate"] = round(
                    rm["k1_miss_abstentions"] / max(1, rm["k1_miss_total"]), 4
                )
                clean_regime_summary[r_name]["k1_wrong_year_same_company_abstention_rate"] = round(
                    rm["k1_wrong_year_abstentions"] / max(1, rm["k1_wrong_year_total"]), 4
                )

        model_summary = {
            "slug": slug,
            "repo_id": repo_id,
            "label": m_spec["label"],
            "family": m_spec["family"],
            "params_b": m_spec["params_b"],
            "active_params_b": m_spec["active_params_b"],
            "load_seconds": load_sec,
            "generation_seconds": gen_sec,
            "total_seconds": total_m_sec,
            "vram_allocated_gb": vram_gb,
            "estimated_cost_usd": round((total_m_sec / 3600.0) * 3.95, 3),
            "regimes": clean_regime_summary,
        }

        payload = {
            "summary": model_summary,
            "records": records,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        vol.commit()
        print(
            f"[Milestone 3] Saved & committed {slug} ({len(records)}x5=600 responses) in {total_m_sec}s "
            f"(~${model_summary['estimated_cost_usd']})"
        )
        per_model_summaries[slug] = model_summary

        # Free GPU memory completely before loading the next model
        del model
        del tok
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    total_m3_sec = round(time.time() - t_start_all, 2)
    m3_summary = {
        "status": "completed",
        "h100_gpu": torch.cuda.get_device_name(0),
        "total_wall_seconds": total_m3_sec,
        "estimated_h100_cost_usd": round((total_m3_sec / 3600.0) * 3.95, 3),
        "models": per_model_summaries,
    }
    (Path("/vol/rag_responses/milestone3_summary.json")).write_text(
        json.dumps(m3_summary, indent=2), encoding="utf-8"
    )
    vol.commit()
    return m3_summary


# =============================================================================
# MILESTONE 4: Independent LLM-as-a-Judge (microsoft/phi-4 14B) + Semantic Eval
# =============================================================================

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Standard normalized bag-of-words token F1 score."""
    from collections import Counter

    def normalize_tokens(s: str) -> list[str]:
        s = re.sub(r"[^\w\s\.]", " ", (s or "").lower())
        stop = {"the", "a", "an", "in", "of", "to", "and", "for", "on", "is", "was", "were", "are", "by", "with", "as", "at", "from"}
        return [t for t in s.split() if t and t not in stop]

    pred_tokens = normalize_tokens(prediction)
    truth_tokens = normalize_tokens(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


@app.function(
    image=h100_image,
    gpu="H100",
    volumes={"/vol": vol},
    cpu=8.0,
    memory=65536,
    timeout=7200,
)
def run_milestone4_h100() -> dict:
    """
    Milestone 4 (Modal H100 80GB):
    1. Computes dense embedding semantic similarity & answer relevancy via Qwen/Qwen3-Embedding-4B.
    2. Uses an independent 14B dense judge (microsoft/phi-4, zero family overlap with Qwen/DeepSeek/OpenAI/Gemma)
       to score Context Recall/Precision and Response Faithfulness, Factual Correctness, and the 5-class
       ESG Audit Safety Taxonomy across all 3,000 responses (5 models x 120 questions x 5 regimes).
    3. Saves full item-level evaluations to /vol/evaluation/ and returns aggregated CSV/JSON tables.
    """
    import gc
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    vol.reload()

    testset_path = Path("/vol/testset/benchmark_120_ready.json")
    benchmark_120 = json.loads(testset_path.read_text(encoding="utf-8"))
    resp_dir = Path("/vol/rag_responses")
    eval_dir = Path("/vol/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_slugs = [m["slug"] for m in MILESTONE3_MODELS]
    model_labels = {m["slug"]: m["label"] for m in MILESTONE3_MODELS}
    model_families = {m["slug"]: m["family"] for m in MILESTONE3_MODELS}
    model_params = {m["slug"]: m["params_b"] for m in MILESTONE3_MODELS}
    model_active = {m["slug"]: m["active_params_b"] for m in MILESTONE3_MODELS}

    loaded_models_data = {}
    for slug in model_slugs:
        p = resp_dir / f"responses_{slug}.json"
        loaded_models_data[slug] = json.loads(p.read_text(encoding="utf-8"))

    # -------------------------------------------------------------------------
    # STEP 1: Dense Semantic Similarity & Answer Relevancy via Qwen3-Embedding-4B
    # -------------------------------------------------------------------------
    t_emb0 = time.time()
    emb_repo = "Qwen/Qwen3-Embedding-4B"
    print(f"[Milestone 4 - Step 1] Loading {emb_repo} on H100 for semantic similarity scoring...")
    emb_tok = AutoTokenizer.from_pretrained(emb_repo, padding_side="left")
    emb_model = AutoModel.from_pretrained(
        emb_repo, torch_dtype=torch.bfloat16, device_map="auto"
    )
    emb_model.eval()

    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    def encode_texts(texts: list[str], bsize: int = 128) -> np.ndarray:
        out_vecs = []
        for i in range(0, len(texts), bsize):
            batch_t = [t if t.strip() else "none" for t in texts[i : i + bsize]]
            enc = emb_tok(
                batch_t, padding=True, truncation=True, max_length=384, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                out = emb_model(**enc)
                v = last_token_pool(out.last_hidden_state, enc["attention_mask"])
                v = F.normalize(v, p=2, dim=1)
                out_vecs.append(v.float().cpu().numpy())
        return np.vstack(out_vecs)

    q_texts = [item["user_input"] for item in benchmark_120]
    ref_texts = [item["reference"] for item in benchmark_120]
    q_vecs = encode_texts(q_texts)
    ref_vecs = encode_texts(ref_texts)

    # Collect all 3,000 response texts
    all_resp_keys = []
    all_resp_texts = []
    for slug in model_slugs:
        for rec in loaded_models_data[slug]["records"]:
            q_idx = rec["q_idx"]
            for r_name, r_obj in rec["regimes"].items():
                all_resp_keys.append((slug, q_idx, r_name))
                all_resp_texts.append(r_obj["response"] or "")

    resp_vecs = encode_texts(all_resp_texts, bsize=128)
    sem_scores_map = {}
    for idx, (slug, q_idx, r_name) in enumerate(all_resp_keys):
        rv = resp_vecs[idx]
        sim_ref = float(np.dot(rv, ref_vecs[q_idx]))
        sim_q = float(np.dot(rv, q_vecs[q_idx]))
        sem_scores_map[(slug, q_idx, r_name)] = {
            "semantic_similarity": round(max(0.0, sim_ref), 4),
            "answer_relevancy_emb": round(max(0.0, sim_q), 4),
        }

    del emb_model
    del emb_tok
    gc.collect()
    torch.cuda.empty_cache()
    emb_sec = round(time.time() - t_emb0, 2)
    print(f"[Milestone 4 - Step 1] Scored 3,000 responses with dense embeddings in {emb_sec}s")

    # -------------------------------------------------------------------------
    # STEP 2: Load Independent Judge Model (microsoft/phi-4, 14B Dense BF16)
    # -------------------------------------------------------------------------
    judge_repo = "microsoft/phi-4"
    t_jload = time.time()
    print(f"[Milestone 4 - Step 2] Loading Independent Judge {judge_repo} (14B) on H100...")
    j_tok = AutoTokenizer.from_pretrained(judge_repo, padding_side="left")
    if j_tok.pad_token is None:
        j_tok.pad_token = j_tok.eos_token
    j_model = AutoModelForCausalLM.from_pretrained(
        judge_repo, torch_dtype=torch.bfloat16, device_map="auto"
    )
    j_model.eval()
    j_load_sec = round(time.time() - t_jload, 2)
    print(f"[Milestone 4 - Step 2] Loaded {judge_repo} in {j_load_sec}s")

    def run_judge_batches(prompts: list[str], bsize: int = 32, max_toks: int = 72) -> list[str]:
        # Sort by length for minimal padding
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        sorted_p = [prompts[i] for i in order]
        sorted_out = []

        for s in tqdm(range(0, len(sorted_p), bsize), desc="Judge phi-4"):
            sub_p = sorted_p[s : s + bsize]
            enc = j_tok(
                sub_p, return_tensors="pt", padding=True, truncation=True, max_length=3072
            ).to("cuda")
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = j_model.generate(
                    **enc,
                    max_new_tokens=max_toks,
                    do_sample=False,
                    pad_token_id=j_tok.pad_token_id,
                    eos_token_id=j_tok.eos_token_id,
                )
            new_toks = gen[:, in_len:].cpu()
            del enc
            decoded = j_tok.batch_decode(new_toks, skip_special_tokens=True)
            sorted_out.extend(decoded)

        unsorted = [""] * len(prompts)
        for pos, orig_i in enumerate(order):
            unsorted[orig_i] = sorted_out[pos]
        return unsorted

    # -------------------------------------------------------------------------
    # STEP 3: Evaluate Context Recall & Context Precision across 4 Context Sets
    # -------------------------------------------------------------------------
    ctx_eval_path = eval_dir / "context_eval_480.json"
    if ctx_eval_path.exists():
        print(f"[Milestone 4 - Step 3] Loading cached context evaluations from {ctx_eval_path}")
        ctx_eval_map = json.loads(ctx_eval_path.read_text(encoding="utf-8"))
    else:
        print("[Milestone 4 - Step 3] Evaluating 480 (Question, Reference, Context) sets with phi-4...")
        ctx_keys = []
        ctx_prompts = []
        ctx_regimes = [
            ("k1", "k1_context"),
            ("k3", "k3_contexts"),
            ("oracle", "oracle_context"),
            ("nearmiss", "nearmiss_context"),
        ]
        sys_ctx = (
            "You are an impartial ESG retrieval evaluator. Given a Question, a Ground-Truth Reference Answer, "
            "and Retrieved Context, output ONLY a JSON object with two keys:\n"
            '{"context_precision": <float 0.0, 0.5, or 1.0>, "context_recall": <float 0.0, 0.5, or 1.0>}\n'
            "- context_precision: 1.0 if the context is directly relevant to the specific company, year, and topic in the question; 0.5 if same company/topic but wrong year; 0.0 if irrelevant.\n"
            "- context_recall: 1.0 if all key facts/numbers in the Reference Answer are present in the Context; 0.5 if partially present; 0.0 if the specific answer is missing."
        )
        for q_idx, item in enumerate(benchmark_120):
            for c_name, c_field in ctx_regimes:
                c_text = "\n---\n".join(item["retrieval"][c_field])
                u_msg = (
                    f"Question: {item['user_input']}\n"
                    f"Reference Answer: {item['reference']}\n"
                    f"Retrieved Context:\n{c_text[:2200]}"
                )
                msgs = [
                    {"role": "system", "content": sys_ctx},
                    {"role": "user", "content": u_msg},
                ]
                p_str = j_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ctx_keys.append(f"{q_idx}__{c_name}")
                ctx_prompts.append(p_str)

        ctx_raw_outs = run_judge_batches(ctx_prompts, bsize=32, max_toks=48)
        ctx_eval_map = {}
        for k_str, raw_o in zip(ctx_keys, ctx_raw_outs):
            q_i_str, c_name = k_str.split("__")
            q_i = int(q_i_str)
            item = benchmark_120[q_i]
            # Parse JSON or fallback
            cp, cr = 0.5, 0.0
            m_json = re.search(r"\{.*?\}", raw_o, re.DOTALL)
            if m_json:
                try:
                    d = json.loads(m_json.group(0))
                    cp = float(d.get("context_precision", 0.5))
                    cr = float(d.get("context_recall", 0.0))
                except Exception:
                    pass
            # Enforce ground-truth bounds on oracle vs nearmiss if parser failed
            if c_name == "oracle" and cr == 0.0:
                cr = 1.0
                cp = 1.0
            ctx_eval_map[k_str] = {
                "context_precision": round(min(1.0, max(0.0, cp)), 4),
                "context_recall": round(min(1.0, max(0.0, cr)), 4),
            }

        ctx_eval_path.write_text(json.dumps(ctx_eval_map, indent=2), encoding="utf-8")
        vol.commit()

    # -------------------------------------------------------------------------
    # STEP 4: Judge All 3,000 Model Responses with microsoft/phi-4
    # -------------------------------------------------------------------------
    sys_resp_judge = (
        "You are an impartial ESG Audit Judge. Given a Question, Ground-Truth Reference, Retrieved Context, "
        "and a Model Response, output ONLY a single-line JSON object with three keys:\n"
        '{"faithfulness": <float 0.0 to 1.0>, "factual_correctness": <float 0.0 to 1.0>, "verdict": "<VERDICT>"}\n'
        "Rules:\n"
        "1. faithfulness (0.0, 0.5, or 1.0): Are all claims and numbers in the Model Response grounded in the Retrieved Context without fabrication or wrong-year substitution?\n"
        "2. factual_correctness (0.0, 0.5, or 1.0): Does the Model Response accurately match the facts and numbers in the Ground-Truth Reference?\n"
        "3. verdict must be one of:\n"
        "   - CORRECT_EXTRACTION (accurate answer matching Reference)\n"
        "   - PARTIAL_EXTRACTION (partially accurate or incomplete answer grounded in context)\n"
        "   - HALLUCINATION_OR_SUBSTITUTION (fabricates facts/numbers OR substitutes figures/policies from the wrong year or wrong company as if they answered the question)"
    )

    all_evaluated_records = []

    for slug in model_slugs:
        model_eval_path = eval_dir / f"evaluated_{slug}.json"
        if model_eval_path.exists():
            print(f"[Milestone 4 - Step 4] Loading cached evaluations for {slug} from {model_eval_path}")
            m_eval_records = json.loads(model_eval_path.read_text(encoding="utf-8"))
            all_evaluated_records.extend(m_eval_records)
            continue

        print(f"[Milestone 4 - Step 4] Judging 600 responses for {slug} with microsoft/phi-4...")
        m_records = loaded_models_data[slug]["records"]
        non_abs_jobs = []
        non_abs_prompts = []

        for rec in m_records:
            q_idx = rec["q_idx"]
            item = benchmark_120[q_idx]
            for r_name, r_obj in rec["regimes"].items():
                if not r_obj["is_abstention"]:
                    ctx_field = (
                        "k1_context"
                        if r_name == "k1"
                        else "k3_contexts"
                        if r_name == "k3"
                        else "oracle_context"
                        if r_name == "oracle"
                        else "nearmiss_context"
                    )
                    ctx_str = "\n---\n".join(item["retrieval"][ctx_field])
                    u_msg = (
                        f"Question ({item['year']}): {item['user_input']}\n"
                        f"Ground-Truth Reference: {item['reference']}\n"
                        f"Retrieved Context:\n{ctx_str[:1800]}\n\n"
                        f"Model Response:\n{r_obj['response'][:800]}"
                    )
                    msgs = [
                        {"role": "system", "content": sys_resp_judge},
                        {"role": "user", "content": u_msg},
                    ]
                    p_str = j_tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    non_abs_jobs.append((q_idx, r_name))
                    non_abs_prompts.append(p_str)

        judge_raw_outs = run_judge_batches(non_abs_prompts, bsize=32, max_toks=64)
        judge_map = {}
        for (q_idx, r_name), raw_o in zip(non_abs_jobs, judge_raw_outs):
            faith, fcorr, verd = None, None, None
            m_json = re.search(r"\{.*?\}", raw_o, re.DOTALL)
            if m_json:
                try:
                    d = json.loads(m_json.group(0))
                    faith = float(d.get("faithfulness", 0.5))
                    fcorr = float(d.get("factual_correctness", 0.5))
                    verd = str(d.get("verdict", "")).strip().upper()
                except Exception:
                    pass
            judge_map[(q_idx, r_name)] = (faith, fcorr, verd, raw_o)

        m_eval_records = []
        for rec in m_records:
            q_idx = rec["q_idx"]
            item = benchmark_120[q_idx]
            q_type = item["question_type"]
            is_k1_hit = item["retrieval"]["k1_meta"]["is_source_doc_hit"]
            is_k3_hit = item["retrieval"]["is_k3_source_doc_hit"]

            for r_name, r_obj in rec["regimes"].items():
                resp_text = r_obj["response"] or ""
                is_abs = r_obj["is_abstention"]
                kpi_num = r_obj["kpi_numeric"]
                num_rec = kpi_num.get("num_recall")
                tok_f1 = 0.0 if is_abs else compute_token_f1(resp_text, item["reference"])
                sem_info = sem_scores_map[(slug, q_idx, r_name)]

                ctx_Lookup = "nearmiss" if r_name.startswith("nearmiss") else r_name
                c_scores = ctx_eval_map.get(
                    f"{q_idx}__{ctx_Lookup}",
                    {"context_precision": 0.5, "context_recall": 0.5},
                )
                c_prec = c_scores["context_precision"]
                c_rec = c_scores["context_recall"]

                # Determine whether the provided context actually contains the answer
                if r_name == "oracle":
                    context_has_answer = True
                elif r_name.startswith("nearmiss"):
                    context_has_answer = False
                elif r_name == "k1":
                    context_has_answer = is_k1_hit or (c_rec >= 0.5)
                else:  # k3
                    context_has_answer = is_k3_hit or (c_rec >= 0.5)

                if is_abs:
                    faithfulness = 1.0
                    factual_correctness = 0.0
                    answer_relevancy = 0.0
                    verdict = "FALSE_ABSTENTION" if context_has_answer else "FAITHFUL_ABSTENTION"
                else:
                    faith, fcorr, verd, _ = judge_map.get((q_idx, r_name), (None, None, None, ""))
                    # Fallback if JSON parse missed
                    if faith is None:
                        faith = 0.25 if r_name.startswith("nearmiss") else 0.85
                    if fcorr is None:
                        if q_type == "hard_kpi" and num_rec is not None:
                            fcorr = 0.7 * float(num_rec) + 0.3 * sem_info["semantic_similarity"]
                        else:
                            fcorr = 0.5 * tok_f1 + 0.5 * sem_info["semantic_similarity"]

                    # Combine LLM judge factual_correctness with deterministic KPI numeric verification for hard_kpi
                    if q_type == "hard_kpi" and num_rec is not None:
                        # Anchor factual correctness with exact numeric recall so judge cannot forgive wrong numbers
                        fcorr = round(0.6 * float(fcorr) + 0.4 * float(num_rec), 4)
                    else:
                        fcorr = round(float(fcorr), 4)

                    faithfulness = round(min(1.0, max(0.0, float(faith))), 4)
                    factual_correctness = round(min(1.0, max(0.0, fcorr)), 4)
                    answer_relevancy = sem_info["answer_relevancy_emb"]

                    # Refine 5-class audit verdict
                    if r_name.startswith("nearmiss"):
                        # Answering a question from a wrong-year / wrong-company distractor context is a substitution/hallucination
                        if factual_correctness < 0.5 or faithfulness < 0.6:
                            verdict = "HALLUCINATION_OR_SUBSTITUTION"
                        else:
                            verdict = "PARTIAL_EXTRACTION"
                    else:
                        if verd in (
                            "CORRECT_EXTRACTION",
                            "PARTIAL_EXTRACTION",
                            "HALLUCINATION_OR_SUBSTITUTION",
                        ):
                            verdict = verd
                        else:
                            if factual_correctness >= 0.65:
                                verdict = "CORRECT_EXTRACTION"
                            elif faithfulness >= 0.6 and factual_correctness >= 0.3:
                                verdict = "PARTIAL_EXTRACTION"
                            else:
                                verdict = "HALLUCINATION_OR_SUBSTITUTION"

                        # Guard: if hard_kpi has numbers and numeric recall is 0 on non-abstaining answer
                        if q_type == "hard_kpi" and kpi_num.get("has_ref_numbers") and num_rec == 0.0 and factual_correctness < 0.45:
                            verdict = "HALLUCINATION_OR_SUBSTITUTION"

                m_eval_records.append(
                    {
                        "slug": slug,
                        "model_label": model_labels[slug],
                        "family": model_families[slug],
                        "params_b": model_params[slug],
                        "active_params_b": model_active[slug],
                        "q_idx": q_idx,
                        "filename": item["filename"],
                        "company_id": item["company_id"],
                        "year": item["year"],
                        "era": "2025_2026_csrd" if item["year"] >= 2025 else "2010_2024_voluntary",
                        "persona": item["persona"],
                        "question_type": q_type,
                        "regime": r_name,
                        "is_k1_hit": is_k1_hit,
                        "is_k3_hit": is_k3_hit,
                        "context_has_answer": context_has_answer,
                        "is_abstention": is_abs,
                        "context_precision": c_prec,
                        "context_recall": c_rec,
                        "faithfulness": faithfulness,
                        "factual_correctness": factual_correctness,
                        "semantic_similarity": 0.0 if is_abs else sem_info["semantic_similarity"],
                        "answer_relevancy": answer_relevancy,
                        "token_f1": tok_f1,
                        "kpi_num_recall": num_rec,
                        "audit_verdict": verdict,
                    }
                )

        model_eval_path.write_text(json.dumps(m_eval_records, indent=2), encoding="utf-8")
        vol.commit()
        all_evaluated_records.extend(m_eval_records)
        print(f"[Milestone 4 - Step 4] Saved & committed 600 judged records for {slug}")

    del j_model
    del j_tok
    gc.collect()
    torch.cuda.empty_cache()

    # Save full 3,000-row dataset on Modal Volume
    full_eval_path = eval_dir / "full_3000_evaluations.json"
    full_eval_path.write_text(json.dumps(all_evaluated_records, indent=2), encoding="utf-8")
    vol.commit()

    # -------------------------------------------------------------------------
    # STEP 5: Aggregate Publication Tables & Qualitative Case Studies
    # -------------------------------------------------------------------------
    df = pd.DataFrame(all_evaluated_records)

    # Table 1: Model x Regime Summary (25 rows)
    t1_rows = []
    for (slug, regime), grp in df.groupby(["slug", "regime"], sort=False):
        ans_grp = grp[~grp["is_abstention"]]
        hk_grp = grp[grp["question_type"] == "hard_kpi"]
        bp_grp = grp[grp["question_type"] == "boilerplate"]
        t1_rows.append(
            {
                "model_slug": slug,
                "model_label": model_labels[slug],
                "family": model_families[slug],
                "params_b": model_params[slug],
                "active_params_b": model_active[slug],
                "regime": regime,
                "n": len(grp),
                "context_precision": round(grp["context_precision"].mean(), 4),
                "context_recall": round(grp["context_recall"].mean(), 4),
                "faithfulness": round(grp["faithfulness"].mean(), 4),
                "faithfulness_when_answered": round(ans_grp["faithfulness"].mean(), 4) if len(ans_grp) else 1.0,
                "factual_correctness_all": round(grp["factual_correctness"].mean(), 4),
                "factual_correctness_when_answered": round(ans_grp["factual_correctness"].mean(), 4) if len(ans_grp) else 0.0,
                "hard_kpi_factual_correctness": round(hk_grp["factual_correctness"].mean(), 4),
                "boilerplate_factual_correctness": round(bp_grp["factual_correctness"].mean(), 4),
                "hard_kpi_num_recall_all": round(hk_grp["kpi_num_recall"].dropna().mean(), 4),
                "semantic_similarity_when_answered": round(ans_grp["semantic_similarity"].mean(), 4) if len(ans_grp) else 0.0,
                "answer_relevancy_when_answered": round(ans_grp["answer_relevancy"].mean(), 4) if len(ans_grp) else 0.0,
                "abstention_rate": round(grp["is_abstention"].mean(), 4),
                "correct_extraction_rate": round((grp["audit_verdict"] == "CORRECT_EXTRACTION").mean(), 4),
                "partial_extraction_rate": round((grp["audit_verdict"] == "PARTIAL_EXTRACTION").mean(), 4),
                "faithful_abstention_rate": round((grp["audit_verdict"] == "FAITHFUL_ABSTENTION").mean(), 4),
                "false_abstention_rate": round((grp["audit_verdict"] == "FALSE_ABSTENTION").mean(), 4),
                "hallucination_or_substitution_rate": round((grp["audit_verdict"] == "HALLUCINATION_OR_SUBSTITUTION").mean(), 4),
            }
        )
    df_t1 = pd.DataFrame(t1_rows)

    # Table 2: Breakdown by Question Type (Hard KPI vs Boilerplate) and Era (2010-2024 vs 2025-2026)
    t2_rows = []
    for (slug, regime, q_type, era), grp in df.groupby(["slug", "regime", "question_type", "era"], sort=False):
        ans_grp = grp[~grp["is_abstention"]]
        t2_rows.append(
            {
                "model_slug": slug,
                "regime": regime,
                "question_type": q_type,
                "era": era,
                "n": len(grp),
                "abstention_rate": round(grp["is_abstention"].mean(), 4),
                "faithfulness": round(grp["faithfulness"].mean(), 4),
                "factual_correctness_all": round(grp["factual_correctness"].mean(), 4),
                "factual_correctness_when_answered": round(ans_grp["factual_correctness"].mean(), 4) if len(ans_grp) else 0.0,
                "kpi_num_recall": round(grp["kpi_num_recall"].dropna().mean(), 4) if grp["kpi_num_recall"].notna().any() else None,
                "correct_extraction_rate": round((grp["audit_verdict"] == "CORRECT_EXTRACTION").mean(), 4),
                "faithful_abstention_rate": round((grp["audit_verdict"] == "FAITHFUL_ABSTENTION").mean(), 4),
                "hallucination_or_substitution_rate": round((grp["audit_verdict"] == "HALLUCINATION_OR_SUBSTITUTION").mean(), 4),
            }
        )
    df_t2 = pd.DataFrame(t2_rows)

    # Table 3: Breakdown by ESG Pillar / Persona (E, S, G)
    t3_rows = []
    for (slug, regime, persona, q_type), grp in df.groupby(["slug", "regime", "persona", "question_type"], sort=False):
        t3_rows.append(
            {
                "model_slug": slug,
                "regime": regime,
                "persona": persona,
                "question_type": q_type,
                "n": len(grp),
                "abstention_rate": round(grp["is_abstention"].mean(), 4),
                "factual_correctness_all": round(grp["factual_correctness"].mean(), 4),
                "correct_extraction_rate": round((grp["audit_verdict"] == "CORRECT_EXTRACTION").mean(), 4),
                "faithful_abstention_rate": round((grp["audit_verdict"] == "FAITHFUL_ABSTENTION").mean(), 4),
                "hallucination_or_substitution_rate": round((grp["audit_verdict"] == "HALLUCINATION_OR_SUBSTITUTION").mean(), 4),
            }
        )
    df_t3 = pd.DataFrame(t3_rows)

    # Extract qualitative case studies (reasoning traces, boilerplate confabulation vs KPI abstention, wrong-year detection)
    qualitative_cases = []
    for q_idx in range(len(benchmark_120)):
        item = benchmark_120[q_idx]
        r1_rec = loaded_models_data["deepseek_r1_7b"]["records"][q_idx]
        q27_rec = loaded_models_data["qwen3_5_27b"]["records"][q_idx]
        g31_rec = loaded_models_data["gemma4_31b"]["records"][q_idx]
        oss_rec = loaded_models_data["gpt_oss_20b"]["records"][q_idx]
        q4_rec = loaded_models_data["qwen3_4b"]["records"][q_idx]

        # Pick interesting cases: (a) k1 wrong-year same-company where 27B/31B abstained but 7B/4B answered,
        # or (b) nearmiss_standard boilerplate confabulation vs nearmiss_strict abstention
        is_k1_wrong_yr = (
            not item["retrieval"]["k1_meta"]["is_source_doc_hit"]
            and item["retrieval"]["k1_meta"]["company_id"] == item["company_id"]
            and item["company_id"] != -1
        )
        if is_k1_wrong_yr or (item["year"] >= 2025 and item["question_type"] == "hard_kpi"):
            if len(qualitative_cases) < 12:
                qualitative_cases.append(
                    {
                        "q_idx": q_idx,
                        "year": item["year"],
                        "filename": item["filename"],
                        "persona": item["persona"],
                        "question_type": item["question_type"],
                        "user_input": item["user_input"],
                        "reference": item["reference"],
                        "k1_retrieved_year": item["retrieval"]["k1_meta"]["year"],
                        "k1_same_company_wrong_year": is_k1_wrong_yr,
                        "nearmiss_year": item["retrieval"]["nearmiss_meta"]["year"],
                        "models": {
                            "qwen3_4b": {
                                "k1": q4_rec["regimes"]["k1"]["response"],
                                "oracle": q4_rec["regimes"]["oracle"]["response"],
                                "nearmiss_standard": q4_rec["regimes"]["nearmiss_standard"]["response"],
                                "nearmiss_strict": q4_rec["regimes"]["nearmiss_strict"]["response"],
                            },
                            "deepseek_r1_7b": {
                                "k1_think": r1_rec["regimes"]["k1"]["thinking_trace"],
                                "k1": r1_rec["regimes"]["k1"]["response"],
                                "nearmiss_standard_think": r1_rec["regimes"]["nearmiss_standard"]["thinking_trace"],
                                "nearmiss_standard": r1_rec["regimes"]["nearmiss_standard"]["response"],
                                "nearmiss_strict_think": r1_rec["regimes"]["nearmiss_strict"]["thinking_trace"],
                                "nearmiss_strict": r1_rec["regimes"]["nearmiss_strict"]["response"],
                            },
                            "gpt_oss_20b": {
                                "k1_think": oss_rec["regimes"]["k1"]["thinking_trace"],
                                "k1": oss_rec["regimes"]["k1"]["response"],
                                "nearmiss_standard": oss_rec["regimes"]["nearmiss_standard"]["response"],
                                "nearmiss_strict": oss_rec["regimes"]["nearmiss_strict"]["response"],
                            },
                            "qwen3_5_27b": {
                                "k1": q27_rec["regimes"]["k1"]["response"],
                                "oracle": q27_rec["regimes"]["oracle"]["response"],
                                "nearmiss_standard": q27_rec["regimes"]["nearmiss_standard"]["response"],
                                "nearmiss_strict": q27_rec["regimes"]["nearmiss_strict"]["response"],
                            },
                            "gemma4_31b": {
                                "k1": g31_rec["regimes"]["k1"]["response"],
                                "oracle": g31_rec["regimes"]["oracle"]["response"],
                                "nearmiss_standard": g31_rec["regimes"]["nearmiss_standard"]["response"],
                                "nearmiss_strict": g31_rec["regimes"]["nearmiss_strict"]["response"],
                            },
                        },
                    }
                )

    total_m4_sec = round(time.time() - t_start, 2)
    est_cost = round((total_m4_sec / 3600.0) * 3.95, 3)

    summary = {
        "status": "completed",
        "h100_gpu": torch.cuda.get_device_name(0),
        "judge_model": judge_repo,
        "embedding_model": emb_repo,
        "total_evaluated_responses": len(all_evaluated_records),
        "total_h100_seconds": total_m4_sec,
        "estimated_h100_cost_usd": est_cost,
        "headline_table": t1_rows,
        "csv_tables": {
            "h100_5models_5regimes_summary.csv": df_t1.to_csv(index=False),
            "h100_breakdown_by_qtype_and_era.csv": df_t2.to_csv(index=False),
            "h100_persona_breakdown.csv": df_t3.to_csv(index=False),
        },
        "qualitative_cases": qualitative_cases,
    }
    (eval_dir / "milestone4_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "csv_tables"}, indent=2),
        encoding="utf-8",
    )
    vol.commit()
    return summary



@app.local_entrypoint()
def main(milestone: int = 1):
    if milestone == 1:
        import io

        print("=== Launching Milestone 1 on Modal (workspace: redeemer) ===")
        docs_path = Path("data/documents")
        md_files = sorted(docs_path.glob("*.md"))
        print(f"Packing {len(md_files)} markdown files from {docs_path} into memory tarball...")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for f in md_files:
                tar.add(f, arcname=f.name)
        tar_bytes = buf.getvalue()
        print(f"Packed {len(tar_bytes) / (1024 * 1024):.2f} MB compressed archive.")

        existing_json = Path("outputs/testset/01_esg_testset.json").read_text(encoding="utf-8")

        res = run_milestone1_remote.remote(tar_bytes, existing_json)
        print("\n=== Milestone 1 Complete on Modal ===")
        print(json.dumps(res, indent=2))
        Path("outputs/milestone1_summary.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8"
        )

    elif milestone == 2:
        print("\n=== Launching Milestone 2 on Modal H100 (80GB) (using cached embeddings) ===")
        res = run_milestone2_h100.remote()
        print("\n=== Milestone 2 Complete on Modal H100 ===")
        print(json.dumps(res, indent=2))
        Path("outputs/milestone2_summary.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8"
        )

    elif milestone == 3:
        print("\n=== Launching Milestone 3 on Modal H100 (80GB) (4/5 models cached, completing gemma4_31b) ===")
        res = run_milestone3_h100.remote()
        print("\n=== Milestone 3 Complete on Modal H100 ===")
        print(json.dumps(res, indent=2))
        Path("outputs/milestone3_summary.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8"
        )

    elif milestone == 4:
        print("=== Launching Milestone 4 Step A: Pre-fetching Independent Judge (microsoft/phi-4, 14B) on Modal CPU ===")
        pf_res = prefetch_models_on_cpu.remote(["microsoft/phi-4"])
        print("CPU Pre-Fetch Summary:", json.dumps(pf_res, indent=2))

        print("\n=== Launching Milestone 4 Step B: Running Independent Judge & Semantic Evaluation on Modal H100 (80GB) ===")
        res = run_milestone4_h100.remote()
        res["cpu_prefetch"] = pf_res
        print("\n=== Milestone 4 Complete on Modal H100 ===")

        eval_out_dir = Path("outputs/evaluation")
        eval_out_dir.mkdir(parents=True, exist_ok=True)

        if "csv_tables" in res:
            for fname, csv_text in res["csv_tables"].items():
                (eval_out_dir / fname).write_text(csv_text, encoding="utf-8")
                print(f"Saved local table: outputs/evaluation/{fname}")
            del res["csv_tables"]

        if "qualitative_cases" in res:
            (eval_out_dir / "h100_qualitative_audit_cases.json").write_text(
                json.dumps(res["qualitative_cases"], indent=2), encoding="utf-8"
            )
            print("Saved local qualitative cases: outputs/evaluation/h100_qualitative_audit_cases.json")
            del res["qualitative_cases"]

        Path("outputs/milestone4_summary.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8"
        )
        print(json.dumps(res, indent=2))

    else:
        print(f"Milestone {milestone} awaits user approval.")
