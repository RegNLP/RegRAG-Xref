
# RegRAG-Xref

RegRAG-Xref is a research codebase for building and analyzing **dual-passage regulatory QA datasets** and **cross-reference–aware RAG** over a curated corpus of **40 ADGM/FSRA documents**.

The project has three main pillars:

1.  **Generation** – LLM-generated QAs from manually extracted cross-references (two methods: DPEL & SCHEMA).
    
2.  **Curation** – LLM-as-a-judge + multi-system IR concordance (and later RAG concordance) to approximate human evaluation.
    
3.  **Evaluation** – intrinsic analysis of the dataset + extrinsic evaluation via RAG.
    

_Last updated: 4 Nov 2025 (Asia/Dubai)_

## 1. Corpus & Cross-Reference Ground Truth

-   **Corpus:** 40 ADGM/FSRA regulatory documents (rules, regulations, guidance).
    
-   **Pre-processing:** documents are manually segmented into passages (section/rule level).
    
-   **Cross-references:** all intra-/inter-document references are manually extracted into:
    
    -   `data/CrossReferenceData.csv`
        

**Illustrative fields:**

-   `SourceDocumentName`, `SourcePassageID`, `SourceText`
    
-   `TargetDocumentName`, `TargetPassageID`, `TargetText`
    
-   `ReferenceType` ∈ {Internal, External, NotDefined}
    
-   `ReferenceText` (e.g., “Rule 3.6A.4”, “95(2)”)
    

This CSV is the canonical ground truth for **paired evidence** (SOURCE ↔ TARGET).

## 2. QA Generation (Two Methods, Two Personas)

For each cross-reference pair, we generate QAs that **require both passages** (SOURCE & TARGET) to answer correctly.

### 2.1 Personas

Two personas are used for questions; answers are always professional.

**PROFESSIONAL_STYLE**

> Regulator / counsel tone, precise legal terms, formal, may be multi-clause.

**BASIC_STYLE**

> Plain language for a smart non-expert, short sentences, keeps actor names as written.

**System discipline (shared):**

-   Use **only** SOURCE/TARGET text (no external knowledge).
    
-   Every substantive claim must be grounded in at least one of the two passages.
    
-   Answers must include both tags: `[#SRC:<source_passage_id>]` and `[#TGT:<target_passage_id>]`.
    
-   Output must be **valid JSON** (no markdown).
    

### 2.2 Method A — DPEL (Direct Passage Evidence Linking)

-   **Input:** `data/CrossReferenceData.csv`
    
-   **Goal:** generate QAs directly from cross-reference pairs.
    
-   **Filters:**
    
    -   Skip empty/degenerate pairs (`source_id == target_id`, identical texts, empty passages).
        
    -   Optional sampling (`--row_sample_n`, `--max_pairs`).
        
    -   Optional `--drop_title_targets`.
        
-   **Answer requirements:**
    
    -   Professional paragraph (~180–230 words, min 160).
        
    -   Must contain `[#SRC:…]` and `[#TGT:…]`.
        
-   **Citation policy:** questions may include citations.
    
-   **Command (example):**
    
    ```
    python srs/generate_qas_method_DPEL.py \
      --input_csv data/CrossReferenceData.csv \
      --output_jsonl outputs/generation/dpel/all/answers.jsonl \
      --report_json outputs/generation/dpel/all/report.json \
      --model gpt-4o \
      --max_q_per_pair 2 \
      --sample_n 3 \
      --temperature 0.2 \
      --seed 13 \
      --dedup \
      --verbose
    
    ```
    

### 2.3 Method B — SCHEMA (Schema-Anchored Generation)

Two steps: schema extraction, then QA generation from schema.

**Step 1: Schema Extraction**

```
python srs/extract_schemas.py \
  --input_csv data/CrossReferenceData.csv \
  --output_jsonl outputs/extracted_schema.jsonl \
  --model gpt-4o \
  --drop_title_targets

```

Each item includes:

-   `semantic_hook`, `citation_hook`
    
-   `source_passage_id`, `source_text`, `target_passage_id`, `target_text`
    
-   `source_item_type`, `target_item_type` ∈ {Obligation, Prohibition, Permission, Definition, Scope, Procedure, Other}
    
-   `answer_spans` (typed spans: FREEFORM, DATE, MONEY, TERM, SECTION, …)
    
-   `target_is_title` (bool)
    

Heuristics:

-   Title detection & optional skipping (`--drop_title_targets`).
    
-   Span validation; fallback to modal clause or leading FREEFORM snippet.
    
-   Light deduplication.
    

**Step 2: QA Generation From Schema**

```
python srs/generate_qas_method_schema.py \
  --input_jsonl outputs/extracted_schema.jsonl \
  --output_jsonl outputs/generation/schema/all/answers.jsonl \
  --report_json outputs/generation/schema/all/report.json \
  --model gpt-4o \
  --max_q_per_pair 2 \
  --sample_n 3 \
  --temperature 0.2 \
  --seed 13 \
  --dual_anchors_mode freeform_only \
  --dedup \
  --verbose

```

Key points:

-   Same answer constraints as DPEL (length, dual tags).
    
-   Uses `semantic_hook`, item types, and `answer_spans` to guide question/answer content.
    
-   Optional `--no_citations` to forbid citations in questions for SCHEMA.
    

## 3. Curation via LLM-as-a-Judge (Ensemble)

We filter generated QAs using an LLM rubric tuned to cross-reference reasoning.

### 3.1 Hard Gates

Local script gates:

-   Answer must include both tags `[#SRC:id]` and `[#TGT:id]`.
    
-   Optional citation gates (e.g., forbid citations in SCHEMA questions).
    

LLM gate:

-   Dual-evidence: the question must truly require both passages.
    

### 3.2 Scoring Rubric

Each judge assigns:

-   `realism` (0–2): realistic compliance question.
    
-   `dual_use` (0–4): answer breaks if one passage is removed.
    
-   `correctness` (0–4): factual and grounded.
    
-   `final_score` = `realism` + `dual_use` + `correctness` (0–10).
    

Pass conditions:

-   All hard gates pass.
    
-   `final_score` ≥ 7.
    
-   `dual_use` ≥ 3.
    

### 3.3 Ensemble & Fusion

-   Multiple lightweight models (e.g. `gpt-4.1-mini`, `gpt-4o-mini` + repeated seed).
    
-   Fused scores via median; pass/fail via majority + `dual_use` tie-break.
    

**Command:**

```
python srs/judge_qas_ensemble.py \
  --inputs outputs/generation/dpel/all/answers.jsonl \
           outputs/generation/schema/all/answers_nociteQ.jsonl \
  --out_jsonl outputs/judging/ensemble/judgments.jsonl \
  --report_json outputs/judging/ensemble/summary.json \
  --ensemble_models gpt-4.1-mini,gpt-4o-mini \
  --repeat_first_with_seed 17 \
  --pass_threshold 7 \
  --require_dual_use_k 2 \
  --forbid_citations_in_question_for_schema \
  --allow_citations_in_answer \
  --temperature 0.0 \
  --seed 13 \
  --verbose

```

Curated sets are then split into:

-   `outputs/judging/curated/DPEL/{kept,eliminated}.jsonl`
    
-   `outputs/judging/curated/SCHEMA/{kept,eliminated}.jsonl`
    

## 4. Full-Corpus IR & Concordance

We evaluate retrieval over the full 40-document corpus and use multi-system IR concordance as a proxy for human agreement.

### 4.1 Full Passage Corpus

Source documents and paths are configured in:

-   `srs/doc_manifest.py`
    

Full corpus builder:

```
python srs/build_full_passages.py \
  --out_passages data/passages_full.jsonl \
  --out_json_collection passages_json/collection_full.jsonl

```

Each line:

```
{"pid": "...", "text": "...", "document_id": 3, "passage_id": "1.1"}

```

### 4.2 Queries & Qrels from Curated QAs

From curated QAs, we build query sets and qrels:

```
python srs/build_ir_inputs.py \
  --inputs \
    outputs/judging/curated/DPEL/kept.jsonl \
    outputs/judging/curated/DPEL/eliminated.jsonl \
    outputs/judging/curated/SCHEMA/kept.jsonl \
    outputs/judging/curated/SCHEMA/eliminated.jsonl

```

Produces:

-   `inputs/ir/queries_kept.tsv`, `inputs/ir/qrels_kept.txt`
    
-   `inputs/ir/queries_eliminated.tsv`, `inputs/ir/qrels_eliminated.txt`
    

For method-specific slicing:

```
python srs/build_method_qrels.py \
  --dpel-kept outputs/judging/curated/DPEL/kept.jsonl \
  --dpel-elim outputs/judging/curated/DPEL/eliminated.jsonl \
  --schema-kept outputs/judging/curated/SCHEMA/kept.jsonl \
  --schema-elim outputs/judging/curated/SCHEMA/eliminated.jsonl

```

Produces:

-   `inputs/ir/qrels_kept_dpel.txt`, `inputs/ir/qrels_eliminated_dpel.txt`
    
-   `inputs/ir/qrels_kept_schema.txt`, `inputs/ir/qrels_eliminated_schema.txt`
    

Each query has exactly two relevant passages: the SOURCE and TARGET.

### 4.3 IR Methods (5-System Suite)

All evaluated on `data/passages_full.jsonl`:

1.  **BM25** (Pyserini / Lucene)
    
2.  **Dense e5** (`intfloat/e5-base-v2`)
    
3.  **Dense BGE** (`BAAI/bge-base-en-v1.5`)
    
4.  **BM25 → e5 rerank** (two-stage retriever)
    
5.  **Hybrid RRF** (BM25 + e5) (BM25 + e5 fusion)
    

Example BM25 index & run:

```
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input passages_json \
  --index indexes/bm25_full \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.search.lucene \
  --index indexes/bm25_full \
  --topics inputs/ir/queries_kept.tsv \
  --bm25 --k1 0.9 --b 0.4 \
  --hits 100 \
  --batch-size 64 --threads 4 \
  --output runs_full/kept/bm25.txt

```

Dense e5 example:

```
python srs/run_dense_e5_sbert.py \
  --passages data/passages_full.jsonl \
  --model-name intfloat/e5-base-v2 \
  --queries inputs/ir/queries_kept.tsv \
  --output runs_full/kept/e5.txt \
  --k 100

```

Hybrid RRF example:

```
python srs/fuse_rrf.py \
  --bm25 runs_full/kept/bm25.txt \
  --dense runs_full/kept/e5.txt \
  --output runs_full/kept/hybrid_rrf_bm25_e5.txt \
  --k 100 --rrf-k 60

```

### 4.4 IR Evaluation & Concordance

We compute metrics:

-   Recall@10 (R@10)
    
-   MAP@10
    
-   nDCG@10
    

...using `srs/eval_ir.py` for each slice:

-   DPEL-kept / DPEL-eliminated
    
-   SCHEMA-kept / SCHEMA-eliminated
    

Example:

```
python srs/eval_ir.py \
  --qrels inputs/ir/qrels_kept_dpel.txt \
  --runs \
    runs_full/kept/bm25.txt \
    runs_full/kept/bge.txt \
    runs_full/kept/e5.txt \
    runs_full/kept/bm25_e5_rerank.txt \
    runs_full/kept/hybrid_rrf_bm25_e5.txt \
  --k 10

```

To approximate human agreement, we compute per-query concordance across the 5 IR methods:

```
python srs/concordance_ir.py \
  --qrels inputs/ir/qrels_kept_dpel.txt \
  --runs \
    runs_full/kept/bm25.txt \
    runs_full/kept/bge.txt \
    runs_full/kept/e5.txt \
    runs_full/kept/bm25_e5_rerank.txt \
    runs_full/kept/hybrid_rrf_bm25_e5.txt \
  --k 10 \
  --out-jsonl outputs/judging/analysis/concordance_kept_dpel.jsonl \
  --out-csv   outputs/judging/analysis/concordance_kept_dpel.csv

```

`concordance_ir.py` records, for each query:

-   **for each method:**
    
    -   whether any relevant passage appears in top-k (`hit_any`),
        
    -   whether all relevant passages appear (`hit_all`),
        
    -   rank of the first relevant hit,
        
-   **global counts:**
    
    -   `num_methods_hit_any`, `num_methods_hit_all`,
        
    -   simple labels: `high_concordance_any` (≥4/5 methods hit) and `low_concordance_any` (≤1/5).
        

## 5. Toward Pseudo-Gold Datasets

Because there is no human evaluation, RegRAG-Xref uses:

1.  **LLM-as-a-judge** (quality / dual-use), and
    
2.  **Multi-system IR concordance** (agreement across 5 retrievers)
    

...as a surrogate panel for human assessment.

The intended “gold-ish” splits:

-   **Gold-DPEL:**
    
    -   QAs from DPEL,
        
    -   labeled as `kept` by the judge and
        
    -   high IR/RAG concordance (e.g., ≥4/5 methods hit a relevant passage).
        
-   **Gold-SCHEMA:**
    
    -   same idea for SCHEMA.
        

Later, RAG experiments (5 RAG methods with the same retrieval suite) will provide extrinsic checks and help refine these splits.

## 6. Folder Layout (Canonical)

```
RegRAG-Xref/
├─ data/
│  ├─ CrossReferenceData.csv
│  ├─ passages_full.jsonl
│  └─ Documents/            # 40 source JSON docs
├─ passages_json/
│  ├─ collection_full.jsonl
│  └─ ...
├─ inputs/
│  └─ ir/
│     ├─ queries_kept.tsv
│     ├─ queries_eliminated.tsv
│     ├─ qrels_kept*.txt
│     └─ qrels_eliminated*.txt
├─ runs_full/
│  ├─ kept/
│  │  ├─ bm25.txt
│  │  ├─ bge.txt
│  │  ├─ e5.txt
│  │  ├─ bm25_e5_rerank.txt
│  │  └─ hybrid_rrf_bm25_e5.txt
│  └─ eliminated/
│     └─ (same pattern)
├─ outputs/
│  ├─ extracted_schema.jsonl
│  ├─ generation/
│  │  ├─ dpel/all/...
│  │  └─ schema/all/...
│  └─ judging/
│     ├─ ensemble/...
│     ├─ curated/
│     │  ├─ DPEL/{kept,eliminated}.jsonl
│     │  └─ SCHEMA/{kept,eliminated}.jsonl
│     └─ analysis/
│        └─ concordance_*.{jsonl,csv}
└─ srs/
   ├─ doc_manifest.py
   ├─ extract_schemas.py
   ├─ generate_qas_method_DPEL.py
   ├─ generate_qas_method_schema.py
   ├─ judge_qas_ensemble.py
   ├─ build_ir_inputs.py
   ├─ build_method_qrels.py
   ├─ build_full_passages.py
   ├─ run_dense_e5_sbert.py
   ├─ fuse_rrf.py
   ├─ rerank_bm25_with_e5.py
   ├─ eval_ir.py
   └─ concordance_ir.py

```

## 7. Quick Repro Recipe

Very short version:

1.  Build schemas (SCHEMA) & generate QAs (DPEL + SCHEMA)
    
2.  Run LLM-as-a-judge and produce curated `kept` / `eliminated` sets
    
3.  Build full passages + queries/qrels
    
4.  Run 5 IR systems over full corpus
    
5.  Evaluate & compute concordance per slice
    
6.  (Next) Define Gold-DPEL / Gold-SCHEMA and run RAG experiments.
    

## 8. Status & Future Work

✅ Dual-passage QAs from cross-references (DPEL + SCHEMA) ✅ LLM-based curation (ensemble judge) ✅ Full-corpus IR evaluation (5 methods) ✅ IR-based concordance analysis

**Planned:**

-   [ ] 5×RAG setups with the same retrieval suite
    
-   [ ] RAG-based concordance
    
-   [ ] Final Gold-DPEL / Gold-SCHEMA definitions
    
-   [ ] Intrinsic (stats) + extrinsic (RAG performance) evaluation write-up
