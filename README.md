
# RegRAG-Xref: Methodology (Dataset Generation & Curation)

_Last updated: 30 Oct 2025 (Asia/Dubai)_

## 1) Corpus & Cross-Reference Ground Truth

-   **Source corpus:** 40 ADGM/FSRA regulatory documents.
    
-   **Pre-processing:** documents are manually structured and segmented into passages (rule/section granularity).
    
-   **Cross-references:** all intra- and inter-document references are manually extracted and consolidated into:
    
    -   `data/CrossReferenceData.csv`
        

**Illustrative fields**

-   `SourceDocumentName`, `SourcePassageID`, `SourceText`
    
-   `TargetDocumentName`, `TargetPassageID`, `TargetText`
    
-   `ReferenceType` ∈ {Internal, External, NotDefined}
    
-   `ReferenceText` (e.g., “Rule 3.6A.4”, “95(2)”)
    

This CSV is the canonical supervision for paired evidence (SOURCE ↔ TARGET).

## 2) Dataset Generation (Two Methods, Two Personas)

We generate QA pairs such that each QA must require BOTH passages (SOURCE & TARGET). Two complementary methods:

### 2.1 Personas (same for both methods)

**PROFESSIONAL_STYLE**

```
"Write the question like a regulator or compliance counsel. Prefer precise terms
(Issuer, Applicant, RIE, Authorised Person) and crisp modality (must/shall/may).
Questions may be multi-clause or two sentences to encode scope, preconditions,
exceptions, or timing. Tone: formal and unambiguous."

```

**BASIC_STYLE**

```
"Write the question for a smart non-expert compliance analyst. Use plain words, short
sentences, and clear structure. Questions can be longer when needed to state conditions
(if/when/unless), but prefer one or two short sentences. Keep actor names exactly as written."

```

**System discipline (both methods)**

```
"You generate regulatory Q&As and must follow the user instructions exactly.
Use ONLY the provided SOURCE and TARGET texts (no outside knowledge).
Every substantive claim must be grounded in at least one of the two passages.
Return VALID JSON only—no markdown, no commentary."

```

**Traceability (answers must include both):**

-   `[#SRC:<source_passage_id>]` and `[#TGT:<target_passage_id>]`
    

### 2.2 Method A — DPEL (Direct Passage Evidence Linking)

-   **Input:** `data/CrossReferenceData.csv`
    
-   **Goal:** generate dual-passage QAs directly from cross-reference pairs.
    
-   **Pre-processing filters**
    
    -   **Empty/degenerate pairs:** skip if `source_text` or `target_text` is empty; also skip if `source_id == target_id` or `source_text == target_text` (cannot require dual evidence).
        
    -   **Sampling/cap:** `--row_sample_n` (random sample), `--max_pairs` (hard cap).
        
    -   **(Optional) drop title-like targets:** `--drop_title_targets` (if implemented in your DPEL script).
        
-   **Prompt & generation logic (key rules)**
    
    -   **Core goal:** center the question on the pair’s substance and ensure the answer needs both passages.
        
    -   **Persona:** controls question style only; answers are always professional.
        
    -   **Answer requirements:**
        
        -   **Length:** compact professional paragraph (~180–230 words; hard minimum 160).
            
        -   **Tags:** must include both `[#SRC:…]` and `[#TGT:…]`.
            
    -   **Citation policy (DPEL):** questions may include citations.
        
    -   **Lexical hints:** prompt can include “light lexical hints” (tokens unique to `source_text`) to keep the question anchored.
        
-   **Post-collection validation**
    
    -   **Tag enforcement:** save only if the answer includes both exact tags.
        
    -   **Dedup:** if `--dedup` is set, normalize question text and drop duplicates.
        
-   **Example command**
    
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
    
-   **Output JSON (per line)**
    
    ```
    {
      "qa_id": "…",
      "persona": "professional|basic",
      "question": "…",
      "expected_answer": "… [#SRC:…] … [#TGT:…] …",
      "debug_context": {
        "source_passage_id": "…",
        "target_passage_id": "…",
        "source_text": "…",
        "target_text": "…",
        "reference_type": "Internal|External|NotDefined",
        "reference_text": "Rule 3.6A.4 | 95(2) | …",
        "semantic_hook": "",
        "citation_hook": "",
        "answer_spans": [],
        "source_item_type": "",
        "target_item_type": ""
      },
      "method": "DPEL",
      "gen_model": "gpt-4o",
      "gen_ts": 0,
      "run_seed": 13
    }
    
    ```
    

### 2.3 Method B — SCHEMA (Schema-Anchored Generation)

**Step-1: Schema extraction (per SOURCE, TARGET pair)**

```
python srs/extract_schemas.py \
  --input_csv data/CrossReferenceData.csv \
  --output_jsonl outputs/extracted_schema.jsonl \
  --model gpt-4o \
  --drop_title_targets

```

**Schema fields (per item)**

-   `item_id` (UUID)
    
-   `reference_type`, `reference_text` (copied from CSV)
    
-   `semantic_hook` — short, citation-free phrase (6–12 tokens) capturing core action/policy (from source).
    
-   `citation_hook` — citation-like token (e.g., “Rule 9.7.5”); prefer `reference_text` if valid.
    
-   `source_passage_id`, `source_text`, `target_passage_id`, `target_text`
    
-   `source_item_type`, `target_item_type` ∈ {Obligation, Prohibition, Permission, Definition, Scope, Procedure, Other}
    
-   `answer_spans` — up to 3 spans from target with types: FREEFORM, DURATION, DATE, MONEY, PERCENT, TERM, SECTION
    
-   `target_is_title` (bool)
    
-   `provenance` (model, timestamp)
    

**Extraction heuristics**

-   **Title handling:** a heuristic flags `target_is_title`. With `--drop_title_targets`, such items are skipped.
    
-   **Span validation/fallback:**
    
    1.  Accept LLM spans if offsets/types are valid within `target_text`.
        
    2.  Else pick a core clause (modal sentence: must/shall/may) as FREEFORM.
        
    3.  Else take the first ≤220 chars of `target_text` as FREEFORM.
        
    4.  Titles → `answer_spans = []`.
        
-   **Soft dedupe:** hash {`source_text`, `target_text`, hooks, `reference_text`}; drop exact repeats.
    

**Item-type phrasing hints (used later in prompts)**

-   **Obligation** → must/shall; deadlines/requirements
    
-   **Prohibition** → must not/shall not/is prohibited; exceptions
    
-   **Permission** → may/can/is permitted; conditions
    
-   **Definition** → precise criteria; avoid long quotes
    
-   **Scope** → applicability boundaries (who/what/when/exclusions)
    
-   **Procedure** → steps/approvals/calculations; minimal but clear
    
-   **Other** → no extra guidance
    

**Step-2: QA generation from schema**

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

**Pre-generation filters**

-   Skip empty/degenerate pairs (`source_id == target_id` or identical texts).
    
-   Optional sampling/cap (`--row_sample_n`, `--max_pairs`).
    
-   (If enabled in this script) `--drop_title_targets` to skip `target_is_title`.
    

**Prompt & generation logic**

-   **Core goal:** center the question on the `semantic_hook`; ensure the answer truly needs both passages.
    
-   **Persona:** controls question style only; answers are always professional.
    
-   **Answer requirements:**
    
    -   **Length:** ~180–230 words (min 160).
        
    -   **Tags:** include both exact tags.
        
-   **Schema-driven guidance:**
    
    -   **Item types:** inject phrasing hints (above).
        
    -   **Structured spans:** if `answer_spans` include DATE/MONEY/TERM/SECTION/..., the answer must explicitly include those details.
        
-   **Lexical hints:** include light hints from `source_text` (words not found in `target_text`).
    
-   **Dual-anchors enforcement:**
    
    -   `--dual_anchors_mode freeform_only` (default): enforce when all spans are FREEFORM.
        
    -   `--dual_anchors_mode always`: enforce for all items.
        
-   **Citation policy (SCHEMA):**
    
    -   By default, questions may include citations.
        
    -   Add `--no_citations` to forbid citations in the question (the `[#SRC/TGT]` tags are not “citations”).
        
-   **Post-collection validation**
    
    -   **Tag enforcement:** require both tags with distinct IDs.
        
    -   **Dedup:** drop normalized duplicate questions.
        
-   **Output:** same shape as DPEL, but `debug_context` includes full schema (`semantic_hook`, `citation_hook`, `answer_spans`, `source_item_type`, `target_item_type`).
    

## 3) Post-Generation Diagnostics

Run a quick inspection over one or more generated files.

```
python srs/inspect_qas_stats.py \
  --inputs outputs/generation/dpel/all/answers.jsonl \
           outputs/generation/schema/all/answers.jsonl \
  --out_csv outputs/generation/qa_stats_summary.csv \
  --sample 5

```

**Reports**

-   Totals by method/persona
    
-   `ReferenceType` distribution (Internal/External)
    
-   Item types (for SCHEMA)
    
-   Length stats for questions/answers
    
-   Question citation rate (helps audit SCHEMA `--no_citations`)
    
-   Answer tag completeness ([#SRC:], [#TGT:])
    
-   Duplicates (within & across inputs)
    

## 4) Curation via LLM-as-a-Judge (Ensemble)

We filter QAs using a rubric tuned to cross-reference reasoning.

### 4.1 Inputs per QA

`question`, `expected_answer`, `source_text`, `target_text`, passage IDs, `method` (DPEL/SCHEMA).

### 4.2 Hard gates (auto-fail → score=0)

-   **Local pre-checks (script):**
    
    -   **Tag gate:** `expected_answer` must include both `[#SRC:id]` and `[#TGT:id]`.
        
    -   **Question citation gate (SCHEMA only, optional):** if `--forbid_citations_in_question_for_schema`, then the question must not include citation-like tokens (regex `CITATION_PAT`). DPEL is exempt.
        
    -   **Answer citation gate (optional):** if `--allow_citations_in_answer` is not set, then the answer must not contain citations.
        
-   **LLM-judged gate (in the rubric):**
    
    -   **Dual-evidence gate:** the question must require BOTH passages. If fully answerable from only one passage, this gate fails.
        

### 4.3 Scoring rubric (integers)

-   **Realism (0–2):** realistic compliance question.
    
-   **Dual_use (0–4):** removing either passage breaks correctness.
    
-   **Correctness (0–4):** answer accurate & grounded in SOURCE/TARGET.
    
-   `final_score = realism + dual_use + correctness` (0–10)
    

**Pass rule (fused):**

1.  All hard gates pass, **and**
    
2.  `final_score ≥ 7`, **and**
    
3.  `dual_use ≥ 3`.
    

### 4.4 Ensemble & fusion

-   **Models:** small/light LLMs, e.g., `gpt-4.1-mini`, `gpt-4o-mini`, plus a repeat of the first model with a different `--repeat_first_with_seed`.
    
-   **Fusion logic:**
    
    -   **Subscores & final score** → median across judges.
        
    -   **Pass/fail** → majority vote.
        
    -   **Tie-break:** count judges with `dual_use ≥ 3`. If `< require_dual_use_k` (default 2) → fail. Otherwise, compare medians of “pass” vs “fail” voters’ `final_score`; pass if “pass” median ≥ “fail” median.
        

### 4.5 Run the judge

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

**Outputs**

-   Per-QA judgments (individual judges + fused).
    
-   Summary report: pass rates by method/persona; averages of fused subscores and distributions.
    

## 5) Folder Layout (canonical)

```
RegRAG-Xref/
├─ data/
│  └─ CrossReferenceData.csv
├─ outputs/
│  ├─ extracted_schema.jsonl
│  ├─ generation/
│  │  ├─ dpel/
│  │  │  └─ all/
│  │  │     ├─ answers.jsonl
│  │  │     └─ report.json
│  │  └─ schema/
│  │     └─ all/
│  │        ├─ answers.jsonl
│  │        ├─ answers_nociteQ.jsonl  # (SCHEMA with --no_citations)
│  │        └─ report.json
│  ├─ judging/
│  │  └─ ensemble/
│  │     ├─ judgments.jsonl
│  │     └─ summary.json
│  └─ generation/
│     └─ qa_stats_summary.csv
└─ srs/
   ├─ extract_schemas.py
   ├─ generate_qas_method_DPEL.py
   ├─ generate_qas_method_schema.py
   ├─ inspect_qas_stats.py
   └─ judge_qas_ensemble.py

```

## 6) Repro Quickstart

**Schema extraction (Method B)**

```
python srs/extract_schemas.py \
  --input_csv data/CrossReferenceData.csv \
  --output_jsonl outputs/extracted_schema.jsonl \
  --model gpt-4o \
  --drop_title_targets

```

**Generate QAs (DPEL)**

```
python srs/generate_qas_method_DPEL.py \
  --input_csv data/CrossReferenceData.csv \
  --output_jsonl outputs/generation/dpel/all/answers.jsonl \
  --report_json outputs/generation/dpel/all/report.json \
  --model gpt-4o --max_q_per_pair 2 --sample_n 3 --temperature 0.2 --seed 13 \
  --dedup --verbose

```

**Generate QAs (SCHEMA)** (add `--no_citations` to forbid citations in questions)

```
python srs/generate_qas_method_schema.py \
  --input_jsonl outputs/extracted_schema.jsonl \
  --output_jsonl outputs/generation/schema/all/answers.jsonl \
  --report_json outputs/generation/schema/all/report.json \
  --model gpt-4o --max_q_per_pair 2 --sample_n 3 --temperature 0.2 --seed 13 \
  --dual_anchors_mode freeform_only --dedup --verbose

```

**Inspect**

```
python srs/inspect_qas_stats.py \
  --inputs outputs/generation/dpel/all/answers.jsonl \
           outputs/generation/schema/all/answers.jsonl \
  --out_csv outputs/generation/qa_stats_summary.csv \
  --sample 5

```

**Judge (ensemble)**

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
  --temperature 0.0 --seed 13 --verbose

```

**Curate final set:** keep QAs with fused pass.

## 7) Argument Reference (quick tables)

**Generation (DPEL/SCHEMA shared)** 
| Arg | Meaning | 
|---|---| 
| `--model` | LLM name for generation | 
| `--max_q_per_pair` | Max Qs per persona per pair | 
| `--sample_n` | Brainstorm cap hint per persona | 
| `--temperature` | Decoding temperature | 
| `--seed` | RNG seed (and LLM seed where supported) | 
| `--dedup` | Drop duplicate questions (normalized) | 
| `--row_sample_n` | Randomly sample N items from input | 
| `--row_sample_seed` | RNG seed for sampling | 
| `--max_pairs` | Hard cap on total items processed | 
| `--verbose` | Periodic progress logs | 
| `--dry_run` | Scan/filter only; no model calls |

**SCHEMA-only** 
| Arg | Meaning |
 |---|---|
  | `--input_jsonl` | Schema file from `extract_schemas.py` | 
  | `--dual_anchors_mode` | `{off,freeform_only,always}`Enforce explicit SOURCE+TARGET anchors | 
  | `--no_citations` | Forbid citations in question (tags still required) | 
  | `--drop_title_targets` | Skip items flagged `target_is_title` |

**Judge (ensemble)** 
| Arg | Meaning | 
|---|---| 
| `--inputs` | One or more generated JSONLs | 
| `--ensemble_models` | Comma-sep model list (e.g., `gpt-4.1-mini,gpt-4o-mini`) | 
| `--repeat_first_with_seed` | Add a second pass of first model with a new seed | 
| `--pass_threshold` | Min fused score to pass (default 7) | 
| `--require_dual_use_k` | Tie-break: min #judges with `dual_use≥3` (default 2) | 
| `--forbid_citations_in_question_for_schema` | Enforce no citations in SCHEMA questions | 
| `--allow_citations_in_answer` | If omitted, citations in answers are forbidden | 
| `--temperature`, `--seed`, `--verbose` | Usual controls |

## 8) Notes & Recommendations

-   Dual-passage integrity is non-negotiable; the judge’s dual-evidence gate is central.
    
-   Structured spans → explicit details in the answer (DATE/MONEY/TERM/SECTION…).
    
-   Citation style:
    
    -   DPEL: questions may cite.
        
    -   SCHEMA: prefer no citations in Q (use `--no_citations` + judge enforcement).
        
-   Tuning: stricter curation → set `--pass_threshold 8` and/or `--require_dual_use_k 3`.
    
-   Reproducibility: keep `--seed` fixed across runs; capture `gen_model` and timestamps (`gen_ts`) already saved in outputs.
    

## 9) Deliverables

-   Generated QA datasets (per method/persona) with full `debug_context` and IDs.
    
-   Judgment logs (per-judge + fused) and a summary JSON.
    
-   Stats CSV (sizes, citation rates, lengths, dups).
    

## 10) Future Work

-   Topic-balanced sampling across `ReferenceType`/`ItemType`.
    
-   Human spot-checks on borderline fused cases (`final_score` ∈ {6,7} or `dual_use`==3).
    
-   Public leaderboard split with hidden evidence for benchmarking dual-passage RAG.
