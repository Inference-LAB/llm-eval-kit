# LLM Eval Kit — Engineering Lab Notes
**INFERENCE Lab Engineering Fellowship · Cohort 01**  
*Project: `llm-eval-kit` — Lightweight, Deterministic Offline LLM Evaluation Engine*

---

## Executive Summary & Engineering Roster

Modern LLM evaluation workflows routinely depend on third-party "LLM-as-a-judge" APIs or human review teams. While flexible, this paradigm introduces severe latency bottlenecks, non-deterministic scores, prohibitive costs, and privacy vulnerabilities that prevent continuous evaluation within automated CI/CD pipelines. 

`llm-eval-kit` was engineered by a three-person team to provide an offline, deterministic, sub-second evaluation suite that scores model outputs locally using fast semantic embeddings, rule-based heuristics, and numeric consistency verification.

| Team Member | Role | Core Responsibility | Key Artifacts |
| :--- | :--- | :--- | :--- |
| **Mahrukh Baig** | Lead Engineer | Engine architecture, registry pattern, fail-fast orchestration, CLI, packaging | `registry.py`, `evaluator.py`, `cli.py`, `pyproject.toml` |
| **Muhammad Maaz** | Research / Implementation Engineer | Four core evaluation criteria, hybrid numeric verification, Urdu multilingual support | `criteria/`, `numeric_utils.py`, `embeddings.py`, regex & config loaders |
| **Warisha Arshad** | Evaluation Engineer | Test suites, ground truth fixtures, edge-case probing, SLA benchmarking | `tests/`, `tests/fixtures/`, test suites across all 4 criteria |

---

## Table of Contents
1. [Part 1: Architecting an Extensible, Deterministic Offline LLM Evaluation Engine (Mahrukh Baig)](#part-1-architecting-an-extensible-deterministic-offline-llm-evaluation-engine)
2. [Part 2: Building Deterministic Offline Evaluation Criteria for llm-eval-kit (Muhammad Maaz)](#part-2-building-deterministic-offline-evaluation-criteria-for-llm-eval-kit)
3. [Part 3: Building an Offline Evaluation Suite for llm-eval-kit (Warisha Arshad)](#part-3-building-an-offline-evaluation-suite-for-llm-eval-kit)
4. [Consolidated System Specifications & Outcomes](#consolidated-system-specifications--outcomes)

---

## Part 1: Architecting an Extensible, Deterministic Offline LLM Evaluation Engine
**Author:** Mahrukh Baig  
**Role:** Lead Engineer  
**Fellowship:** INFERENCE Lab Engineering Fellowship, Cohort 01  

Modern LLM evaluation workflows routinely depend on third-party "LLM-as-a-judge" APIs or human review teams. While flexible, this paradigm introduces severe latency bottlenecks, non-deterministic scores, prohibitive costs, and privacy vulnerabilities that prevent continuous evaluation within automated CI/CD pipelines. As Lead Engineer on `llm-eval-kit`, my objective was to design and build a clean, deterministic, and fully offline Python library capable of evaluating model outputs in milliseconds without cloud dependencies, subscriptions, or external network calls.

**System Architecture and the Registry Pattern.** The core architectural challenge was designing an evaluation engine that could scale across arbitrary criteria without creating a fragile monolithic codebase. Traditional evaluation pipelines rely on hardcoded conditional branches that require modifying the central orchestrator whenever a new metric is introduced. To eliminate this coupling, I designed a decorator-based criteria registry (`registry.py`) that maps criterion identifiers to their respective checking functions at import time. The central `Evaluator` class (`evaluator.py`) never interacts with criterion implementations directly; it solely inspects `CRITERIA_REGISTRY`. This guarantees that adding future criteria requires writing only an isolated function decorated with `@register_criterion`, without touching a single line of core orchestration code.

**Orchestration, Fail-Fast Guarantees, and Developer Experience.** In production pipelines, failing late after consuming heavy compute cycles is unacceptable. I implemented a strict fail-fast validation layer in `Evaluator.evaluate()`: before invoking any embedding models or compute-heavy heuristics, the engine cross-references all requested criteria against the registry. If an unknown or misspelled name is detected, execution halts immediately with a descriptive `ValueError`. Furthermore, I established a deliberate scoring convention: criteria that cannot run due to omitted inputs (e.g., factual grounding when reference context is absent) return `score: None` and are cleanly excluded from the overall average rather than penalized as zeros. To make the tool immediately accessible across CI workflows, I engineered a zero-overhead CLI wrapper (`cli.py`) powered by Typer, allowing developers to pipe model inputs and receive structured, machine-readable JSON outputs directly in their terminal.

**Packaging, Integration, and Engineering Outcomes.** Delivering a truly offline library required addressing subtle distribution pitfalls. I structured the project under `pyproject.toml` with explicit package data inclusion rules (`[tool.setuptools.package-data]`) to guarantee that static criterion configuration JSON files are bundled into distribution wheels, avoiding runtime `FileNotFoundError` exceptions in clean environments. Coordinating with Muhammad Maaz (Research Engineer) and Warisha Arshad (Integration Engineer), we established clear interface contracts that enabled independent workstreams across criteria development, fixture authoring, and integration testing. The combined test suite spans unit tests, regression tests, edge-case validation, and performance benchmarks—verifying sub-second sentence segmentation, memory-safe embedding singletons, and overall evaluation well within our 2,000ms SLA.

**What I Learned.** Leading this project reinforced the primacy of architectural minimalism. Establishing explicit interface contracts and single-responsibility boundaries early saved dozens of hours of integration churn. Designing for offline-first reliability proved that intelligent combination of dense semantic representations, symbolic parsing, and rule-based heuristics can match or exceed expensive LLM judges for targeted regression testing in continuous delivery.

Mahrukh Baig  
Lead Engineer  
INFERENCE Lab Engineering Fellowship, Cohort 01  

---

## Part 2: Building Deterministic Offline Evaluation Criteria for llm-eval-kit
**Author:** Muhammad Maaz  
**Role:** Research / Implementation Engineer  
**Fellowship:** INFERENCE Lab Engineering Fellowship, Cohort 01  

Evaluating Large Language Model (LLM) responses in production typically relies on human annotators or expensive, non-deterministic "LLM-as-a-judge" API calls. This creates latency bottlenecks, non-reproducible scores, and recurring operational costs that make continuous evaluation in local CI/CD pipelines impractical. `llm-eval-kit` set out to close this gap by building a deterministic, lightweight, and fully offline response evaluation library that scores model outputs locally using fast semantic embeddings, rule-based heuristics, and numeric consistency verification.

**What changed.** The evaluation suite evolved from basic prototype checks into a layered, four-criteria architecture: directional refusal detection with opening-window constraints and substantive-tail heuristics, factual grounding combining dense sentence embeddings with orthogonal numeric-claim verification, prompt-to-response topical relevance scoring, and rule-based aspect decomposition for completeness. Input defense mechanisms were integrated across all four criteria to handle edge cases—such as empty, whitespace, and null responses—without incurring unnecessary transformer compute or producing tokenization artifacts.

**The numeric verification and grounding model.** To detect factual contradictions and hallucinations without external world-knowledge graphs, I developed a hybrid verification layer on top of dense sentence embeddings. Standard semantic similarity treats syntactically identical sentences with swapped numbers (e.g., "water boils at 50°C" versus "water boils at 100°C") as near-identical (~0.90 similarity), failing to catch blatant factual errors. To solve this, I designed an extraction and validation engine in `numeric_utils.py` that isolates numeric claims (values, units, scientific notation, negative numbers, and spelled-out scales up to billions) and verifies them against the union of all context sentences using a subset check. If an unsupported figure appears, the similarity score is capped at 0.30. This multi-sentence union approach allows responses to synthesize facts across multiple sentences while still penalizing ungrounded numbers. This check is an offline consistency heuristic rather than an absolute fact-checker, as it targets numeric and semantic alignment rather than unstated world knowledge.

**Engineering outcomes.** Automated test coverage expanded to 51 unit tests across three dedicated test suites (`test_numeric_parser.py`, `test_edge_cases.py`, and `test_relevance_completeness.py`), achieving 100% test pass rates across all four criteria. Performance benchmarks on 5,000-word documents demonstrated that sentence segmentation runs in under 0.5 milliseconds using linear lookbehind regexes with zero ReDoS risk, while batched embedding inference on CPU processes 250 sentences in ~350 milliseconds. Multilingual adaptations added 13 curated Urdu refusal phrases, native Urdu punctuation parsing (`۔`, `؟`), and Eastern Arabic numeral translation (`۰–۹` to `0–9`).

**What I learned.** Working through architectural iterations reinforced concrete software engineering habits: implementing lazy-loaded model singletons so importing the library incurs zero initialization penalty, externalizing phrase lists and thresholds into JSON configs so non-code changes avoid code reviews, and defensively handling boundary inputs. Discovering that empty strings produce non-zero cosine similarities when mapped to BERT's `[CLS]` and `[SEP]` tokens was a reminder that deep learning primitives must always be wrapped with defensive pre-encoding guards.

Building an offline, deterministic evaluation suite that combines dense representations with fast symbolic sanity checks—while being explicit about where heuristic approximations end and absolute verification begins—was the central engineering contribution of this work.

Muhammad Maaz  
Research / Implementation Engineer  
INFERENCE Lab Engineering Fellowship, Cohort 01  

---

## Part 3: Building an Offline Evaluation Suite for llm-eval-kit
**Author:** Warisha Arshad  
**Role:** Evaluation Engineer  
**Fellowship:** INFERENCE Lab Engineering Fellowship, Cohort 01  

Testing LLM answers usually needs human checking or paid AI tools. Both are slow and cost money. `llm-eval-kit` wanted to fix this by building a free, offline tool that checks LLM answers by itself. My job was to test if it really worked.

**What I did.** I made test cases—example prompts, answers, and context—where I already knew what score they should get. I did this for all four checks: factual grounding, relevance, refusal detection, and completeness. I also wrote tests to check the whole system works correctly, and checked that it runs fast enough to use daily.

**Why this was hard.** Making test cases was not easy. I had to understand how each check worked before I could guess the right score for it. Simple-looking cases, like changed numbers or half-complete answers, needed careful thinking so the test was actually useful.

**What I learned.** This work taught me to think like someone trying to break the system, not just test the easy cases. I also learned where simple similarity scores work well, and where extra checks (like number checking) are needed to catch mistakes.

Building the testing part of a free, offline, reliable evaluation tool—one that works without internet or extra cost—was my main contribution.

Warisha Arshad  
Evaluation Engineer  
INFERENCE Lab Engineering Fellowship, Cohort 01  

---

## Consolidated System Specifications & Outcomes

```
+---------------------------------------------------------------------------------------+
|                                    llm-eval-kit                                       |
|               Deterministic, Offline Evaluation Engine for Local CI/CD                |
+---------------------------------------------------------------------------------------+
                                           |
      +------------------------------------+------------------------------------+
      |                                    |                                    |
      v                                    v                                    v
[ Mahrukh Baig ]                  [ Muhammad Maaz ]                   [ Warisha Arshad ]
- Registry Pattern                - 4 Core Criteria                   - Ground Truth Fixtures
- Fail-Fast Layer                 - Hybrid Numeric Verification       - Edge Probing (Empty/Null)
- Typer CLI Wrapper               - Urdu Localization                 - End-to-End Test Suite
- Packaging / Wheel Data          - Lazy Model Singletons             - SLA Performance Audits
```

### Performance & Latency Benchmarks
- **Sentence Segmentation:** < 0.5 ms for 5,000-word documents (linear lookbehind regexes, 0 ReDoS risk).
- **CPU Embedding Inference:** ~350 ms for 250 sentences via batched `all-MiniLM-L6-v2` singletons.
- **End-to-End Evaluation SLA:** < 2,000 ms per complete request (typically ~400–600 ms).
- **Automated Verification:** 100% test pass rate across 51+ unit, edge-case, and regression tests.
- **External Dependencies:** 0 network calls, 0 API subscriptions, 100% offline air-gapped support.
