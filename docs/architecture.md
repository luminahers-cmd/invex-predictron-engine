# System Architecture
<p align="center">
  <img src="../assets/architecture.png"
       alt="Predictron Engine Architecture"
       width="100%">
</p>

## Overview

Predictron Engine is a modular venture intelligence system designed to transform publicly available startup information into structured, explainable intelligence.

Rather than relying on a single prediction model, the engine decomposes startup analysis into independent intelligence domains. Each domain extracts structured signals that are aggregated into evidence, reasoned over, and synthesized into a comprehensive venture intelligence report.

The architecture emphasizes explainability, deterministic extraction, modularity, and continuous evaluation.

---

## High-Level Architecture

```text
Public Startup Data
        │
        ▼
Collection + Website Evidence Collection
        │
        ▼
Extraction
 ├── Company / Market / Founder
 ├── Product / Technology / Business Model
 └── Traction / Competition / Risk / Metadata
        │
        ▼
Evidence Gathering (knowledge providers)
        │
        ▼
Reasoning (observations)
        │
        ▼
Evaluation (dimension assessments) ──► Scoring
        │
        ▼
Recommendation + Confidence
        │
        ▼
Investment Decision
        │
        ▼
Calibration ──► Decision Synthesis
        │
        ▼
Report Generation
```

---

## Core Components

### Data Collection

Predictron Engine begins by collecting publicly available startup information, including company websites, product descriptions, documentation, and other structured sources.

The objective of this stage is to normalize raw startup information before downstream analysis.

---

### Intelligence Extraction

Instead of treating startup information as unstructured text, the engine organizes knowledge into specialized intelligence domains.

Current extraction modules include (each a single-responsibility extractor
composed by the `CompositeExtractor`, then enriched by deterministic
derived-metric inference):

- Company
- Market
- Founder
- Product
- Technology
- Business Model
- Traction
- Competition
- Risk
- Metadata

Each module produces structured features that can be independently validated, benchmarked, and extended without affecting the remainder of the system.
Evaluation
---

Scoring
### Evidence Framework

Recommendations
Extracted features are transformed into structured evidence.

Confidence
The Evidence Framework maintains traceability between raw observations and downstream reasoning, allowing every significant conclusion to be supported by observable evidence rather than opaque inference.

Report Generation

### Reasoning Framework

The Reasoning Framework synthesizes evidence across intelligence domains.

Rather than evaluating isolated signals, it considers relationships between market, founders, product, technology, and business model to generate coherent venture intelligence.

This separation between evidence generation and reasoning improves transparency, maintainability, and future extensibility.

---

### Venture Scoring

Structured evidence contributes to multiple evaluation dimensions.

Scores are intended to support decision-making by summarizing evidence across the venture, while preserving the underlying reasoning that produced each assessment.

---

### Report Generation

The final stage produces structured venture intelligence reports containing:

- Intelligence summaries
- Supporting evidence
- Venture scores
- Confidence indicators
- Key observations

The generated report is designed to be explainable, reproducible, and suitable for further human review.

---

## Architectural Principles

Predictron Engine is built around five engineering principles:

- Explainability
- Modular architecture
- Deterministic extraction
- Structured reasoning
- Continuous evaluation

These principles allow individual components to evolve independently while preserving transparency, reproducibility, and engineering rigor across the system.
