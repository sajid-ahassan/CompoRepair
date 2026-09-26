# CompoRepair-RAG

**A Controlled Study of Compositional Repair, Order Effects, and Regression in Retrieval-Augmented Generation**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Research Artifact](https://img.shields.io/badge/Artifact-Reproducibility-4C8BF5)](https://github.com/sajid-ahassan/CompoRepair/releases)

CompoRepair-RAG is a controlled empirical study of whether failure-specific RAG repair capabilities that work in isolation remain effective and safe when multiple failures occur together. The study evaluates compositions of three failure families:

- **M — Missing evidence:** required supporting information is absent.
- **D — Distractor evidence:** plausible but misleading evidence is introduced.
- **L — Evidence-linkage failure:** an answer-critical relationship between pieces of evidence is disrupted.

Rather than evaluating repair only through final answer accuracy, CompoRepair-RAG measures semantic recovery, repair-induced regression, structural restoration, Joint Full Repair, and repair-order effects.

## Research questions

The project investigates:

1. Whether repair capabilities established under isolated failures retain their utility under pairwise and triple failure compositions.
2. Whether repair improves failed traces without damaging traces that were already correct.
3. Whether restoring the intended evidence structure also restores semantic correctness.
4. Whether the order of specialized repair operations materially changes the outcome.

## Experimental design

The controlled evaluation uses:

- **Datasets:** HotpotQA and 2WikiMultiHopQA
- **Generator models:** Llama 3.1 8B and Qwen3.5 9B through Ollama
- **Failure conditions:** `M`, `D`, `L`, `M_D`, `M_L`, `D_L`, and `M_D_L`
- **Compound repair strategies:** fixed specialized composition, reverse composition, generic self-reflection, and one-shot joint repair
- **Statistical procedures:** paired bootstrap confidence intervals, exact two-sided McNemar/binomial tests, and Holm correction for multiple comparisons

The active failure family is known when selecting the corresponding repair module. The study therefore evaluates **oracle-routed repair composition**, not autonomous failure diagnosis.

## Main findings

- Fixed specialized composition improved semantic accuracy over the unrepaired state in all 16 evaluated compound settings, with every improvement remaining significant after Holm correction.
- It achieved higher semantic accuracy than both generic repair approaches in all settings, with 15 of 16 comparisons significant against each generic approach.
- Repair recovery can coexist with repair-induced regression, so final accuracy alone is insufficient for evaluating reliability.
- Structural restoration and semantic correctness can diverge.
- Semantic repair-order effects were localized rather than universal; no single ordering should be assumed optimal in every setting.

These findings are bounded to the evaluated datasets, models, controlled interventions, eligible traces, and oracle-routing design.

## Repository structure

```text
CompoRepair/
├── src/                         # Experiment and repair implementation
│   └── comporepair/
│       ├── analysis/            # Experiment-level analysis utilities
│       ├── models/              # Local model interface
│       ├── pipeline/            # Baseline RAG pipeline and state
│       ├── repair/              # Failure-specific repair modules
│       └── retrieval/           # Retrieval and vector-store utilities
├── Statistical_Analysis/        # Public statistical analysis package
│   ├── outputs/                 # Aggregate results and validation outputs
│   ├── tests/                   # Statistical helper tests
│   ├── analyze_comporepair_stats.py
│   └── validate_results.py
├── requirements.txt
├── run_pipeline.ps1
├── test.py
└── test.ipynb
```

## Installation

### Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) for local model execution
- Access to HotpotQA and 2WikiMultiHopQA under their respective terms

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install or configure the model identifiers required by the experiment before running the pipeline. Dataset files and model services are not downloaded automatically by this repository.

## Running the experiment pipeline

On Windows, the repository includes a PowerShell runner:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

Before execution, review the script and local configuration paths for dataset locations, output directories, model identifiers, and Ollama availability. The complete experiment is computationally expensive and was designed for resumable local execution.

## Statistical analysis

The public statistical package includes the analysis code, aggregate outputs, tests, run metadata, and validation report.

To inspect and validate the published aggregate results:

```bash
cd Statistical_Analysis
python -m pip install -r requirements.txt
python validate_results.py
```

To regenerate the statistics from the authorized trace-level archive, place the archive in `Statistical_Analysis/` as `raw_results.zip`, verify its SHA-256 value, and run:

```bash
python analyze_comporepair_stats.py raw_results.zip --output outputs --bootstrap 10000 --seed 20260830
python validate_results.py
```

Convenience runners are also provided:

```powershell
.\run_analysis.bat
```

```bash
./run_analysis.sh
```

## Public outputs

The `Statistical_Analysis/outputs/` directory contains derived aggregate artifacts, including:

- Complete machine-readable statistical results
- Single-repair and compound-repair summaries
- Primary paired semantic and multi-metric comparisons
- Pairing and evaluation-quality audits
- Paper-oriented result summaries
- Run metadata and validation reports

These files are derived statistical outputs. They do not redistribute the original benchmark records or the complete trace-level experimental archive.

## Data and artifact availability

The original HotpotQA and 2WikiMultiHopQA records are not redistributed. The trace-level `raw_results.zip` is also excluded because it contains benchmark-derived content. It may be provided for research verification upon reasonable request, subject to the source datasets’ applicable terms.

The repository and release artifacts provide the experiment code, statistical analysis code, aggregate outputs, tests, metadata, and validation materials required to inspect the reported analyses.

### Recorded SHA-256 identifiers

```text
Source snapshot:
8732a1f5475cb3b4256530bb08a6de61320a297216941f9416e40cc7f3c5ebf9

Trace-level raw-results archive (not publicly redistributed):
18ed827e962204ec7122a7f86eee3cbe521acea299a314ebcde44a197803c497

Public statistical-analysis archive:
a0444d6057710144b93418b9238c7875e91faba537e53fe328e25ce0761d3dc2
```

## Citation

If you use this repository, please cite the project as:

```bibtex
@misc{ahasan2026comporepairrag,
  author       = {Sajid Ahasan},
  title        = {CompoRepair-RAG: A Controlled Study of Compositional Repair,
                  Order Effects, and Regression in Retrieval-Augmented Generation},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/sajid-ahassan/CompoRepair}
}
```

The citation should be updated with the final venue, DOI, and publication details after publication.

## Author

**Sajid Ahasan**  
Daffodil International University  
[ahasan241-15-909@diu.edu.bd](mailto:ahasan241-15-909@diu.edu.bd)

## Research-use notice

This repository is released as a research artifact. Users are responsible for complying with the licenses and terms of the source datasets, model providers, and third-party dependencies.
