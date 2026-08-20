# Fermi data preparation

This directory prepares a merged Fermi-problem dataset from three sources:

- `SynthFP`: 10,000 synthetic problems from AllenAI.
- `RealFP`: real-world problems from AllenAI.
- `Fermi-Eval`: Science Olympiad questions from Open Scioly.

The current outputs contain 11,680 training rows, 1,460 validation rows, and 1,460 test rows. They are built from pinned source revisions with a deterministic default seed of `42`.

## Sources

- AllenAI [Fermi repository](https://github.com/allenai/fermi) and EMNLP 2021 paper, [*How Much Coffee Was Consumed During EMNLP 2019? Fermi Problems: A New Reasoning Challenge for AI*](https://aclanthology.org/2021.emnlp-main.582/).
- Fermi-Eval [repository](https://github.com/landy8697/open-scioly-fermi), [practice site](https://landy8697.github.io/open-scioly-fermi/), the paper [*LLMs are Overconfident: Evaluating Confidence Interval Calibration with FermiEval*](https://arxiv.org/abs/2510.26995).

## Usage

From the repository root:

```bash
uv sync --project fermi --locked
uv run --project fermi fermi-data all --refresh
```

Later runs can omit `--refresh`.

`all` runs the full pipeline: download the pinned sources, prepare and audit Fermi-Eval, then merge, split, tokenize, and report the data.

The same stages can be run separately:

```bash
uv run --project fermi fermi-data download --refresh
uv run --project fermi fermi-data audit
uv run --project fermi fermi-data process
```

- `download` fetches the pinned AllenAI and Open Scioly repositories into `fermi/.cache/`.
- `audit` normalizes and deduplicates Fermi-Eval, removes exact overlaps with the AllenAI questions, and applies a rule-based heuristic filter to discard questions lacking necessary units.
- `process` converts all sources to the common schema, creates deterministic per-source 80/10/10 splits, merges them and shuffles, and writes the datasets and analysis outputs.

## Data preparation

AllenAI's published train, validation, and test files are pooled for `SynthFP` and `RealFP`. Their answer strings are separated into numeric value and unit fields while retaining the solution program and supporting context.

For Fermi-Eval, the pipeline reads Open Scioly's `data.js`. Its integer answer exponent `K` is stored as `1eK`; solution program and context fields are empty. Questions are normalized, deduplicated, checked for exact overlap with AllenAI, and filtered with the unit heuristic described above.

Each source is then independently shuffled and split 80/10/10. The corresponding source splits are concatenated and shuffled to produce the final train, validation, and test files.

Examples of answer conversion:

| Source answer | `answer_value` | `answer_unit` |
| --- | --- | --- |
| `5.573750E+08 kg` | `5.573750E+08` | `kg` |
| `$7755273000000` | `7755273000000` | `$` |
| Fermi-Eval exponent `12` | `1e12` | empty |
| Fermi-Eval exponent `-4` | `1e-4` | empty |

## Output columns

The three output Parquet files share this schema:

| Column | Description |
| --- | --- |
| `question` | Problem text. |
| `answer_value` | Numeric answer stored as text. |
| `answer_unit` | Unit or currency marker supplied with the answer, or an empty string. |
| `program` | AllenAI solution program; empty for Fermi-Eval. |
| `context` | AllenAI supporting context; empty for Fermi-Eval. |
| `problem_source` | `SynthFP`, `RealFP`, or the attributed Fermi-Eval source and question number. |

## Outputs

The merged datasets are:

```text
fermi/outputs/fermi_train.parquet
fermi/outputs/fermi_val.parquet
fermi/outputs/fermi_test.parquet
```

`fermi_analysis.txt` records source counts and summary statistics. `fermi_answer_log_distribution.png` and `fermi_token_lengths.png` visualize answer magnitudes and token lengths; tokenization uses `Qwen/Qwen3-0.6B` by default.

## Example rows

| Source | `question` | `answer_value` | `answer_unit` |
| --- | --- | --- | --- |
| `RealFP` | How long would it take to count to a million? | `11.5` | `days` |
| `Fermi-Eval` | What is the distance to the sun in meters? | `1e11` | empty |
