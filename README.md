# LoRA composition benchmark

This repository contains a controlled benchmark for studying when independently trained LoRA adapters retain their constituent capabilities, interfere with one another, or implement an unseen composed transformation. It includes deterministic structured-record tasks, atomic and composite controls, additive and CAT composition, structured evaluation, saved predictions, and adapter-geometry diagnostics.

## Setup

```bash
pip install -r requirements.txt
```

The single experiment configuration is [`configs/config.yaml`](configs/config.yaml). It contains the model, dataset, LoRA, training, evaluation, atomic-task, and composition settings.

## Run atomic adapters

```bash
python main.py --config configs/config.yaml --mode atomic --run-name atomic
```

Use `--skills` followed by one or more task IDs to run a subset.

## Run compositions

```bash
python main.py --config configs/config.yaml --mode compose \
  --run-name selection_sorting \
  --composition-pairs value_selection:field_sorting
```

Optional sweeps:

```bash
python main.py --config configs/config.yaml --mode compose \
  --run-name selection_sorting_sweep \
  --composition-pairs value_selection:field_sorting \
  --sweep-weights --sweep-methods
```

The selection--sorting generator always requires at least two selected fields and a substantive reordering step.

## Outputs

- `datasets/`: generated atomic and composite examples plus manifests.
- `adapters/`: saved LoRA weights and tokenizer/trainer artifacts.
- `runs/atomic/`: atomic training runs and manifests.
- `runs/composition/`: composition diagnostics, controls, predictions, and manifests.
- `evaluation/`: aggregate summaries, raw predictions, and split-level metrics.
- `compositions/`: composition specifications, registries, and qualification reports.
- `data/datasets/`: packaged datasets from retained configurations.
- `data/results/`: packaged evaluation outputs and predictions from retained configurations.
- `data/analysis/`: packaged cross-configuration analysis and adapter-feature tables.

Every composition run saves base, constituent, dedicated, dynamic, atomic-retention, endpoint, determinism, mutation, parsing, and prediction diagnostics, allowing metrics to be recomputed without rerunning training.

## Offline aggregation

```bash
python utils/aggregate_results.py --clean
```

The script can be rerun after new result archives are added. It writes the inclusion manifest, split-level performance tables, retention and interference summaries, task features, adapter tensor features, pair geometry, module geometry, and geometry/outcome joins.

## Research scope

The benchmark distinguishes capability coexistence from functional composition. Dedicated composite adapters test whether a target is learnable; atomic prompts measure retention; composite prompts measure the unseen ordered transformation; and saved adapter features support analysis of interference and update geometry. Additive and CAT are static parameter-space merges. Explicit routing or sequential execution test different hypotheses.
