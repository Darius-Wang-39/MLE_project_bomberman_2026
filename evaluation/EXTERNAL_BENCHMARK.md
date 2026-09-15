# External previous-cohort benchmark

External agents are evaluation-only and use names such as
`external_double_q`. Their local adapter directories contain file-level symbolic
links to the sibling `external_previous_cohort_benchmark` checkout. Generated
logs and bytecode stay local, and `train.py` is deliberately not linked.
The three legacy neural-network pickles also receive minimal local package-name
aliases (`dqn`, `dqn_5tile`, and `feature_is_everything`) because pickle embeds
their old import paths. These aliases link only class-definition modules and are
not used as benchmark agent names.

Create or refresh the adapters:

```bash
bash tools/setup_external_benchmark.sh
```

Run the gated benchmark stages in order:

```bash
python evaluation/run_external_benchmark.py smoke
python evaluation/run_external_benchmark.py screen
python evaluation/run_external_benchmark.py confirm
```

Use `--resume` with `screen` or `confirm` to reuse completed raw result files.
All reports and raw statistics are kept under `results/external_benchmark/`, so
they are not mixed with internal candidate-selection records.

The expected upstream list includes `baseline`, but the current external clone
does not contain `agent_code/baseline`; the smoke report therefore records it as
incompatible/missing rather than substituting another agent.
