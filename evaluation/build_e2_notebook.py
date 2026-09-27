"""Build notebook 13 from the locked E2 protocol and completed summary."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "e2_bomb_escape_branches" / "summary_threeway.json"
TARGET = ROOT / "notebooks" / "13_e2_bomb_escape_branches.ipynb"
VARIANTS = ("two_branch_final", "original_hybrid", "one_branch_relaxed")
CONTRASTS = (
    ("two_branch_final", "one_branch_relaxed"),
    ("two_branch_final", "original_hybrid"),
    ("original_hybrid", "one_branch_relaxed"),
)


def markdown_table(rows, columns, labels=None, digits=4):
    labels = labels or columns
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.{digits}f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def cell(cell_type, source):
    payload = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(True)}
    if cell_type == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def conclusion(summary):
    comparisons = summary["paired_bootstrap"]

    def result(lineup, contrast, metric):
        return next(
            row
            for row in comparisons
            if row["lineup"] == lineup
            and row["contrast"] == contrast
            and row["metric"] == metric
        )

    exposure = sum(row["bomb_proposals_total"] for row in summary["aggregate"])
    traps = sum(row["self_trapped_total"] for row in summary["aggregate"])
    if exposure < 20 or traps == 0:
        return (
            "**Conclusion: exposure is insufficient, so the absence of a detected "
            "difference cannot be interpreted as proof that the rule is safe.** "
            f"The experiment recorded {exposure} BOMB proposals and {traps} actual "
            "self-traps. Report the estimates and intervals, but keep the safety "
            "conclusion qualified."
        )
    hybrid_beats_strict_score = all(
        result(lineup, "two_branch_final-original_hybrid", "score")["ci_high"] < 0
        for lineup in summary["protocol"]["core_lineups"]
    )
    hybrid_vs_relaxed = [
        result(lineup, "original_hybrid-one_branch_relaxed", "score")
        for lineup in summary["protocol"]["core_lineups"]
    ]
    if hybrid_beats_strict_score:
        relaxed_score_clear = all(row["ci_high"] < 0 for row in hybrid_vs_relaxed)
        hybrid_safety_clear = all(
            result(
                lineup,
                "original_hybrid-one_branch_relaxed",
                "self_trapped_after_bomb",
            )["ci_high"] < 0
            for lineup in summary["protocol"]["core_lineups"]
        )
        if relaxed_score_clear and hybrid_safety_clear:
            relaxed_relation = (
                "The fully relaxed arm has clearly higher mean score, bombs, crates, "
                "and coins in both lineups, whereas the hybrid clearly reduces actual "
                "self-traps per round in both. Total suicides and survival do not show "
                "an equally clear hybrid advantage. The two arms therefore present a "
                "genuine productivity–direct-safety trade-off: choose relaxed if the "
                "official mean score is the priority; choose hybrid as a defensible "
                "compromise if a limited productivity cost is acceptable in exchange "
                "for fewer direct self-traps."
            )
        else:
            relaxed_relation = (
                "The hybrid and fully relaxed arms do not have a single clear ordering "
                "across lineups; assess both score and direct safety metrics."
            )
        return (
            "**Conclusion: the original hybrid rule clearly repairs the main "
            "productivity collapse caused by strict two-branch filtering.** The strict "
            "arm's additional safety benefit over the hybrid is not measurable in "
            "`external_mixed`; although present against `frozen_dddqn`, it comes with "
            "very large score and economic losses, so strict two-branch filtering "
            "should not be the final rule. "
            + relaxed_relation
        )
    return (
        "**Conclusion: the evidence does not establish a single ordering of the three "
        "rules across lineups.** Report every pairwise bootstrap confidence interval "
        "and qualify the conclusion as opponent-pool dependent."
    )


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    aggregate_rows = []
    for row in summary["aggregate"]:
        enriched = dict(row)
        enriched["self_traps_per_100_bombs"] = (
            100.0 * row["self_trapped_total"] / row["actual_bombs_total"]
        )
        aggregate_rows.append(enriched)
    aggregate_columns = [
        "lineup", "variant", "rounds", "score", "bombs", "crates", "coins",
        "suicides", "survived", "self_trapped_total", "bomb_proposals_total",
        "bomb_rejections_total", "actual_bombs_total", "self_traps_per_100_bombs",
        "decision_ms_p95",
    ]
    aggregate_table = markdown_table(aggregate_rows, aggregate_columns)
    inference_metrics = {
        "score", "bombs", "crates", "coins", "suicides", "survived",
        "self_trapped_after_bomb", "bomb_proposals", "bomb_rejections",
    }
    inference_rows = [
        row for row in summary["paired_bootstrap"] if row["metric"] in inference_metrics
    ]
    inference_table = markdown_table(
        inference_rows,
        [
            "lineup", "contrast", "metric", "mean_difference", "ci_low", "ci_high",
            "positive_seeds", "zero_seeds", "negative_seeds",
        ],
    )
    rate_rows = []
    bootstrap_rng = random.Random(protocol["bootstrap"]["rng_seed"])
    for lineup in protocol["lineups"]:
        for left_name, right_name in CONTRASTS:
            differences = []
            for seed in protocol["seeds"]:
                arm = {}
                for variant in (left_name, right_name):
                    row = next(
                        item for item in summary["seed_rows"]
                        if item["lineup"] == lineup
                        and item["variant"] == variant
                        and item["seed"] == seed
                    )
                    arm[variant] = 100.0 * row["self_trapped_after_bomb"] / row["actual_bombs"]
                differences.append(arm[left_name] - arm[right_name])
            bootstrap = sorted(
                statistics.mean(bootstrap_rng.choices(differences, k=len(differences)))
                for _ in range(protocol["bootstrap"]["replicates"])
            )
            rate_rows.append(
                {
                    "lineup": lineup,
                    "contrast": f"{left_name}-{right_name}",
                    "metric": "self_traps_per_100_actual_bombs",
                    "mean_difference": statistics.mean(differences),
                    "ci_low": bootstrap[249],
                    "ci_high": bootstrap[9749],
                    "positive_seeds": sum(value > 0 for value in differences),
                    "zero_seeds": sum(value == 0 for value in differences),
                    "negative_seeds": sum(value < 0 for value in differences),
                }
            )
    rate_table = markdown_table(
        rate_rows,
        [
            "lineup", "contrast", "metric", "mean_difference", "ci_low", "ci_high",
            "positive_seeds", "zero_seeds", "negative_seeds",
        ],
    )
    initial_check = []
    manifests = {}
    for item in summary["ledger"]:
        path = ROOT / item["manifest"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifests[(item["lineup"], item["variant"], item["seed"])] = manifest
    for lineup in protocol["lineups"]:
        matched = 0
        for seed in protocol["seeds"]:
            hashes = [
                [row["sha256"] for row in manifests[(lineup, variant, seed)]["initial_conditions"]]
                for variant in VARIANTS
            ]
            matched += sum(len({arm[index] for arm in hashes}) == 1 for index in range(5))
        initial_check.append({"lineup": lineup, "matched": matched, "expected": 200})
    initial_table = markdown_table(initial_check, ["lineup", "matched", "expected"])

    cells = [
        cell(
            "markdown",
            """# Experiment 2 — Active bomb-placement escape redundancy (`e2`)

**Research question.** Does requiring two independent first-step bomb-escape branches reduce catastrophic self-traps enough to justify potentially fewer productive bombs?

**Additional framing.** Before placing a bomb, does requiring at least two independent first-step escape branches substantially reduce the risk of becoming trapped after placement, and is that safety benefit worth the potential loss in bombs, crates, or total score?

This is not an experiment on the passive one-step lethality mask. The shared immediate-lethality filter runs first. The three arms require at least two branches for every bomb, use the original hybrid rule, or permit every one-branch bomb, respectively.""",
        ),
        cell(
            "markdown",
            f"""## Locked protocol

- Candidate/checkpoint: `{protocol['candidate']}` / `{protocol['checkpoint_sha256']}`
- Conditions: `two_branch_final` (≥2), `original_hybrid` (original quiet-opening exception), `one_branch_relaxed` (≥1)
- Core lineups: {', '.join(f'`{x}`' for x in protocol['core_lineups'])}
- Fresh seeds: `{protocol['seeds'][0]}–{protocol['seeds'][-1]}`; 5 rounds per seed
- Paired unit: seed cluster after averaging its five rounds
- Total: {protocol['total_rounds']} rounds; zero exploration; 10,000 paired bootstrap resamples for every pair
- Locked: checkpoint, danger model, objective routing, hunt-centre bias λ=.04, opponent lineup, engine files and code hashes

Implementation note: the repository feature extractor's quiet-opening one-branch exception is now a third, explicitly measured arm. The original two arms remain unchanged append-only records. All three leave the checkpoint, feature values, adjusted Q-values, passive survival filter, round seeding and opponents unchanged at a shared state.

Raw stats, decision traces, bomb-event traces, manifests, SHA-256 ledger and seed-level records are under `results/e2_bomb_escape_branches/`.""",
        ),
        cell(
            "code",
            """from pathlib import Path
import json
import pandas as pd

ROOT = Path('..').resolve()
payload = json.loads((ROOT / 'results/e2_bomb_escape_branches/summary_threeway.json').read_text())
aggregate = pd.DataFrame(payload['aggregate'])
seed_level = pd.DataFrame(payload['seed_rows'])
paired = pd.DataFrame(payload['paired_bootstrap'])""",
        ),
        cell(
            "markdown",
            """## Pairing and completeness

Every condition has 40 complete seed clusters × 5 rounds. The table counts exact initial-condition hash matches across all three arms; 200/200 is required for each lineup.

""" + initial_table,
        ),
        cell(
            "markdown",
            """## Aggregate results and exposure integrity check

`score`, `bombs`, `crates`, `coins`, `suicides`, and `survived` are per round. `decision_ms_p95` pools decision-level timings. Counts ending in `_total` are totals over 200 rounds per row. `self_traps_per_100_bombs` is a descriptive exposure-normalized rate.

""" + aggregate_table,
        ),
        cell(
            "markdown",
            """## Paired seed-cluster bootstrap

All three pairwise differences are reported. For each `left-right` contrast, positive values favor the left arm for score/productivity/survival; negative values favor it for suicides, self-traps, and rejections. Intervals are percentile 95% paired-bootstrap intervals over the 40 seed means.

""" + inference_table + "\n\n### Exposure-normalized safety diagnostic\n\n" + rate_table,
        ),
        cell(
            "markdown",
            """## Operational trace definitions

- `bomb_proposal`: `BOMB` is among the maximum adjusted-Q actions after the shared immediate-lethality filter, before applying branch redundancy.
- `bomb_rejected`: a bomb proposal fails the arm's branch threshold.
- `self_trapped_after_bomb`: after a real bomb placement and before that bomb explodes, at least one observed post-action world state has no legal action with a survivable continuation under the locked danger simulator.
- Different continuations sharing the same first move count as one branch, not multiple branches.

This attribution is narrower than total suicides: an avoidable bad choice despite an available path is not relabelled as a forced self-trap.""",
        ),
        cell("markdown", "## Decision\n\n" + conclusion(summary)),
        cell(
            "markdown",
            """## Reproduction

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e2_bomb_escape_branches.py status
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e2_hybrid_extension.py status
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e2_hybrid_extension.py summarize
```

The locked protocol and every run manifest contain hashes. Completed raw runs are append-only; the summarizer refuses missing clusters, protocol mismatches, or changed result hashes.""",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
