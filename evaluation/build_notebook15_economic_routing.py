"""Build notebook 15 from the completed E3 economic-routing experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "e3_economic_routing" / "summary.json"
EXTENSION_SUMMARY = ROOT / "results" / "e3_economic_routing_extension" / "summary.json"
TARGET = ROOT / "notebooks" / "15_e3_economic_routing.ipynb"
VARIANTS = ("no_economic_route", "coin_only", "full_economic_route")


def table(rows, columns, digits=4):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.{digits}f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def cell(cell_type, source):
    value = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(True)}
    if cell_type == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def comparison(summary, lineup, contrast, metric):
    return next(
        row for row in summary["paired_bootstrap"]
        if row["lineup"] == lineup
        and row["contrast"] == contrast
        and row["metric"] == metric
    )


def estimate(row):
    return f"{row['mean_difference']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"


def interpretation(summary):
    full = "full_economic_route-no_economic_route"
    crate = "full_economic_route-coin_only"
    lines = []
    for lineup in summary["protocol"]["lineups"]:
        score = comparison(summary, lineup, full, "score")
        coins = comparison(summary, lineup, full, "coins")
        crates = comparison(summary, lineup, full, "crates")
        kills = comparison(summary, lineup, full, "kills")
        suicides = comparison(summary, lineup, full, "suicides")
        trapped = comparison(summary, lineup, full, "self_trapped_after_bomb")
        crate_score = comparison(summary, lineup, crate, "score")
        crate_crates = comparison(summary, lineup, crate, "crates")
        lines.append(
            f"- **{lineup}.** Full − no-route: score {estimate(score)}, coins "
            f"{estimate(coins)}, crates {estimate(crates)}, kills {estimate(kills)}, "
            f"suicides {estimate(suicides)}, and post-bomb self-traps {estimate(trapped)}. "
            f"The incremental crate-routing comparison (full − coin-only) gives score "
            f"{estimate(crate_score)} and crates {estimate(crate_crates)}."
        )

    clear_score = []
    for lineup in summary["protocol"]["lineups"]:
        row = comparison(summary, lineup, full, "score")
        if row["ci_low"] > 0:
            clear_score.append(f"a positive effect in `{lineup}`")
        elif row["ci_high"] < 0:
            clear_score.append(f"a negative effect in `{lineup}`")
    conclusion = (
        "The paired intervals establish " + " and ".join(clear_score) + "."
        if clear_score else
        "Neither lineup has a full-versus-no-route score interval excluding zero; the "
        "resource and trade-off estimates should therefore be read as mechanism evidence, "
        "not as proof of a total-score improvement."
    )
    external_trap = comparison(
        summary, "external_mixed", full, "self_trapped_after_bomb"
    )
    external_invalid = comparison(summary, "external_mixed", full, "invalid")
    frozen_trap = comparison(
        summary, "frozen_dddqn", full, "self_trapped_after_bomb"
    )
    rule_trap = comparison(summary, "rule3", full, "self_trapped_after_bomb")
    internal_trap = comparison(
        summary, "internal_mixed", full, "self_trapped_after_bomb"
    )
    internal_kills = comparison(summary, "internal_mixed", full, "kills")
    internal_crate_score = comparison(summary, "internal_mixed", crate, "score")
    max_p95 = max(row["decision_ms_p95"] for row in summary["aggregate"])
    crate_bombs = [
        comparison(summary, lineup, crate, "safe_productive_bombs")
        for lineup in summary["protocol"]["lineups"]
    ]
    mechanism = (
        "The crate component is strongly operative: full − coin-only adds "
        + " and ".join(
            f"{row['mean_difference']:+.3f} safe productive bombs/round in `{lineup}`"
            for lineup, row in zip(summary["protocol"]["lineups"], crate_bombs)
        )
        + ", with every interval excluding zero. The incremental crate-routing score "
        "effect is clear only in `internal_mixed` "
        f"({estimate(internal_crate_score)}); it remains inconclusive in the other three "
        "lineups. In `external_mixed`, full routing reduces post-bomb self-traps "
        f"({estimate(external_trap)}) rather than creating the anticipated safety cost, "
        f"but invalid actions increase ({estimate(external_invalid)}). In `frozen_dddqn`, "
        f"`rule3`, and `internal_mixed`, the self-trap differences are inconclusive "
        f"({estimate(frozen_trap)}, {estimate(rule_trap)}, and {estimate(internal_trap)}). "
        "Suicide and survival intervals cross zero in all four lineups. Kill rate has no "
        "clear change in three lineups, while `internal_mixed` increases by "
        f"{estimate(internal_kills)}. The maximum aggregate decision-time p95 is "
        f"{max_p95:.1f} ms, far inside the 500 ms constraint."
    )
    return "\n".join(lines) + "\n\n" + conclusion + "\n\n" + mechanism


def main():
    main_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    extension_summary = json.loads(EXTENSION_SUMMARY.read_text(encoding="utf-8"))
    summary = dict(main_summary)
    summary["protocol"] = dict(main_summary["protocol"])
    summary["protocol"]["lineups"] = {
        **main_summary["protocol"]["lineups"],
        **extension_summary["protocol"]["lineups"],
    }
    for key in ("ledger", "round_rows", "seed_rows", "aggregate", "paired_bootstrap"):
        summary[key] = main_summary[key] + extension_summary[key]
    protocol = summary["protocol"]
    total_rounds = (
        main_summary["protocol"]["total_rounds"]
        + extension_summary["protocol"]["total_rounds"]
    )
    aggregate_columns = [
        "lineup", "variant", "rounds", "score", "coins", "crates", "bombs",
        "safe_productive_bombs", "economic_activation_rate",
        "economic_target_selection_rate", "coin_distance_reduction",
        "crate_distance_reduction", "kills", "hunt_steps",
        "opponent_activation_rate", "suicides", "invalid", "survived",
        "self_trapped_after_bomb", "decision_ms_p95",
    ]
    aggregate_table = table(summary["aggregate"], aggregate_columns)
    inference_metrics = {
        "score", "coins", "crates", "bombs", "safe_productive_bombs",
        "economic_target_selections", "coin_distance_reduction",
        "crate_distance_reduction", "kills", "hunt_steps", "opponent_activations",
        "suicides", "invalid", "survived", "self_trapped_after_bomb",
        "decision_ms_mean",
    }
    inference = [
        row for row in summary["paired_bootstrap"] if row["metric"] in inference_metrics
    ]
    inference_table = table(
        inference,
        ["lineup", "contrast", "metric", "mean_difference", "ci_low", "ci_high",
         "positive_seeds", "zero_seeds", "negative_seeds", "paired_seeds"],
    )

    manifests = {}
    for row in summary["ledger"]:
        manifests[(row["lineup"], row["variant"], row["seed"])] = json.loads(
            (ROOT / row["manifest"]).read_text(encoding="utf-8")
        )
    pairing = []
    for lineup in protocol["lineups"]:
        matches = 0
        for seed in protocol["seeds"]:
            hashes = [
                [x["sha256"] for x in manifests[(lineup, variant, seed)]["initial_conditions"]]
                for variant in VARIANTS
            ]
            matches += sum(
                len({arm[index] for arm in hashes}) == 1
                for index in range(protocol["rounds_per_seed"])
            )
        pairing.append({
            "lineup": lineup, "matched_rounds": matches,
            "expected": len(protocol["seeds"]) * protocol["rounds_per_seed"],
        })
    pairing_table = table(pairing, ["lineup", "matched_rounds", "expected"])

    cells = [
        cell("markdown", """# Notebook 15 — E3 economic routing

**Question.** Does phase-prioritised economic routing improve resource conversion under the same safety layer, and what does it cost in offensive opportunity or safety?

**Additional framing.** With the same multi-step safety layer held fixed, does phase-based coin–crate routing improve resource conversion, and does that benefit come at the cost of offensive opportunities or safety?"""),
        cell("markdown", f"""## Locked experimental design

- Frozen candidate: `{protocol['candidate']}`; checkpoint `{protocol['checkpoint_sha256']}`
- Conditions: `no_economic_route`, `coin_only`, `full_economic_route`
- Main lineups: `external_mixed`, `frozen_dddqn`; added robustness lineups: `rule3`, `internal_mixed`
- Fresh seeds: `{protocol['seeds'][0]}–{protocol['seeds'][-1]}`; {protocol['rounds_per_seed']} rounds per seed
- Workload: {total_rounds:,} rounds; no training and zero exploration
- Primary metric: mean score per round
- Prespecified contrasts: full − no-route, and full − coin-only
- Inference: average five rounds within each seed, then paired 10,000-replicate seed-cluster bootstrap

The manipulation is source-selective. `objective_target` is cleared before network inference and before the direct +1.0 Q bonus only for the disabled economic sources. Escape and opponent objectives, the safety mask and hybrid bomb rule, the +0.04 hunt-centre bias, revisit penalty, model weights, and architecture are common to all arms."""),
        cell("code", """from pathlib import Path
import json, pandas as pd
ROOT = Path('..').resolve()
payload = json.loads((ROOT/'results/e3_economic_routing/summary.json').read_text())
extension = json.loads((ROOT/'results/e3_economic_routing_extension/summary.json').read_text())
aggregate = pd.DataFrame(payload['aggregate'] + extension['aggregate'])
paired = pd.DataFrame(payload['paired_bootstrap'] + extension['paired_bootstrap'])
aggregate"""),
        cell("markdown", "## Completeness and matched initial conditions\n\n" + pairing_table),
        cell("markdown", """## Metric definitions

- `economic_activation_rate`: counterfactual coin/crate objective activations divided by all candidate decisions.
- `economic_target_selection_rate`: among those activations, the fraction in which the final selected action belongs to the counterfactual economic target set.
- `coin_distance_reduction` / `crate_distance_reduction`: mean chosen-action progress on activated objectives; positive means one BFS step closer.
- `safe_productive_bombs`: actually placed bombs carrying the frozen extractor's safe-and-crate-productive flag.
- `self_trapped_after_bomb`: an observed state before the agent's own bomb explodes with no legal action having a survivable continuation.

Rates and progress values can differ across arms because disabling routing changes later visited states; they are descriptive mechanism outcomes, not fixed exposure covariates."""),
        cell("markdown", "## Aggregate results\n\nAll count outcomes are means per round; `hunt_steps` is therefore steps per round.\n\n" + aggregate_table),
        cell("markdown", "## Prespecified paired comparisons\n\nEvery contrast is `left − right`. Intervals resample the 40 paired seed clusters, not the 200 rounds as independent observations.\n\n" + inference_table),
        cell("markdown", "## Important result analysis\n\n" + interpretation(summary)),
        cell("markdown", """## Interpretation guardrails

1. Full − no-route estimates the total marginal value of direct coin/crate routing while preserving safety and opponent routing.
2. Full − coin-only isolates the incremental crate-routing component, including immediate productive BOMB and movement toward productive bomb tiles.
3. The network still contains its learned weights on `coin_progress`, `crate_progress`, `bomb_crate_yield`, and `safe_productive_bomb` in every arm. This is intentionally an ablation of the inference-time routing layer, not of learned economic representations.
4. A resource increase with flat score indicates policy-priority redistribution; higher crates together with more self-traps or suicides indicates tension with the safety layer; null resource and score effects suggest little marginal value beyond the frozen DDDQN's learned features."""),
        cell("markdown", """## Reproduction and provenance

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e3_economic_routing.py status
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e3_economic_routing.py summarize
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e3_economic_routing_extension.py status
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e3_economic_routing_extension.py summarize
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/build_notebook15_economic_routing.py
```

Raw stats, per-decision objective-source traces, bomb lifecycle traces, manifests, frozen-input hashes, seed rows, and bootstrap results are stored under `results/e3_economic_routing/` and `results/e3_economic_routing_extension/`."""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
