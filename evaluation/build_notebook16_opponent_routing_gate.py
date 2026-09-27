"""Build notebook 16 from the completed E4 opponent-routing experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_SUMMARY = ROOT / "results" / "e4_opponent_routing_gate" / "summary.json"
EXTENSION_SUMMARY = (
    ROOT / "results" / "e4_opponent_routing_gate_extension" / "summary.json"
)
TARGET = ROOT / "notebooks" / "16_e4_opponent_routing_gate.ipynb"
VARIANTS = ("opponent_route_off", "fallback_final", "strict_hunt_only")
CONTRASTS = (
    "fallback_final-opponent_route_off",
    "strict_hunt_only-fallback_final",
    "strict_hunt_only-opponent_route_off",
)


def fmt(value, digits=4):
    if isinstance(value, float):
        return "NA" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def table(rows, columns, digits=4):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(fmt(row[column], digits) for column in columns) + " |"
        )
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
    if math.isnan(row["mean_difference"]):
        return "NA (no exposed paired seeds)"
    return (
        f"{row['mean_difference']:+.3f} "
        f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
    )


def interval_label(row):
    if math.isnan(row["mean_difference"]):
        return "not estimable"
    if row["ci_low"] > 0:
        return "positive"
    if row["ci_high"] < 0:
        return "negative"
    return "inconclusive"


def interpretation(summary):
    direct = "fallback_final-opponent_route_off"
    strictness = "strict_hunt_only-fallback_final"
    terminal = "strict_hunt_only-opponent_route_off"
    lines = []
    for lineup in summary["protocol"]["lineups"]:
        score = comparison(summary, lineup, direct, "score")
        kills = comparison(summary, lineup, direct, "kills")
        coins = comparison(summary, lineup, direct, "coins")
        crates = comparison(summary, lineup, direct, "crates")
        suicides = comparison(summary, lineup, direct, "suicides")
        traps = comparison(summary, lineup, direct, "trap_pressure_bombs")
        progress = comparison(summary, lineup, direct, "opponent_progress")
        lines.append(
            f"- **{lineup}, fallback − off.** Score {estimate(score)} "
            f"({interval_label(score)}); kills {estimate(kills)}; coins "
            f"{estimate(coins)}; crates {estimate(crates)}; suicides "
            f"{estimate(suicides)}; non-zero trap-pressure bombs "
            f"{estimate(traps)}; selected-action opponent progress "
            f"{estimate(progress)}."
        )

        strict_score = comparison(summary, lineup, strictness, "score")
        strict_coins = comparison(summary, lineup, strictness, "coins")
        strict_kills = comparison(summary, lineup, strictness, "kills")
        terminal_score = comparison(summary, lineup, terminal, "score")
        lines.append(
            f"- **{lineup}, strict − fallback.** Score {estimate(strict_score)}, "
            f"coins {estimate(strict_coins)}, and kills {estimate(strict_kills)}. "
            f"The pure terminal-guidance contrast (strict − off) gives score "
            f"{estimate(terminal_score)}."
        )

    total_opportunities = sum(
        row["opponent_route_opportunities"] * row["rounds"]
        for row in summary["aggregate"] if row["variant"] == "fallback_final"
    )
    total_with_coin = sum(
        row["opponent_routes_with_coin"] * row["rounds"]
        for row in summary["aggregate"] if row["variant"] == "fallback_final"
    )
    total_hunt_rounds = sum(
        row["hunt_round"] * row["rounds"]
        for row in summary["aggregate"] if row["variant"] == "fallback_final"
    )
    strict_identical = all(
        comparison(summary, lineup, strictness, metric)["mean_difference"] == 0.0
        for lineup in summary["protocol"]["lineups"]
        for metric in ("score", "coins", "crates", "kills", "suicides", "invalid")
    )
    exposure = (
        f"Across the {len(summary['protocol']['lineups'])} fallback lineup arms "
        f"there were {total_opportunities:.0f} "
        f"opponent-route opportunities, {total_with_coin:.0f} while at least one "
        f"coin remained, and {total_hunt_rounds:.0f} round-level hunt exposures."
    )
    if strict_identical:
        gate = (
            " Strict and fallback produced identical primary, economic, offensive, "
            "and safety outcomes in both lineups. This means the strictness contrast "
            "was not behaviorally activated often enough to estimate a protection "
            "benefit; it is evidence about reachability/exposure, not proof that the "
            "two gate definitions are universally equivalent."
        )
    else:
        gate = (
            " Strict and fallback diverged on at least one outcome, so the strictness "
            "contrast is behaviorally identified in this benchmark."
        )
    return "\n".join(lines) + "\n\n" + exposure + gate


def main():
    main_summary = json.loads(MAIN_SUMMARY.read_text(encoding="utf-8"))
    extension_summary = json.loads(EXTENSION_SUMMARY.read_text(encoding="utf-8"))
    summary = dict(main_summary)
    summary["protocol"] = dict(main_summary["protocol"])
    summary["protocol"]["lineups"] = {
        **main_summary["protocol"]["lineups"],
        **extension_summary["protocol"]["lineups"],
    }
    summary["protocol"]["total_rounds"] = (
        main_summary["protocol"]["total_rounds"]
        + extension_summary["protocol"]["total_rounds"]
    )
    for key in ("ledger", "round_rows", "seed_rows", "aggregate", "paired_bootstrap"):
        summary[key] = main_summary[key] + extension_summary[key]
    protocol = summary["protocol"]

    aggregate_columns = [
        "lineup", "variant", "rounds", "score", "coins", "crates", "kills",
        "suicides", "survived", "invalid", "opponent_route_opportunity_rate",
        "opponent_route_enabled_rate", "opponent_target_selection_rate",
        "opponent_progress", "trap_pressure_bombs",
        "nonzero_trap_pressure_bomb_rate", "bomb_opponent_yield",
        "coin_routing_activation_rate", "opponent_with_coin_rate", "hunt_steps",
        "hunt_round", "self_trapped_after_bomb", "decision_ms_p95",
    ]
    aggregate_table = table(summary["aggregate"], aggregate_columns)

    key_metrics = {
        "score", "kills", "coins", "crates", "suicides", "survived", "invalid",
        "opponent_target_selections", "opponent_progress", "trap_pressure_bombs",
        "bomb_opponent_yield", "coin_routing_activations",
        "opponent_routes_with_coin", "hunt_steps", "hunt_round",
        "self_trapped_after_bomb",
    }
    inference = [
        row for row in summary["paired_bootstrap"] if row["metric"] in key_metrics
    ]
    inference_table = table(
        inference,
        [
            "lineup", "contrast", "metric", "mean_difference", "ci_low", "ci_high",
            "positive_seeds", "zero_seeds", "negative_seeds", "paired_seeds",
        ],
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
                [
                    x["sha256"] for x in
                    manifests[(lineup, variant, seed)]["initial_conditions"]
                ]
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

    # This gate-specific exposure is intentionally reconstructed from immutable
    # decision traces: the runner summary stores counterfactual opportunities,
    # whereas E4 also needs the number actually retained after the gate.
    coin_exposure = []
    for lineup in protocol["lineups"]:
        for variant in VARIANTS:
            decisions = opportunities = enabled_with_coin = opportunity_with_coin = 0
            for seed in protocol["seeds"]:
                trace_path = ROOT / manifests[(lineup, variant, seed)]["trace"]
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                for row in trace["decisions"]:
                    decisions += 1
                    if row["opponent_route_opportunity"]:
                        opportunities += 1
                        if row["coins_present"]:
                            opportunity_with_coin += 1
                            if row["opponent_route_enabled"]:
                                enabled_with_coin += 1
            coin_exposure.append({
                "lineup": lineup,
                "variant": variant,
                "opponent_opportunities": opportunities,
                "opportunities_with_coin": opportunity_with_coin,
                "enabled_routes_with_coin": enabled_with_coin,
                "enabled_with_coin_per_10k_decisions": (
                    10000.0 * enabled_with_coin / max(decisions, 1)
                ),
            })
    coin_exposure_table = table(
        coin_exposure,
        [
            "lineup", "variant", "opponent_opportunities",
            "opportunities_with_coin", "enabled_routes_with_coin",
            "enabled_with_coin_per_10k_decisions",
        ],
    )

    cells = [
        cell("markdown", """# Notebook 16 — E4 opponent-routing gate ablation

**Question.** How should opponent-directed routing be gated relative to remaining economic opportunities?

**Additional framing.** When economic objectives remain or are unreachable, when is it worthwhile to direct safe actions towards an opponent? This experiment tests whether fallback opponent routing improves offensive conversion and whether a strict hunt-only gate better protects remaining economic opportunities."""),
        cell("markdown", f"""## Locked experimental design

- Frozen candidate: `{protocol['candidate']}`; checkpoint `{protocol['checkpoint_sha256']}`
- Conditions: `opponent_route_off`, `fallback_final`, `strict_hunt_only`
- Main lineups: `external_mixed` and `frozen_dddqn`
- Added robustness lineups: `rule3` and `internal_mixed`
- Fresh seeds: `{protocol['seeds'][0]}–{protocol['seeds'][-1]}`; {protocol['rounds_per_seed']} rounds per seed
- Workload: {protocol['total_rounds']:,} rounds; no training and zero exploration
- Primary metric: mean score per round
- Prespecified contrasts: fallback − off, strict − fallback, strict − off
- Inference: average five rounds within seed, then paired 10,000-replicate seed-cluster bootstrap

The manipulation is source-selective: only an `objective_target` produced by the `opponent` source is cleared, before both network inference and the +1.0 direct Q bonus. The 6-step safety layer, hybrid bomb rule, economic/escape routes, +0.04 centre bias, 0.60 revisit penalty, architecture, weights, and learned opponent features are fixed."""),
        cell("code", """from pathlib import Path
import json, pandas as pd
ROOT = Path('..').resolve()
main_payload = json.loads((ROOT/'results/e4_opponent_routing_gate/summary.json').read_text())
extension_payload = json.loads((ROOT/'results/e4_opponent_routing_gate_extension/summary.json').read_text())
aggregate = pd.DataFrame(main_payload['aggregate'] + extension_payload['aggregate'])
paired = pd.DataFrame(main_payload['paired_bootstrap'] + extension_payload['paired_bootstrap'])
aggregate"""),
        cell("markdown", "## Completeness and matched initial conditions\n\n" + pairing_table),
        cell("markdown", """## Metric definitions

- `opponent_route_opportunity_rate`: counterfactual opponent objectives divided by all candidate decisions. It is measured before the arm-specific gate.
- `opponent_route_enabled_rate`: enabled opponent objectives divided by all decisions.
- `opponent_target_selection_rate`: among opponent-route opportunities, fraction whose ultimately selected action was in the counterfactual target set.
- `opponent_progress`: chosen-action BFS progress toward an opponent-adjacent tile, conditional on an opponent-route opportunity.
- `trap_pressure_bombs`: selected bombs with positive simulated reduction in opponents' safe reachable positions.
- `nonzero_trap_pressure_bomb_rate`: those bombs divided by all selected bombs.
- `bomb_opponent_yield`: summed normalized direct opponent blast coverage per round.
- `opponent_with_coin_rate`: opponent-route opportunities occurring while at least one coin remained, divided by all opponent-route opportunities.
- `hunt_round`: fraction of rounds containing at least one true hunt-phase step.
- `self_trapped_after_bomb`: placed candidate bombs after which no viable safe continuation remained while that bomb was active."""),
        cell("markdown", "## Aggregate results\n\nAll count metrics are means per round.\n\n" + aggregate_table),
        cell("markdown", """## Strict-gate economic exposure

`opportunities_with_coin` is measured before the arm-specific gate; `enabled_routes_with_coin` is measured after it. This distinction is essential: the strict arm can observe the same counterfactual opponent objective while correctly suppressing it.

""" + coin_exposure_table),
        cell("markdown", "## Prespecified paired contrasts\n\nValues are mean per-round differences with percentile 95% paired seed-cluster intervals.\n\n" + inference_table),
        cell("markdown", "## Result interpretation\n\n" + interpretation(summary)),
        cell("markdown", """## Conclusion and implementation decision

The direct opponent route is **mechanistically effective in all four pools and outcome-effective in one**. Relative to `opponent_route_off`, fallback routing strongly increases selection of opponent-directed actions, chosen-action BFS progress, positive trap-pressure bombs, and direct bomb coverage everywhere. In the added `internal_mixed` robustness pool, both score (+0.250, 95% CI +0.050 to +0.475) and kills (+0.050, +0.010 to +0.095) improve clearly. `rule3` is directionally supportive but its score interval starts at zero; the two main-lineup score and kill intervals cross zero. Thus the offensive benefit is opponent-pool dependent rather than universal.

Coins and crates are exactly unchanged in every direct comparison. Safety also improves in several pools: suicides and post-bomb self-traps fall clearly in `frozen_dddqn`, `rule3`, and `internal_mixed`, while survival rises clearly in the first two of those. Against every pool, however, opponent routing significantly increases invalid actions—most likely from more close-contact movement attempts—which is the consistent adverse effect requiring follow-up.

The strict-vs-fallback distinction remains extremely sparse: only 9 coin-present opponent objectives were enabled by fallback across all 800 fallback rounds (3 each in `external_mixed`, `rule3`, and `internal_mixed`, none in `frozen_dddqn`); strict suppressed all nine. No lineup shows a score, coin, crate, or kill difference between strict and fallback. Therefore E4 does **not** establish an economic-conversion benefit for strict gating. It does show that `strict_hunt_only` preserves every observed primary/offensive outcome—including the clear `internal_mixed` gain—while removing the rare semantically ambiguous coin-present fallback at no observed economic cost.

**Decision:** prefer `strict_hunt_only` as the conservative specification for the paper and final inference code, while describing its advantage as semantic/economic-risk control rather than a demonstrated score improvement. Retain the learned opponent features and direct route, but describe the latter as a context-dependent optimization supported by one clear robustness result and consistent mechanism effects—not as a universal score gain. Investigate the invalid-action increase before strengthening the +1.0 routing bias."""),
        cell("markdown", """## Validity checks and limitations

- All three arms use identical initial-condition hashes within every lineup/seed/round block.
- Every raw stats file and decision trace is hash-addressed by its run manifest; the summary validates hashes and frozen agent inputs before analysis.
- Conditional mechanism means can have fewer informative seed pairs than outcome metrics. `paired_seeds` is therefore reported explicitly.
- Kills are sparse and opponent-route opportunities depend on trajectories. A confidence interval crossing zero is not evidence of equivalence.
- The experiment estimates inference-time routing effects for this checkpoint and these opponent pools; it does not estimate how retraining without the routing feature would change the learned policy."""),
        cell("markdown", f"""## Reproduction

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e4_opponent_routing_gate.py status
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e4_opponent_routing_gate.py summarize
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e4_opponent_routing_gate_extension.py status
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e4_opponent_routing_gate_extension.py summarize
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/build_notebook16_opponent_routing_gate.py
```

Protocol SHA-256 is recorded in every run manifest. Main runner implementation SHA-256 locked before execution: `{protocol['implementation_sha256']}`. Extension wrapper SHA-256: `{extension_summary['protocol']['extension_wrapper_sha256']}`."""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
