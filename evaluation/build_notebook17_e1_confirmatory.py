"""Build notebook 17 from the locked confirmatory E1 summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "e1_confirmatory_phase_gating"
TARGET = ROOT / "notebooks" / "17_e1_confirmatory_phase_gating.ipynb"


def lines(text: str) -> list[str]:
    parts = text.strip("\n").splitlines()
    return [line + "\n" for line in parts[:-1]] + ([parts[-1]] if parts else [])


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": lines(text),
    }


def row(summary: dict, lineup: str, contrast: str, metric: str) -> dict:
    return next(
        item for item in summary["paired_bootstrap"]
        if item["lineup"] == lineup
        and item["contrast"] == contrast
        and item["metric"] == metric
    )


def interval(item: dict) -> str:
    return (
        f"{item['mean_difference']:+.3f} "
        f"(95% CI {item['ci_low']:+.3f} to {item['ci_high']:+.3f})"
    )


def result_text(summary: dict) -> str:
    primary = "center_hunt_04-center_ungated_04"
    secondary = "center_hunt_04-center_off"
    lineups = list(summary["protocol"]["lineups"])
    score_primary = "; ".join(
        f"`{name}` {interval(row(summary, name, primary, 'score'))}"
        for name in lineups
    )
    score_secondary = "; ".join(
        f"`{name}` {interval(row(summary, name, secondary, 'score'))}"
        for name in lineups
    )
    frozen_crates = row(summary, "frozen_dddqn", primary, "crates")
    aggregates = {
        (item["lineup"], item["variant"]): item for item in summary["aggregate"]
    }
    ungated = [aggregates[(name, "center_ungated_04")] for name in lineups]
    changed_range = (
        100 * min(item["center_changed_action_rate"] for item in ungated),
        100 * max(item["center_changed_action_rate"] for item in ungated),
    )
    intervention_range = (
        min(item["economic_phase_center_interventions"] for item in ungated),
        max(item["economic_phase_center_interventions"] for item in ungated),
    )
    override_range = (
        min(item["economic_route_overrides"] for item in ungated),
        max(item["economic_route_overrides"] for item in ungated),
    )
    invalid_interventions = sum(
        item["intervention_invalid_actions"] * item["rounds"] for item in ungated
    )
    clear_primary = [
        name for name in lineups
        if row(summary, name, primary, "score")["ci_low"] > 0
        or row(summary, name, primary, "score")["ci_high"] < 0
    ]
    stable = bool(clear_primary) and all(
        row(summary, name, primary, "score")["mean_difference"] > 0
        for name in lineups
    )
    if stable:
        answer = (
            "The result supports hunt-phase gating on score across the tested pools, "
            "subject to the reported uncertainty and explanatory trade-offs."
        )
    else:
        answer = (
            "The preregistered evidence does not establish a universal score advantage "
            "for hunt-only gating. The centre prior is therefore interpreted as a "
            "conservative priority rule, not a generally performance-improving optimisation."
        )
    return f"""
## Confirmatory result

For the primary hunt-only minus ungated contrast, mean-score effects were {score_primary}.

For the secondary hunt-only minus centre-off contrast, effects were {score_secondary}.

{answer} The mechanism tables below distinguish a prior that rarely changes actions from one that intervenes in economic routing with pool-dependent gains and costs. Coins, crates, kills, suicides, invalid actions, and survival remain prespecified explanatory outcomes rather than additional primary endpoints.

The clearest explanatory difference in the primary contrast is in `frozen_dddqn`: hunt-only collected {abs(frozen_crates['mean_difference']):.3f} fewer crates/round than ungated ({frozen_crates['ci_low']:+.3f} to {frozen_crates['ci_high']:+.3f}). Other primary score and safety intervals include zero. Ungated changed only {changed_range[0]:.2f}%–{changed_range[1]:.2f}% of all decisions, corresponding to {intervention_range[0]:.3f}–{intervention_range[1]:.3f} economic-phase interventions per round. Only {override_range[0]:.3f}–{override_range[1]:.3f} per round actually overrode the active coin/crate route, and {invalid_interventions:.0f} logged intervention decisions directly produced an invalid action.

Relative to centre-off, hunt-only reduces hunt centre distance clearly in `frozen_dddqn` and `rule3`, with corresponding clear occupancy increases, but the changes are small and do not yield a stable cross-pool score effect. Overall, the data fit the narrow interpretation that $.04$ controllably modifies some hunt positioning while its score and economy effects remain small or opponent-pool dependent.
"""


def build() -> None:
    summary = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    cells = [
        markdown(r"""
# Experiment 1 — Fresh-seed confirmation of centre phase gating

**Question.** Is the submitted hunt-only weak centre prior ($\lambda=.04$) preferable to applying the identical prior throughout non-emergency play, and does it have an observable effect relative to no centre prior?

**Confirmation scope.** This notebook formally re-evaluates `.04` phase gating with entirely fresh seeds. Notebooks 11 and 12 remain the development calibration and mechanism check; this experiment neither searches `.08/.12` nor reuses `65001–65040`.
"""),
        markdown(r"""
## Locked design

| Variant | Locked definition |
|---|---|
| `center_off` | $\lambda=0$ |
| `center_hunt_04` | $\lambda=.04$, enabled only during the hunt phase |
| `center_ungated_04` | Retains the hunt-phase term and additionally applies the same $\lambda=.04$ outside the hunt phase in non-emergency states |

- Lineups: `external_mixed`, `frozen_dddqn`, `rule3`, `internal_mixed`.
- Fresh paired seeds: `69001–69040`; five rounds per lineup/variant/seed cell.
- Total: $4\times3\times40\times5=2{,}400$ rounds.
- Primary contrast: `center_hunt_04 − center_ungated_04`.
- Secondary contrast: `center_hunt_04 − center_off`.
- Primary outcome: mean score. Explanatory outcomes: coins, crates, kills, suicides, invalid actions, survival.
- Inference: average five rounds within seed, then a 10,000-replicate paired seed-cluster bootstrap.
"""),
        markdown(r"""
## Mechanism definitions

`center_activation_rate` requires a nonzero lambda, an open phase gate, and at least one safe action with nonzero centre progress. `center_changed_action_rate` compares the actual action with a zero-centre counterfactual selected from the same Q values and identical RNG state. An economic intervention is such a change while coins or crates remain. Route overrides, actual invalid events, the network-only Q margin, and the adjusted pre-centre Q margin are logged for those interventions.
"""),
        code(r"""
from __future__ import annotations
import hashlib, json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
RESULT_ROOT = ROOT / 'results' / 'e1_confirmatory_phase_gating'
payload = json.loads((RESULT_ROOT / 'summary.json').read_text())
protocol = payload['protocol']
assert protocol['total_rounds'] == 2400
assert len(payload['round_rows']) == 2400
assert len(payload['seed_rows']) == 480
assert len(payload['ledger']) == 480
{'rounds': len(payload['round_rows']), 'seed_rows': len(payload['seed_rows']),
 'ledger_entries': len(payload['ledger'])}
"""),
        markdown("## Aggregate outcomes\n\nEach row represents 40 seed clusters / 200 rounds. Rates are decision-weighted across all traces."),
        code(r"""
aggregate = pd.DataFrame(payload['aggregate'])
outcomes = ['lineup','variant','score','coins','crates','kills','suicides','invalid','survived']
display(aggregate[outcomes].round(3))
"""),
        markdown("## Preregistered paired inference\n\nPositive values are hunt-only minus comparator. For adverse outcomes such as suicides and invalid actions, negative values favour hunt-only."),
        code(r"""
bootstrap = pd.DataFrame(payload['paired_bootstrap'])
main_metrics = ['score','coins','crates','kills','suicides','invalid','survived']
display(bootstrap[bootstrap.metric.isin(main_metrics)][
    ['lineup','contrast','metric','mean_difference','ci_low','ci_high',
     'positive_seeds','zero_seeds','negative_seeds','paired_seeds']
].round(3))
"""),
        code(r"""
primary = bootstrap[(bootstrap.contrast == 'center_hunt_04-center_ungated_04') &
                    (bootstrap.metric == 'score')].set_index('lineup').loc[
    ['external_mixed','frozen_dddqn','rule3','internal_mixed']]
estimate = primary.mean_difference.to_numpy()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(np.arange(4), estimate,
            yerr=np.vstack([estimate-primary.ci_low.to_numpy(),
                            primary.ci_high.to_numpy()-estimate]),
            fmt='o', capsize=4, color='#2457A7')
ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(np.arange(4), ['external mixed','frozen DDDQN','rule3','internal mixed'], rotation=15)
ax.set_ylabel('mean score difference')
ax.set_title('E1 primary contrast: hunt-only .04 − ungated .04')
ax.grid(axis='y', alpha=.25)
fig.tight_layout()
figure_dir = RESULT_ROOT / 'figures'; figure_dir.mkdir(exist_ok=True)
fig.savefig(figure_dir / 'primary_score_effects.png', dpi=180, bbox_inches='tight')
plt.show()
"""),
        markdown("## Intervention mechanism"),
        code(r"""
mechanism = [
    'lineup','variant','center_activation_rate','center_changed_action_rate',
    'economic_phase_center_interventions','economic_route_overrides',
    'economic_route_override_rate','intervention_invalid_actions',
    'intervention_invalid_rate','intervention_precenter_q_margin',
    'intervention_network_q_margin','hunt_center_distance','hunt_center_occupancy',
]
display(aggregate[mechanism].round(4))
"""),
        markdown(result_text(summary)),
        markdown(r"""
## Integrity audit

Every manifest is bound to the locked protocol and raw/trace hashes. The audit below additionally verifies that all three variants share identical five-round initial-condition hashes within every lineup/seed block.
"""),
        code(r"""
protocol_hash = hashlib.sha256((RESULT_ROOT / 'protocol.json').read_bytes()).hexdigest()
initial = defaultdict(dict)
protocol_hashes = set()
for item in payload['ledger']:
    manifest = json.loads((ROOT / item['manifest']).read_text())
    assert manifest['raw_sha256'] == hashlib.sha256((ROOT / manifest['raw_stats']).read_bytes()).hexdigest()
    assert manifest['trace_sha256'] == hashlib.sha256((ROOT / manifest['trace']).read_bytes()).hexdigest()
    initial[(item['lineup'], item['seed'])][item['variant']] = tuple(
        entry['sha256'] for entry in manifest['initial_conditions'])
    protocol_hashes.add(manifest['protocol_sha256'])
audit = {
    'paired_lineup_seed_blocks': len(initial),
    'all_three_variants_present': all(len(group) == 3 for group in initial.values()),
    'all_initial_conditions_match': all(len(set(group.values())) == 1 for group in initial.values()),
    'one_locked_protocol': protocol_hashes == {protocol_hash},
    'protocol_sha256': protocol_hash,
    'implementation_sha256': protocol['implementation_sha256'],
}
assert all(value for key, value in audit.items() if key not in {'protocol_sha256','implementation_sha256'})
audit
"""),
        markdown(r"""
## Reproduction entry points

The locked runner is `evaluation/e1_confirmatory_phase_gating.py`. The immutable protocol, per-cluster manifests, raw statistics, decision traces, seed rows, aggregate rows, and paired-bootstrap estimates are stored under `results/e1_confirmatory_phase_gating/`. Notebook 11 and notebook 12 remain historical development evidence and are not pooled into this confirmatory analysis.
        """),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"e1-confirm-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "ml_essential", "language": "python", "name": "ml_essential"},
            "language_info": {"name": "python", "version": "3.10.20"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    build()
