"""Build the executed-source notebook for the E1 phase-gating experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "12_e1_phase_gating.ipynb"


def lines(text: str) -> list[str]:
    chunks = text.strip("\n").splitlines()
    return [line + "\n" for line in chunks[:-1]] + ([chunks[-1]] if chunks else [])


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": lines(text),
    }


cells = [
    markdown(r"""
# Experiment 1 — Phase-gating a centre-directed heuristic

**Research question.** Does phase-gating a centre-directed heuristic preserve economic progress and safety better than applying the same preference throughout play?

**Additional framing.** Does restricting the tactical preference for moving towards the centre to the hunt phase preserve coin and crate progress and safety better than enabling the same preference throughout play, while retaining its hunt-phase positional benefit?
"""),
    markdown(r"""
## Relation to the earlier calibration

Notebook 11 remains the 2,000-round calibration / manipulation check. Its defensible conclusion is narrow: increasing centre bias changes hunt-phase positioning as intended, but does not identify a universally superior strength or a cross-pool score improvement. It is not discarded or relabelled as a performance optimization result.

This notebook is the formal Experiment 1. It tests the architectural decision to gate the selected weak preference ($\lambda=.04$), rather than searching for another $\lambda$.
"""),
    markdown(r"""
## Locked design and interpretation rule

| Variant | Inference-time definition |
|---|---|
| No-centre | $\lambda=0$; frozen network, objective routing, revisit penalty, and safety mask |
| Phase-gated centre | $\lambda=.04$ only when no coins/crates remain and an opponent is alive |
| Ungated centre | the same explicit $.04$ Q-bias, additionally active when coins or crates remain and the current position is non-emergency |

- Lineups: `external_mixed`, `internal_mixed`, and `frozen_dddqn`.
- Pairing: seeds 65001–65040, five rounds per seed cluster, with inference-only changes.
- Evidence represented: $3\times3\times40\times5=1{,}800$ rounds.
- Primary contrast: phase-gated minus ungated within each lineup, aggregated to seed clusters before inference.
- Economy: coins and crates. Safety: suicides, survival, and invalid actions. Overall outcome: score; kills are diagnostic. Mechanism: hunt-phase centre distance and occupancy.
- Decision rule: claim support only if paired effects consistently favor gating across pools without an opposing economy or safety signal. A point estimate alone is insufficient; the 95% paired seed-cluster bootstrap interval must show clear directional evidence.

**Holdout disclosure.** Seeds 65001–65040 were already used in the calibration. Exact no-centre / phase-gated clusters were reused when available, and the ungated arm plus missing clusters were newly run. This is therefore a paired reanalysis plus a new treatment arm, not a fresh confirmatory holdout. The ledger contains 160 reused clusters (800 rounds) and 200 final-protocol clusters (1,000 rounds).
"""),
    code(r"""
from __future__ import annotations
import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
RESULT_ROOT = ROOT / 'results' / 'e1_phase_gating'
payload = json.loads((RESULT_ROOT / 'summary.json').read_text())
protocol = payload['protocol']

assert len(payload['round_rows']) == 1800
assert len(payload['seed_rows']) == 360
assert len(payload['ledger']) == 360
dict(Counter(row['source'] for row in payload['ledger']))
"""),
    markdown(r"""
## Aggregate outcomes

All values below are per round. Each row contains 40 seed clusters / 200 rounds.
"""),
    code(r"""
aggregate = pd.DataFrame(payload['aggregate'])
outcome_columns = [
    'lineup', 'variant', 'score', 'coins', 'crates', 'kills',
    'suicides', 'invalid', 'survived', 'timeout', 'action_ms',
]
display(aggregate[outcome_columns].round(3))
"""),
    markdown(r"""
## Primary paired contrast: phase-gated minus ungated

Positive differences favor phase-gated for score, coins, crates, kills, and survival. Negative differences favor phase-gated for suicides and invalid actions. Confidence intervals are paired 10,000-replicate bootstrap intervals over the 40 seed clusters, not over individual rounds.
"""),
    code(r"""
bootstrap = pd.DataFrame(payload['paired_bootstrap'])
primary = bootstrap[bootstrap.contrast == 'phase_gated-ungated_center'].copy()
primary[['ci_low', 'ci_high']] = pd.DataFrame(primary.ci95.tolist(), index=primary.index)
primary_table = primary[
    primary.metric.isin(['score','coins','crates','kills','suicides','invalid','survived'])
][['lineup','metric','mean_difference','ci_low','ci_high','positive_seeds','zero_seeds']]
display(primary_table.round(3))
"""),
    code(r"""
metrics = ['score', 'coins', 'crates', 'suicides']
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, metric in zip(axes.flat, metrics):
    part = primary[primary.metric == metric].set_index('lineup').loc[
        ['external_mixed','internal_mixed','frozen_dddqn']
    ]
    estimate = part.mean_difference.to_numpy()
    low = part.ci_low.to_numpy(); high = part.ci_high.to_numpy()
    ax.errorbar(np.arange(3), estimate,
                yerr=np.vstack([estimate-low, high-estimate]),
                fmt='o', capsize=4, color='#2457A7')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(np.arange(3), ['external', 'internal', 'frozen DDDQN'])
    ax.set_title(f'phase-gated − ungated: {metric}')
    ax.grid(axis='y', alpha=.25)
fig.tight_layout()
figure_dir = RESULT_ROOT / 'figures'; figure_dir.mkdir(exist_ok=True)
fig.savefig(figure_dir / 'phase_minus_ungated_effects.png', dpi=180, bbox_inches='tight')
plt.show()
"""),
    markdown(r"""
## Reference contrast and hunt-phase mechanism

Phase-gated and no-centre are identical during economic play by construction; their coin and crate outcomes are exactly equal in every lineup. The reference contrast asks whether the gated $.04$ term changes hunt behavior or downstream performance after economic objectives are exhausted.
"""),
    code(r"""
reference = bootstrap[bootstrap.contrast == 'phase_gated-no_center'].copy()
reference[['ci_low', 'ci_high']] = pd.DataFrame(reference.ci95.tolist(), index=reference.index)
display(reference[
    reference.metric.isin(['score','hunt_center_distance','hunt_center_occupancy'])
][['lineup','metric','mean_difference','ci_low','ci_high']].round(4))

mechanism_columns = [
    'lineup','variant','hunt_center_distance',
    'hunt_center_progress_rate','hunt_center_occupancy','hunt_decisions',
]
display(aggregate[mechanism_columns].round(4))
"""),
    markdown(r"""
## Results

- **Overall score:** there is no clear phase-gated versus ungated score difference in any pool. The paired effects are +0.060 in `external_mixed` (95% CI −0.245 to +0.365), −0.095 in `internal_mixed` (−0.425 to +0.240), and −0.085 in `frozen_dddqn` (−0.475 to +0.290).
- **Economic progress is opponent-pool dependent.** Ungated gains 0.115 coins/round over phase-gated in `external_mixed` (phase−ungated CI −0.230 to −0.010) and 0.110 in `frozen_dddqn` (−0.220 to −0.005). In contrast, phase-gated retains 0.975 more crates/round in `internal_mixed` (+0.160 to +1.850). Other crate/coin intervals include zero.
- **Safety is not consistently improved by gating.** Suicide and survival intervals include zero in all three pools. `internal_mixed` is the one clear protective case: phase-gated produces 0.410 fewer invalid actions/round than ungated (CI −0.685 to −0.150). Invalid-action intervals include zero in the other pools.
- **The gated heuristic changes hunt positioning without a demonstrated score gain.** Relative to no-centre, phase-gated reduces hunt centre distance and increases occupancy clearly in `internal_mixed` and `frozen_dddqn`; the external interval includes zero. Phase-gated versus no-centre score intervals include zero everywhere, while economic counts are exactly unchanged.

The pattern is mixed in exactly the way the locked rule treats as non-confirmatory: gating protects internal-mixed crate progress and legality, but the ungated preference improves coins in two other pools. No direction dominates overall score or safety across the opponent pools.
"""),
    markdown(r"""
## Answer and design implication

**Answer: not generally.** The experiment does not support a cross-pool claim that phase-gating preserves economy and safety better than applying the same $.04$ centre preference throughout play. It supports a narrower architectural statement: gating prevents the heuristic from perturbing economic decisions and can be protective in at least one opponent pool, but that protection has an opportunity cost because ungated centring increased coin collection in two pools.

Accordingly, phase-gated centring may remain a conservative behavior-shaping choice, but it should not be presented as an established performance optimization. The centre component should stay secondary in the paper; stronger strategic claims should rest on components with direct and consistent practical effects, such as the safety mask and objective routing.
"""),
    markdown(r"""
## Provenance and integrity checks

The following audit verifies all 1,800 represented rounds, one final protocol hash for the 200 newly materialized clusters, and identical five-round initial-condition hashes across the three variants for every lineup/seed pair. Quarantined pre-fix trials are outside the ledger and cannot enter this summary.
"""),
    code(r"""
assert Counter(row['source'] for row in payload['ledger']) == {
    'phase_gating': 200, 'calibration': 160
}
protocol_hash = hashlib.sha256((RESULT_ROOT / 'protocol.json').read_bytes()).hexdigest()
local_protocol_hashes = set()
initial = defaultdict(dict)
for row in payload['ledger']:
    manifest = json.loads((ROOT / row['manifest']).read_text())
    initial[(row['lineup'], row['seed'])][row['variant']] = tuple(
        item['sha256'] for item in manifest['initial_conditions']
    )
    if row['source'] == 'phase_gating':
        local_protocol_hashes.add(manifest['protocol_sha256'])

audit = {
    'represented_rounds': len(payload['round_rows']),
    'seed_rows': len(payload['seed_rows']),
    'ledger_sources': dict(Counter(row['source'] for row in payload['ledger'])),
    'paired_lineup_seed_cells': len(initial),
    'all_initial_conditions_match': all(
        len(group) == 3 and len(set(group.values())) == 1 for group in initial.values()
    ),
    'one_final_protocol_for_new_runs': local_protocol_hashes == {protocol_hash},
    'protocol_sha256': protocol_hash,
    'implementation_sha256': protocol['implementation_sha256'],
}
assert audit['all_initial_conditions_match']
assert audit['one_final_protocol_for_new_runs']
audit
"""),
    markdown(r"""
## Reproduction entry points

The locked implementation is `evaluation/e1_phase_gating.py`; immutable inputs and criteria are recorded in `results/e1_phase_gating/protocol.json`; complete round-, seed-, aggregate-, bootstrap-, and ledger-level results are in `results/e1_phase_gating/summary.json`. The runner is resumable at one five-round seed cluster per manifest. It will refuse changed frozen inputs and will not overwrite valid runs.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "ml_essential", "language": "python", "name": "ml_essential"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3", "version": "3.10.20",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(TARGET)
