"""Build notebook 14 from the completed contextual-hybrid development screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "e2_contextual_hybrid_screen" / "summary.json"
TARGET = ROOT / "notebooks" / "14_e2_contextual_hybrid_screen.ipynb"
VARIANTS = (
    "one_branch_relaxed", "original_hybrid",
    "phase_productive_hybrid", "armed_threat_hybrid",
)


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


def result(summary, lineup, contrast, metric):
    return next(
        row for row in summary["paired_bootstrap"]
        if row["lineup"] == lineup and row["contrast"] == contrast and row["metric"] == metric
    )


def decision_text(summary):
    lines = []
    for candidate in ("phase_productive_hybrid", "armed_threat_hybrid"):
        score_directions = []
        trap_directions = []
        for lineup in summary["protocol"]["lineups"]:
            score = result(summary, lineup, f"original_hybrid-{candidate}", "score")
            trap = result(
                summary, lineup, f"one_branch_relaxed-{candidate}",
                "confirmed_catastrophic_self_trap",
            )
            # Stored contrasts are baseline-candidate; invert for readable candidate-baseline direction.
            score_directions.append((-score["mean_difference"], -score["ci_high"], -score["ci_low"]))
            trap_directions.append((-trap["mean_difference"], -trap["ci_high"], -trap["ci_low"]))
        lines.append(
            f"- `{candidate}` vs original score Δ: "
            + "; ".join(
                f"{lineup} {values[0]:+.3f} [{values[1]:+.3f}, {values[2]:+.3f}]"
                for lineup, values in zip(summary["protocol"]["lineups"], score_directions)
            )
            + ". vs relaxed confirmed catastrophic self-trap Δ: "
            + "; ".join(
                f"{lineup} {values[0]:+.3f} [{values[1]:+.3f}, {values[2]:+.3f}]"
                for lineup, values in zip(summary["protocol"]["lineups"], trap_directions)
            )
            + "."
        )
    advances = []
    for candidate in ("phase_productive_hybrid", "armed_threat_hybrid"):
        score_intervals = []
        for lineup in summary["protocol"]["lineups"]:
            stored = result(summary, lineup, f"original_hybrid-{candidate}", "score")
            score_intervals.append((-stored["ci_high"], -stored["ci_low"]))
        if all(low > 0 for low, _ in score_intervals):
            advances.append(candidate)
    strict_advantage_lineups = [
        lineup for lineup in summary["protocol"]["lineups"]
        if result(
            summary, lineup, "one_branch_relaxed-original_hybrid",
            "confirmed_catastrophic_self_trap",
        )["ci_low"] > 0
    ]
    original_strict_advantage = len(strict_advantage_lineups) == len(summary["protocol"]["lineups"])
    if not advances:
        selection = (
            "No new candidate clears the pre-specified score-improvement requirement, so neither "
            "advances to fresh-seed confirmation. `one_branch_relaxed` remains the score-optimal baseline. "
            + (
                "The strict outcome-confirmed metric also supports `original_hybrid` as a safety compromise."
                if original_strict_advantage else
                "Under the strict outcome-confirmed metric, `original_hybrid` does not show a general "
                "catastrophic-self-trap advantage over relaxed across all lineups. A measurable advantage "
                "appears only in " + (
                    ", ".join(f"`{name}`" for name in strict_advantage_lineups)
                    if strict_advantage_lineups else "no lineup"
                ) + "; the rule should therefore be described as a context-dependent conservative policy, "
                "not a universally established safety compromise."
            )
        )
    else:
        selection = "Candidate(s) meeting the development promotion rule: " + ", ".join(f"`{x}`" for x in advances) + "."
    return (
        "This is a development-screen conclusion, not fresh confirmation.\n\n"
        + "\n".join(lines)
        + "\n\n" + selection
        + "\n\nOnly a candidate that improves the original hybrid without a clear outcome-safety "
        "regression, while retaining a measurable direct-safety advantage over relaxed, may "
        "advance. At most one candidate should be taken to fresh seeds."
    )


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    aggregate_columns = [
        "lineup", "variant", "rounds", "score", "coins", "crates", "bombs", "kills",
        "suicides", "survived", "self_trapped_after_bomb",
        "post_placement_escape_collapse", "confirmed_catastrophic_self_trap",
        "confirmed_catastrophic_self_traps_per_100_bombs", "median_confirmed_trap_step",
        "bomb_proposals_total", "bomb_rejections_total", "decision_ms_p95",
    ]
    aggregate_table = table(summary["aggregate"], aggregate_columns)
    attribution_columns = [
        "lineup", "variant", "actual_bombs_total", "raw_zero_total",
        "same_step_only_zero_total", "post_placement_escape_collapse_total",
        "confirmed_catastrophic_self_trap_total", "confirmed_crate_economy_total",
        "confirmed_hunt_total", "confirmed_placement_step_median",
        "confirmed_death_latency_median", "score_at_placement_mean",
        "score_at_placement_median", "no_score_gain_after_confirmed_rate",
    ]
    attribution_table = table(summary["strict_self_trap_attribution"], attribution_columns)
    robustness_rows = []
    for lineup in protocol.get("robustness_lineups", []):
        for candidate in ("original_hybrid", "phase_productive_hybrid", "armed_threat_hybrid"):
            contrast = f"one_branch_relaxed-{candidate}"
            score = result(summary, lineup, contrast, "score")
            trap = result(summary, lineup, contrast, "confirmed_catastrophic_self_trap")
            suicide = result(summary, lineup, contrast, "suicides")
            robustness_rows.append({
                "lineup": lineup,
                "contrast": contrast,
                "score_delta_95CI": f"{score['mean_difference']:+.3f} [{score['ci_low']:+.3f}, {score['ci_high']:+.3f}]",
                "confirmed_trap_delta_95CI": f"{trap['mean_difference']:+.3f} [{trap['ci_low']:+.3f}, {trap['ci_high']:+.3f}]",
                "suicide_delta_95CI": f"{suicide['mean_difference']:+.3f} [{suicide['ci_low']:+.3f}, {suicide['ci_high']:+.3f}]",
            })
    robustness_table = table(
        robustness_rows,
        ["lineup", "contrast", "score_delta_95CI", "confirmed_trap_delta_95CI", "suicide_delta_95CI"],
    )
    inference_metrics = {
        "score", "coins", "crates", "bombs", "kills", "suicides", "survived",
        "self_trapped_after_bomb", "post_placement_escape_collapse",
        "confirmed_catastrophic_self_trap", "same_step_only_zero",
        "self_trap_to_suicide", "self_trap_to_any_death",
        "economic_confirmed_self_traps", "hunt_confirmed_self_traps",
    }
    inference = [row for row in summary["paired_bootstrap"] if row["metric"] in inference_metrics]
    inference_table = table(
        inference,
        ["lineup", "contrast", "metric", "mean_difference", "ci_low", "ci_high",
         "positive_seeds", "zero_seeds", "negative_seeds"],
    )
    operative = {
        "one_branch_relaxed": {"reject_no_escape_branch"},
        "original_hybrid": {
            "reject_no_escape_branch", "reject_nonproductive", "reject_step_cutoff",
            "reject_active_bomb", "reject_active_explosion", "reject_near_any_opponent",
        },
        "phase_productive_hybrid": {
            "reject_no_escape_branch", "reject_nonproductive", "reject_no_crates_remaining",
            "reject_active_bomb", "reject_active_explosion", "reject_near_any_opponent",
        },
        "armed_threat_hybrid": {
            "reject_no_escape_branch", "reject_nonproductive", "reject_step_cutoff",
            "reject_active_bomb", "reject_active_explosion", "reject_near_armed_opponent",
            "reject_predicted_branch_threat",
        },
    }
    reason_rows = []
    reason_columns = [
        "reject_no_escape_branch", "reject_nonproductive", "reject_no_crates_remaining",
        "reject_step_cutoff", "reject_active_bomb", "reject_active_explosion",
        "reject_near_any_opponent", "reject_near_armed_opponent",
        "reject_predicted_branch_threat",
    ]
    for source in summary["rejection_reason_totals"]:
        row = dict(source)
        for key in reason_columns:
            if key not in operative[source["variant"]]:
                row[key] = "—"
        reason_rows.append(row)
    reason_table = table(
        reason_rows,
        ["lineup", "variant", "reject_no_escape_branch", "reject_nonproductive",
         "reject_no_crates_remaining", "reject_step_cutoff", "reject_active_bomb",
         "reject_active_explosion", "reject_near_any_opponent",
         "reject_near_armed_opponent", "reject_predicted_branch_threat"],
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
            matches += sum(len({arm[index] for arm in hashes}) == 1 for index in range(5))
        pairing.append({"lineup": lineup, "matched_rounds": matches, "expected": 200})
    pairing_table = table(pairing, ["lineup", "matched_rounds", "expected"])

    cells = [
        cell("markdown", """# Notebook 14 — Contextual hybrid bomb-permission screen

This development experiment asks whether contextual one-branch permission can retain most of the relaxed policy's productivity while reducing forced bomb self-traps. Strict two-branch is archived and is not a candidate. The original 1,600-round core screen is supplemented by a separately locked 1,600-round robustness extension on `rule_based` and `internal_mixed`.

**Development-set note.** This experiment uses the previously observed seeds `66001–66040` to compare the relaxed, original hybrid, phase/productive hybrid, and armed-threat hybrid variants. These results are for screening only; final conclusions require fresh seeds."""),
        cell("markdown", f"""## Locked protocol

- Candidate: `{protocol['candidate']}`; checkpoint `{protocol['checkpoint_sha256']}`
- Conditions: {', '.join(f'`{name}`' for name in VARIANTS)}
- Lineups: {', '.join(f'`{name}`' for name in protocol['lineups'])}
- Seeds: `{protocol['seeds'][0]}–{protocol['seeds'][-1]}`; five rounds per seed
- Workload: {protocol['total_rounds']} rounds
- Structure: {protocol.get('core_rounds', 0)} core rounds + {protocol.get('robustness_extension_rounds', 0)} auxiliary robustness rounds
- Inference: five rounds averaged within seed, then 10,000-replicate paired bootstrap
- Status: development screen; strict two-branch archived

The enhanced pipeline records all counterfactual permissions, multi-label rejection reasons, Q margins, branch directions, phase and threat context, and bomb lifecycle outcomes."""),
        cell("code", """from pathlib import Path
import json, pandas as pd
from IPython.display import display
ROOT = Path('..').resolve()
payload = json.loads((ROOT/'results/e2_contextual_hybrid_screen/summary.json').read_text())
aggregate = pd.DataFrame(payload['aggregate'])
paired = pd.DataFrame(payload['paired_bootstrap'])
rejection_reasons = pd.DataFrame(payload['rejection_reason_totals'])
strict_attribution = pd.DataFrame(payload['strict_self_trap_attribution'])
pd.set_option('display.max_columns', 30)"""),
        cell("markdown", "## Completeness and pairing\n\nAll four arms must match on every initial-condition hash."),
        cell("code", """variants = ['one_branch_relaxed', 'original_hybrid',
            'phase_productive_hybrid', 'armed_threat_hybrid']
manifest_map = {}
for row in payload['ledger']:
    manifest_map[(row['lineup'], row['variant'], row['seed'])] = json.loads(
        (ROOT / row['manifest']).read_text()
    )
pairing_rows = []
for lineup in payload['protocol']['lineups']:
    matched = 0
    for seed in payload['protocol']['seeds']:
        hashes = [
            [item['sha256'] for item in manifest_map[(lineup, variant, seed)]['initial_conditions']]
            for variant in variants
        ]
        matched += sum(len({arm[index] for arm in hashes}) == 1 for index in range(5))
    pairing_rows.append({'lineup': lineup, 'matched_rounds': matched, 'expected': 200})
pairing_check = pd.DataFrame(pairing_rows)
display(pairing_check)"""),
        cell("markdown", """## Strict self-trap definition

The original `self_trapped_after_bomb` flag is retained as `raw_zero` for auditability, but it is no longer the primary safety outcome because it includes a non-actionable sample taken on the BOMB placement step.

- `post_placement_escape_collapse`: a zero-viability sample occurs at a strictly later step.
- `confirmed_catastrophic_self_trap`: post-placement escape collapse occurs and the agent subsequently dies with its own bomb causally involved (`own_bomb_suicide` or `own_and_opponent_bomb`). This conservative outcome-confirmed label is the primary self-trap metric below.

This revision re-summarises the immutable traces; it does not rerun or alter any game.
"""),
        cell("markdown", "## Aggregate outcomes and strict lifecycle diagnostics"),
        cell("code", """aggregate_columns = [
    'lineup', 'variant', 'rounds', 'score', 'coins', 'crates', 'bombs', 'kills',
    'suicides', 'survived', 'self_trapped_after_bomb',
    'post_placement_escape_collapse', 'confirmed_catastrophic_self_trap',
    'confirmed_catastrophic_self_traps_per_100_bombs', 'median_confirmed_trap_step',
    'bomb_proposals_total', 'bomb_rejections_total', 'decision_ms_p95',
]
display(aggregate[aggregate_columns].round(4))"""),
        cell("markdown", """## Focused robustness contrasts

All entries are `one_branch_relaxed − comparator`, with paired 95% seed-bootstrap intervals. Positive score values favor relaxed; positive trap or suicide values indicate worse safety for relaxed.

"""),
        cell("code", """def paired_interval(lineup, contrast, metric):
    row = paired[
        (paired.lineup == lineup) &
        (paired.contrast == contrast) &
        (paired.metric == metric)
    ].iloc[0]
    return f\"{row.mean_difference:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]\"

robustness_rows = []
for lineup in payload['protocol']['robustness_lineups']:
    for comparator in ['original_hybrid', 'phase_productive_hybrid', 'armed_threat_hybrid']:
        contrast = f'one_branch_relaxed-{comparator}'
        robustness_rows.append({
            'lineup': lineup,
            'contrast': contrast,
            'score Δ [95% CI]': paired_interval(lineup, contrast, 'score'),
            'confirmed trap Δ [95% CI]': paired_interval(
                lineup, contrast, 'confirmed_catastrophic_self_trap'),
            'suicide Δ [95% CI]': paired_interval(lineup, contrast, 'suicides'),
        })
robustness_effects = pd.DataFrame(robustness_rows)
display(robustness_effects)"""),
        cell("markdown", """### Robustness interpretation

The robustness result is context-dependent. Against three rule-based opponents, relaxed has higher point-estimate score but its interval overlaps zero, while confirmed catastrophic self-traps and suicides are clearly higher than under every hybrid. This is not merely reduced bomb exposure: original hybrid placed 8,366 bombs versus relaxed's 8,135, yet confirmed traps fell from 47 to 30 (0.578 to 0.359 per 100 bombs). Moreover, 45/47 relaxed confirmed traps occurred during crate economy, with median placement step 91, indicating early self-generated bombing risk rather than opponent-driven hunt pressure. Against `internal_mixed`, neither score nor strict-safety contrasts are resolved. Thus the hybrid safety effect is real in the rule-based lineup but does not generalise uniformly across opponent pools; none of the two new hybrids establishes a Pareto improvement over the original hybrid.
"""),
        cell("markdown", """## Strict self-trap attribution

`confirmed_death_latency_median` is measured from the first post-placement zero-viability step to death. Scores are official game scores at bomb placement. The final column is the share of confirmed catastrophic self-traps after which no further official score was gained.
"""),
        cell("code", """attribution_columns = [
    'lineup', 'variant', 'actual_bombs_total', 'raw_zero_total',
    'same_step_only_zero_total', 'post_placement_escape_collapse_total',
    'confirmed_catastrophic_self_trap_total', 'confirmed_crate_economy_total',
    'confirmed_hunt_total', 'confirmed_placement_step_median',
    'confirmed_death_latency_median', 'score_at_placement_mean',
    'score_at_placement_median', 'no_score_gain_after_confirmed_rate',
]
display(strict_attribution[attribution_columns].round(4))"""),
        cell("markdown", """## Rejection attribution

Reasons are multi-label: one rejected proposal can contribute to several columns. Counts therefore need not sum to total rejections. `—` means the contextual flag is not an operative gate for that variant.
"""),
        cell("code", """reason_columns = [
    'reject_no_escape_branch', 'reject_nonproductive', 'reject_no_crates_remaining',
    'reject_step_cutoff', 'reject_active_bomb', 'reject_active_explosion',
    'reject_near_any_opponent', 'reject_near_armed_opponent',
    'reject_predicted_branch_threat',
]
operative = {
    'one_branch_relaxed': {'reject_no_escape_branch'},
    'original_hybrid': {'reject_no_escape_branch', 'reject_nonproductive',
        'reject_step_cutoff', 'reject_active_bomb', 'reject_active_explosion',
        'reject_near_any_opponent'},
    'phase_productive_hybrid': {'reject_no_escape_branch', 'reject_nonproductive',
        'reject_no_crates_remaining', 'reject_active_bomb', 'reject_active_explosion',
        'reject_near_any_opponent'},
    'armed_threat_hybrid': {'reject_no_escape_branch', 'reject_nonproductive',
        'reject_step_cutoff', 'reject_active_bomb', 'reject_active_explosion',
        'reject_near_armed_opponent', 'reject_predicted_branch_threat'},
}
rejection_display = rejection_reasons.copy()
for index, row in rejection_display.iterrows():
    for column in reason_columns:
        if column not in operative[row.variant]:
            rejection_display.loc[index, column] = pd.NA
display(rejection_display[['lineup', 'variant', *reason_columns]])"""),
        cell("markdown", """## All pairwise paired-bootstrap intervals

Every contrast is `left − right`. Positive values favor the left arm for score/productivity/survival; negative values favor it for suicide and self-trap metrics. `confirmed_catastrophic_self_trap` is the primary safety contrast; broader zero-viability signals are diagnostic only.
"""),
        cell("code", """main_inference_metrics = [
    'score', 'bombs', 'crates', 'suicides', 'survived',
    'post_placement_escape_collapse', 'confirmed_catastrophic_self_trap',
]
paired_main = paired[paired.metric.isin(main_inference_metrics)][[
    'lineup', 'contrast', 'metric', 'mean_difference', 'ci_low', 'ci_high',
    'positive_seeds', 'zero_seeds', 'negative_seeds', 'paired_seeds',
]].copy()
with pd.option_context('display.max_rows', None):
    display(paired_main.round(4))"""),
        cell("markdown", "## Development decision\n\n" + decision_text(summary)),
        cell("markdown", """## Reproduction and provenance

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e2_contextual_hybrid_screen.py status
/opt/homebrew/Caskroom/miniconda/base/envs/ml_essential/bin/python evaluation/e2_contextual_hybrid_screen.py summarize
```

Raw stats, version-2 traces, manifests, hashes, seed rows and bootstrap results are stored under `results/e2_contextual_hybrid_screen/`."""),
        cell("markdown", """## Figure 6 — Core productivity–safety trade-off

The following cell is the complete plotting code used for Figure 6. It reads the stored seed-level paired-bootstrap results, regenerates both publication files, and displays the figure inline.
"""),
        cell("code", r'''# Reproduce Figure 6 directly from the E2 paired-bootstrap results.
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

project_root = Path('..').resolve()
summary_path = project_root / 'results/e2_contextual_hybrid_screen/summary.json'
figure_png = project_root / 'results/e2_contextual_hybrid_screen/figure6_e2_core_paired_effects.png'
figure_pdf = project_root / 'results/e2_contextual_hybrid_screen/figure6_e2_core_paired_effects.pdf'
e2 = json.loads(summary_path.read_text())

core_lineups = ('external_mixed', 'frozen_dddqn')
hybrids = (
    ('original_hybrid', 'Original hybrid', 'o', '#0072B2'),
    ('phase_productive_hybrid', 'Phase/productive', 's', '#009E73'),
    ('armed_threat_hybrid', 'Armed threat', 'D', '#E6AB02'),
)
metrics = (
    ('score', 'Score Δ', '↑ better'),
    ('bombs', 'Bombs / round Δ', 'Exposure only'),
    ('confirmed_catastrophic_self_trap',
     'Confirmed traps / round Δ', '↓ safer'),
)

def hybrid_minus_relaxed(lineup, hybrid, metric):
    """Invert the stored relaxed-minus-hybrid contrast."""
    contrast = f'one_branch_relaxed-{hybrid}'
    row = next(
        item for item in e2['paired_bootstrap']
        if item['lineup'] == lineup
        and item['contrast'] == contrast
        and item['metric'] == metric
    )
    return -row['mean_difference'], -row['ci_high'], -row['ci_low']

def shared_limits(metric):
    values = [0.0]
    for lineup in core_lineups:
        for hybrid, *_ in hybrids:
            values.extend(hybrid_minus_relaxed(lineup, hybrid, metric))
    span = max(values) - min(values)
    padding = max(
        span * 0.12,
        0.025 if metric == 'confirmed_catastrophic_self_trap' else 0.1,
    )
    return min(values) - padding, max(values) + padding

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.5,
    'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 9.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.25), constrained_layout=False)
y_positions = list(range(len(hybrids)))[::-1]
x_limits = {metric: shared_limits(metric) for metric, *_ in metrics}

for row_index, lineup in enumerate(core_lineups):
    for column_index, (metric, title, direction) in enumerate(metrics):
        ax = axes[row_index, column_index]
        ax.axvline(0, color='#8A8A8A', linewidth=1.0, linestyle='--', zorder=0)
        ax.axhline(-0.5, color='#D8D8D8', linewidth=0.6, zorder=0)
        for y, (hybrid, label, marker, color) in zip(y_positions, hybrids):
            estimate, low, high = hybrid_minus_relaxed(lineup, hybrid, metric)
            ax.errorbar(
                estimate, y,
                xerr=[[estimate - low], [high - estimate]],
                fmt=marker, markersize=6.2,
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.7,
                ecolor=color, elinewidth=1.7, capsize=3.0, capthick=1.2, zorder=3,
            )
        ax.set_xlim(*x_limits[metric])
        ax.set_ylim(-0.55, len(hybrids) - 0.45)
        ax.set_yticks(y_positions)
        if column_index == 0:
            ax.set_yticklabels([item[1] for item in hybrids])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis='x', color='#E6E6E6', linewidth=0.7, zorder=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel(r'$\Delta$ (hybrid $-$ relaxed)')
        if row_index == 0:
            ax.set_title(f'{title}\n{direction}', pad=10, fontweight='semibold')

legend_handles = [
    Line2D([0], [0], marker=marker, color=color, label=label, linestyle='none',
           markerfacecolor=color, markeredgecolor='white', markersize=7)
    for _, label, marker, color in hybrids
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.55, 0.005), columnspacing=2.2)
fig.suptitle('E2: Paired effects in the core lineups',
             x=0.55, y=0.985, fontsize=13, fontweight='bold')
fig.text(0.55, 0.947,
         'Points: mean seed differences; bars: paired 95% CIs.',
         ha='center', va='top', fontsize=9.5, color='#555555')
fig.subplots_adjust(left=0.21, right=0.985, top=0.84, bottom=0.16,
                    wspace=0.28, hspace=0.56)
for row_index, lineup in enumerate(core_lineups):
    box = axes[row_index, 0].get_position()
    fig.text(0.025, (box.y0 + box.y1) / 2, lineup, rotation=90,
             va='center', ha='center', fontsize=10.5,
             fontweight='bold', color='#333333')

fig.savefig(figure_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(figure_pdf, bbox_inches='tight', facecolor='white')
plt.show()
'''),
        cell("markdown", r"""**Figure 6.** Paired effects of contextual bomb-permission rules relative to `one_branch_relaxed` in the two core lineups. Points denote mean seed-level differences and horizontal bars show paired 95% bootstrap intervals. Positive score differences favour the hybrid rule, whereas negative self-trap differences indicate fewer confirmed catastrophic self-traps. Differences in executed bombs indicate exposure changes and are not themselves a performance objective.

Vector version: [PDF](../results/e2_contextual_hybrid_screen/figure6_e2_core_paired_effects.pdf).
"""),
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
