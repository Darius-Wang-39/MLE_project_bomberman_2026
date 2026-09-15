#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
external_root="${1:-$project_root/../external_previous_cohort_benchmark}"
source_root="$external_root/agent_code"

agents=(
  baseline
  deep_learning_killer
  double_q
  dqn
  dqn_5tile
  feature_is_everything
  q_learning
  sarsa
  sarsa_lambda
)

if [[ ! -d "$source_root" ]]; then
  echo "External agent directory not found: $source_root" >&2
  exit 2
fi

for agent in "${agents[@]}"; do
  source_dir="$source_root/$agent"
  adapter_dir="$project_root/agent_code/external_$agent"
  if [[ ! -d "$source_dir" ]]; then
    echo "missing: external_$agent ($source_dir)"
    continue
  fi

  mkdir -p "$adapter_dir"
  find "$adapter_dir" -maxdepth 1 -type l -delete
  while IFS= read -r source_file; do
    filename="$(basename "$source_file")"
    # Evaluation adapters deliberately exclude training callbacks. This prevents
    # an accidental --train invocation from overwriting a linked model/table.
    [[ "$filename" == "train.py" ]] && continue
    ln -s "$source_file" "$adapter_dir/$filename"
  done < <(find "$source_dir" -maxdepth 1 -type f ! -name '.DS_Store' -print | sort)
  echo "linked: external_$agent"
done

# Older torch/pickle files embed their original Python package names. Provide
# evaluation-only aliases so unpickling can import those classes while the game
# still runs and reports the explicitly external_* adapter name.
pickle_aliases=(dqn dqn_5tile feature_is_everything)
for agent in "${pickle_aliases[@]}"; do
  source_dir="$source_root/$agent"
  alias_dir="$project_root/agent_code/$agent"
  [[ -d "$source_dir" ]] || continue
  if [[ -e "$alias_dir" && ! -f "$alias_dir/.external_benchmark_alias" ]]; then
    echo "refusing to replace existing internal package: $alias_dir" >&2
    exit 3
  fi
  mkdir -p "$alias_dir"
  : > "$alias_dir/.external_benchmark_alias"
  find "$alias_dir" -maxdepth 1 -type l -delete
  for filename in model.py features.py symmetry.py; do
    [[ -f "$source_dir/$filename" ]] && ln -s "$source_dir/$filename" "$alias_dir/$filename"
  done
  echo "aliased for pickle only: agent_code.$agent"
done
