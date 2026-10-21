# SC-MPPO: semantic-candidate constrained Maskable PPO for cross-domain service scheduling

Anonymous code and data for the double-blind review of the paper
(cross-domain service scheduling with semantic candidates and constrained Maskable PPO).

This repository contains:

* the simulation environment (`semantic_lb_env/semantic_lb_env/`);
* the training and evaluation scripts (`semantic_lb_env/scripts/`);
* the per-seed results of every 16-seed experiment in the paper (`semantic_lb_env/results/`, `results/`);
* the statistics scripts that recompute the paper's tables and Holm-corrected p-values from those results.

Trained model weights are not included; every result can be regenerated with the commands below.

## Setup

Python 3.8.10, CPU only:

```bash
python3.8 -m venv rl_env
./rl_env/bin/pip install -r requirements.txt
```

All commands below run from `semantic_lb_env/` with

```bash
cd semantic_lb_env
export PYTHONPATH=$(pwd):$(pwd)/scripts
PY=../rl_env/bin/python
```

## Recompute the tables from the included results (seconds)

```bash
$PY scripts/reproduce_tables.py
```

This prints Table 3 and the Holm-corrected p-values of every pre-specified comparison family
(simulation, single-node testbed, instance-scale ladder, telemetry candidates) and writes
`results/reproduced_tables.json`. The families and their sizes are listed at the top of the script.
Statistics: exact two-sided paired Wilcoxon signed-rank test (zero differences dropped, tie-aware exact null
distribution, `scripts/stats_helpers.py`), Holm correction within each family, the training seed as the
independent unit, and 95% t intervals across seeds.

## Regenerate the per-seed results

Each experiment is a training run over 16 seeds followed by a re-evaluation under the stochastic and the
deterministic (argmax) protocol, 20 episodes per seed. Running these commands overwrites the corresponding
directories under `results/`.

### Strong baselines (Table 3, Table A1)

```bash
$PY scripts/run_multiseed.py --seeds 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 --scenario-seed 42 --eval-seed 10042 --timesteps 30000 --eval-episodes 20 --output-dir results/multiseed_sb16 --trace-profile traces/trace_profile.json --sla-limit 0.115 --cost-limit 0.457 --failure-limit 0.031 --include-p3o --p3o-penalty-coef 0.003 --p3o-cost-vf-coef 0.5 --include-dqn --dqn-timesteps 30000 --cost-model synthetic_uniform
$PY scripts/reeval_protocol.py --source-root results/multiseed_sb16 --out-root results/reeval_sb16 --eval-episodes 20 --trace-profile traces/trace_profile.json --scenario-seed 42 --eval-seed 10042 --own-sla-limit 0.115 --own-cost-limit 0.457 --own-failure-limit 0.031 --cost-model synthetic_uniform
```

### CPO (Table 3, Section 6.1)

The training command also retrains the seven arms of the strong-baseline run; they reproduce it bit for bit.

```bash
$PY scripts/run_multiseed.py --seeds 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 --scenario-seed 42 --eval-seed 10042 --timesteps 30000 --eval-episodes 20 --output-dir results/multiseed_sb16_cpo --trace-profile traces/trace_profile.json --sla-limit 0.115 --cost-limit 0.457 --failure-limit 0.031 --include-p3o --p3o-penalty-coef 0.003 --p3o-cost-vf-coef 0.5 --include-dqn --dqn-timesteps 30000 --cost-model synthetic_uniform --include-cpo
$PY scripts/reeval_protocol.py --source-root results/multiseed_sb16_cpo --out-root results/reeval_sb16_cpo --eval-episodes 20 --trace-profile traces/trace_profile.json --scenario-seed 42 --eval-seed 10042 --own-sla-limit 0.115 --own-cost-limit 0.457 --own-failure-limit 0.031 --cost-model synthetic_uniform
```

### DeepSet scheduler (Table 3, Section 6.1)

As above, the seven strong-baseline arms are retrained alongside and reproduce bit for bit.

```bash
$PY scripts/run_multiseed.py --seeds 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 --scenario-seed 42 --eval-seed 10042 --timesteps 30000 --eval-episodes 20 --output-dir results/multiseed_sb16_deepset --trace-profile traces/trace_profile.json --sla-limit 0.115 --cost-limit 0.457 --failure-limit 0.031 --include-p3o --p3o-penalty-coef 0.003 --p3o-cost-vf-coef 0.5 --include-dqn --dqn-timesteps 30000 --cost-model synthetic_uniform --include-deepset
$PY scripts/reeval_protocol.py --source-root results/multiseed_sb16_deepset --out-root results/reeval_sb16_deepset --eval-episodes 20 --trace-profile traces/trace_profile.json --scenario-seed 42 --eval-seed 10042 --own-sla-limit 0.115 --own-cost-limit 0.457 --own-failure-limit 0.031 --cost-model synthetic_uniform
```

### Instance-scale ladder (Table A3)

```bash
$PY scripts/run_multiseed.py --seeds 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 --scenario-seed 42 --eval-seed 10042 --timesteps 30000 --eval-episodes 20 --output-dir results/multiseed_scale_t15 --trace-profile traces/trace_profile.json --sla-limit 0.1147 --cost-limit 0.4575 --failure-limit 0.0309 --include-p3o --p3o-penalty-coef 0.003 --p3o-cost-vf-coef 0.5 --include-dqn --dqn-timesteps 30000 --cost-model synthetic_uniform --services-per-domain 5
$PY scripts/reeval_protocol.py --source-root results/multiseed_scale_t15 --out-root results/reeval_scale_t15 --eval-episodes 20 --trace-profile traces/trace_profile.json --scenario-seed 42 --eval-seed 10042 --own-sla-limit 0.1147 --own-cost-limit 0.4575 --own-failure-limit 0.0309 --cost-model synthetic_uniform --services-per-domain 5
$PY scripts/run_multiseed.py --seeds 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 --scenario-seed 42 --eval-seed 10042 --timesteps 30000 --eval-episodes 20 --output-dir results/multiseed_scale_t30 --trace-profile traces/trace_profile.json --sla-limit 0.0918 --cost-limit 0.4477 --failure-limit 0.036 --include-p3o --p3o-penalty-coef 0.003 --p3o-cost-vf-coef 0.5 --include-dqn --dqn-timesteps 30000 --cost-model synthetic_uniform --services-per-domain 10
$PY scripts/reeval_protocol.py --source-root results/multiseed_scale_t30 --out-root results/reeval_scale_t30 --eval-episodes 20 --trace-profile traces/trace_profile.json --scenario-seed 42 --eval-seed 10042 --own-sla-limit 0.0918 --own-cost-limit 0.4477 --own-failure-limit 0.036 --cost-model synthetic_uniform --services-per-domain 10
$PY scripts/run_multiseed.py --seeds 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 --scenario-seed 42 --eval-seed 10042 --timesteps 30000 --eval-episodes 20 --output-dir results/multiseed_scale_t45 --trace-profile traces/trace_profile.json --sla-limit 0.1198 --cost-limit 0.4693 --failure-limit 0.0387 --include-p3o --p3o-penalty-coef 0.003 --p3o-cost-vf-coef 0.5 --include-dqn --dqn-timesteps 30000 --cost-model synthetic_uniform --services-per-domain 15
$PY scripts/reeval_protocol.py --source-root results/multiseed_scale_t45 --out-root results/reeval_scale_t45 --eval-episodes 20 --trace-profile traces/trace_profile.json --scenario-seed 42 --eval-seed 10042 --own-sla-limit 0.1198 --own-cost-limit 0.4693 --own-failure-limit 0.0387 --cost-model synthetic_uniform --services-per-domain 15
```

### Telemetry candidates (Section 6.8, Fig. 11)

A1's fixed risk coefficient (0.9617) was calibrated on held-out seeds 90 and 91.

```bash
for seed in $(seq 70 85); do
  common="--timesteps 100000 --eval-episodes 50 --seed $seed --skip-dqn --eval-deterministic true"
  $PY scripts/run_experiment.py --output-dir ../results/telemetry_mask_formal/A0_seed$seed $common
  $PY scripts/run_experiment.py --output-dir ../results/telemetry_mask_formal/A1_seed$seed $common --telemetry-mask-enabled --telemetry-beta-strategy constant --telemetry-beta-fixed 0.9617
  $PY scripts/run_experiment.py --output-dir ../results/telemetry_mask_formal/A2_seed$seed $common --telemetry-mask-enabled --telemetry-beta-strategy linear
done
```

### CPO diagnostic (Appendix A.5)

CPO alone is retrained in a 2 × 3 design: failure-rate limit ∈ {0.031 (main configuration), 0.070 (a level
reached by the other constrained methods at the end of training)} × samples per update ∈ {256, 2048, 16384},
118 updates per seed in every cell, seeds 68–83, one PyTorch thread. Results are in
`results/cpo_diag/<cell>/seed_<k>/` (training: `summary.json`, `cpo_cost_curve.csv` with the branch taken by
every update) and `results/reeval_cpo_diag/<cell>/<protocol>/seed_<k>/metrics.csv` (evaluation). Cell C0 of the
paper is the main CPO run (`results/multiseed_sb16_cpo`). Cells:

| cell | failure limit | samples per update | timesteps per seed |
|---|---|---|---|
| C0p | 0.031 | 256 | 30,000 |
| C1 | 0.070 | 256 | 30,000 |
| C2 | 0.031 | 2048 | 241,664 |
| C3 | 0.070 | 2048 | 241,664 |
| C4 | 0.031 | 16384 | 1,933,312 |
| C5 | 0.070 | 16384 | 1,933,312 |

A cell is the CPO arm of the main CPO command with `--failure-limit`, `--n-steps` and `--timesteps` set as in
the table (`OMP_NUM_THREADS=1`).

## Notes on reproducibility

* Simulation results are bit-for-bit reproducible with the pinned package versions and the same PyTorch
  intra-op thread count; a different thread count changes floating-point reduction order and hence the
  trained weights. The CPO and DeepSet runs record their thread count in `summary.json` (`torch_threads`:
  12 intra-op threads, PyTorch's default on the machine that produced them, no `OMP_NUM_THREADS` set); the
  strong-baseline arms are retrained bit for bit by those two commands. On a machine whose default differs,
  set `OMP_NUM_THREADS=12`. The CPO diagnostic cells used one thread, as stated above.
* Three columns of `metrics.csv` are wall-clock measurements of the scheduler's own overhead
  (`decision_latency_us_mean`, `decision_latency_us_p99`, `mask_build_latency_us_mean`). They depend on the
  machine and its load, are not reproducible, and are not used by any table.
* The single-node testbed results (`results/dsb_testbed_run/`) come from deploying the trained policies,
  without retraining, as an external scheduler in front of DeathStarBench socialNetwork service containers on
  one host. The per-run records include the request-stream hash, per-decision overhead breakdown and the CPU
  interference measured during the run. The container orchestration harness is not part of this release.
* `semantic_lb_env/predictors/risk_mlp_v4.pth` is the risk predictor used by the telemetry-candidate
  experiment (Section 6.8); it is loaded by the environment when `--telemetry-mask-enabled` is set.

## Data

`semantic_lb_env/traces/trace_profile.json` holds latency quantiles fitted from DeathStarBench
socialNetwork traces and arrival-rate quantiles fitted from the Alibaba microservices trace (2021); see
`THIRD_PARTY_NOTICES.md`.

## License

MIT (see `LICENSE`). Third-party notices are in `THIRD_PARTY_NOTICES.md`.
