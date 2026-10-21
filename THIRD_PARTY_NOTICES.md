# Third-party notices

## Stable-Baselines3 and SB3-Contrib (MIT License)

`semantic_lb_env/semantic_lb_env/p3o.py` subclasses `stable_baselines3.PPO` and reuses its rollout and
update structure; `semantic_lb_env/semantic_lb_env/cpo.py` subclasses `sb3_contrib.TRPO` and keeps the
structure of its `train()` method, inserting the constrained trust-region step.

    The MIT License
    Copyright (c) 2019 Antonin Raffin

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
    EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.

## Data used for calibration (not redistributed)

`semantic_lb_env/traces/trace_profile.json` contains only fitted summary statistics:
per-service latency quantiles fitted from DeathStarBench socialNetwork traces (Gan et al., ASPLOS 2019;
Apache License 2.0) and arrival-rate quantiles fitted from the Alibaba cluster-trace-microservices-v2021
dataset (Luo et al., SoCC 2021). The raw traces are not included; obtain them from the original sources and
follow their terms of use. `scripts/fit_trace_distributions.py` regenerates the profile from the raw data.

## DeepSet baseline

`semantic_lb_env/semantic_lb_env/deepset_policy.py` is an independent re-implementation of the
permutation-equivariant actor / invariant critic structure described by DRL-FORCH (Di Cicco et al., NetSoft
2023). No code from the DRL-FORCH repository is included.
