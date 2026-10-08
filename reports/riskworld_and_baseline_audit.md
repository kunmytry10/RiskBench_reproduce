# RiskWorld and Baseline Audit

## RiskWorld Table 1

RiskWorld is `arXiv:2608.21414`, *RiskWorld: Object-Centric Latent World
Modeling for Autonomous Driving Risk Identification*. Its Table 1 reproduces
the RiskBench baseline rows exactly: Random, Range (5m), Range (10m), and
Kalman filter have the same values as RiskBench Table II. The paper says it
compares these baselines, but neither the paper nor the public repository
provides a separate RiskBench-baseline rerun log, prediction archive, or
baseline command. We can verify the values are copied identically, but cannot
claim RiskWorld independently reproduced those four baselines.

RiskWorld reports 6,916 total scenarios and uses the map-based split. The
original RiskBench paper reports 1,689 test scenarios. Our released prediction
files contain exactly 1,632 common test scene keys (515/420/303/394 by type),
so our manifest is correct for the files evaluated here, but it is not the
paper's full 1,689 count.

## Range 5m rerun

Using the official Range rule, the same manifest, and a 5m threshold:

| Type | P | R | PIC | FA |
|---|---:|---:|---:|---:|
| interactive | 38.72 | 10.57 | 7.8 | - |
| collision | 86.83 | 24.44 | 3.6 | - |
| obstacle | 44.15 | 17.89 | 9.6 | - |
| non-interactive | - | - | - | 2.99 |
| all F1 | - | - | - | 16.61 |

These are close to the paper's Range (5m) values (38.7/10.7, 85.6/27.7,
45.0/18.3, FA 3.3). The all-F1 value differs because the released scene
counts and confusion-matrix population differ.

## Random and Kalman diagnosis

The low Recall is not explained by CARLA being a different dataset. The
released files are CARLA recordings and contain actor states, tracks, boxes,
ego states, segmentation, and static-obstacle geometry.

1. Range 10m and Range 5m are close to the paper, including per-type values.
2. The published author Random JSON has multiple `true` object IDs in some
   frames, although the official source code selects one ID with
   `random.choice(all_ids_list)`. Thus the published JSON is not a transparent
   one-to-one dump of the current source path, or it was postprocessed.
3. The upstream Kalman implementation has class-level OpenCV filter state and
   numerical warnings on zero-motion ego segments. Our adapter resets that
   state per scenario and calls the upstream function unchanged. Its remaining
   gap is an unresolved parity issue, not tuned output.
   The obstacle-ID type check in the upstream code was also tested with the
   exact integer list passed by `data_generator.py`; representative obstacle,
   interactive, and collision scenes produced the same selections as the
   previous adapter path, so this is not the source of the large Recall gap.
4. These lightweight rules do not need RGB except obstacle instance
   segmentation; unused camera files are not missing inputs for them.

Random and Kalman should therefore not yet be frozen as paper-equivalent
baselines. Range 10m/5m is currently the trustworthy rule-based baseline. A
rigorous Kalman claim requires the exact official prediction-generation
contract or a rerun of the original CARLA process, followed by frame-level ID
parity checks.
