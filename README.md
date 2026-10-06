# RiskBench Reproduction

This repository records a reproduction of the RiskBench paper and official
code. The project workspace is `/home/gaoh/RiskBench_reproduce`; the input
archives are read-only under `/data/dongzk/RiskBench`.

## Safety and scope

- Do not modify, move, or delete anything outside this workspace.
- Do not modify the input archives under `/data/dongzk/RiskBench`.
- Keep extracted samples, plots, metrics, logs, and reports in this repository.
- Run all experiments from a conda environment and record the exact package,
  CUDA, and command information here.

## Initial findings (2026-10-06)

- Official source: `codes/RiskBench`, currently at upstream commit
  `f130390` (`HCIS-Lab/RiskBench`).
- Paper: `paper/paper-riskbench.pdf`.
- The paper reports 1,689 test scenarios: 521 interactive, 375 collision,
  322 obstacle, and 471 non-interactive.
- Table II reports Precision, Recall, PIC, non-interactive false-alarm rate,
  and all-scenario F1. Table III reports planning-aware IR and collision rate.
- Official planning-aware inference exposes `Random`, `Range`, and
  `Kalman_Filter` as modes, followed by Social-GAN, MANTRA, QCNet, DSA, RRL,
  BP, and BCP variants.
- `metadata.zip` is readable without extraction and contains `GT_risk`,
  `GT_critical_point`, `behavior`, and `state`. The sample GT risk mapping is
  scenario-weather to actor IDs; the sample transform arrays have shape
  `(T, 7)` and velocity arrays `(T, 3)`.
- `data_collection.zip` contains raw interactive and obstacle recordings with
  transforms, velocities, filters, traffic lights, and variant scenario files.
- `Dataset_for_LBC_Training/dataset.zip` is currently not readable by Python's
  ZIP reader or `unzip` because its central directory is unavailable. It is
  preserved untouched; recovery or a replacement archive is required before
  reproducing LBC-dependent planning experiments.
- No NVIDIA driver is visible in the current session. GPU-dependent models
  and CARLA planning-aware runs therefore require a later GPU-enabled conda
  environment/host.

## Reproduction plan

1. Freeze the environment: create a dedicated conda environment matching the
   official Python 3.7, PyTorch 1.10, CUDA 11.3, CARLA 0.9.14 and pinned
   dependencies where the host supports them. Record `conda env export`, GPU
   information, source commit, archive checksums, and all paths.
2. Inventory the archives read-only. Verify ZIP integrity, count scenarios by
   type/split, inspect metadata schemas, and use streaming reads or a small
   extracted sample. Do not commit raw data or model weights.
3. Build dataset exploration scripts that summarize scenario taxonomy,
   weather/density/map distributions, sequence lengths, actor IDs, labels,
   transforms/velocities, bounding boxes, and metadata joins. Generate static
   plots and one or two trajectory/BEV examples under `artifacts/`.
4. Write the input/output contract: each baseline consumes the official
   front-view tracklets/boxes or trajectory history plus metadata as required,
   and emits per-frame actor/event risk scores or selected IDs. Validate this
   contract on one scenario before large runs.
5. Reproduce the lightweight baselines first: Random, Range at 5 m and 10 m,
   and Kalman filter. Use the official data retrieval, inference, final-output,
   and metric code where possible; make compatibility patches only in new
   wrapper files and document every patch.
6. Reproduce trajectory baselines in increasing cost order: Social-GAN,
   MANTRA, then QCNet. Use the supplied weights when present and record the
   exact checkpoint and split. Defer retraining unless inference cannot be
   reproduced from supplied artifacts.
7. Reproduce vision/collision and behavior baselines (DSA, RRL, BP, BCP) only
   after the data contract and metric harness pass on the lightweight methods.
   Keep each method's raw JSON output and normalized metric table separately.
8. Run planning-aware evaluation only on a GPU/CARLA-capable host, beginning
   with Random, Range, and Kalman on interactive/obstacle scenarios, then add
   the learned methods. Compare IR and collision rate with Table III and
   clearly label hardware or data deviations.
9. Add deterministic smoke tests for archive readers, metadata joins, output
   schema, PIC/F1 calculations, and one end-to-end sample. Record full commands,
   exit codes, runtime, and known deviations in this README.
10. Commit each milestone with descriptive messages. Configure and verify the
    user-authorized GitHub remote before pushing; never commit credentials,
    archives, checkpoints, or generated videos.

## Repository state

The managed workspace contains a read-only `.git` directory, so standard
`git init` cannot write there. A writable project Git metadata directory is
temporarily initialized at `.git-repro/.git`; commands currently need
`git --git-dir=.git-repro/.git --work-tree=.` until this work is moved to a
normal checkout. No GitHub remote has been configured for this reproduction
repository yet.
