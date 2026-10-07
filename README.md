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
- An unprivileged sandbox shell cannot access the NVIDIA device nodes, but a
  host-level check confirms six RTX 3090 GPUs and driver 525.125.06. GPU and
  CARLA runs must therefore be launched in the host-level session.

## Environment checkpoint (2026-10-06)

- Created the isolated Conda environment `riskbench` with Python 3.7.16.
- Installed `torch==1.10.0+cu113`, `torchvision==0.11.1+cu113`, and
  `torchaudio==0.10.0+cu113` from the official PyTorch wheel index. The
  upstream README requests torchvision 0.11.0; the available CUDA 11.3 wheel
  is 0.11.1, which is recorded as a compatibility substitution.
- GPU smoke test passed under the host driver: PyTorch CUDA build 11.3,
  `torch.cuda.is_available() == True`, six NVIDIA GeForce RTX 3090 devices,
  and a 1024x1024 CUDA matrix multiplication completed successfully.
- Host driver remains NVIDIA 525.125.06 (driver-reported CUDA 12.0). The
  environment's CUDA runtime is isolated under the Conda prefix and does not
  modify the host driver or system CUDA toolkit.
- Reproducibility files: `artifacts/environment/riskbench-conda.yml`,
  `artifacts/environment/nvidia-smi.txt`, and
  `artifacts/environment/nvcc.txt`.
- The official `ROI_tool.py` separates raw per-frame risk predictions from
  metric summaries: `model.zip` contains the former, while `ROI_result/`
  contains saved metric JSON. A smoke test initially failed when the latter
  was passed as `--model_root`; the corrected command uses an extracted copy
  of `model.zip` under ignored `artifacts/`.
- The corrected Random/interactive smoke test reproduced the official summary
  exactly: TP/FN/FP/TN = 8712/8722/82662/82336, Precision 9.53%, Recall
  49.97%, F1 16.01%, and PIC 16.7. Its command and output are recorded in
  `artifacts/metrics/random_interactive_smoke.txt` and
  `artifacts/metrics/roi_smoke/Random/interactive.json`.
- `riskbench` passes `pip check`; the only compatibility pin added beyond the
  upstream requirements is `safetensors==0.3.3`, needed because the latest
  `timm==0.9.6` dependency resolution otherwise selects a Python-3.7-
  incompatible safetensors build.

## Single-scenario execution checkpoint (2026-10-07)

- The code split used by the official baseline loaders is now confirmed:
  basic scenarios whose name starts with `10`, `A6`, or `B3` are the test
  split. This gives 515 interactive, 420 collision, 305 obstacle, and 394
  non-interactive variants in the current archive. The 1,865 interactive
  entries in metadata are the complete interactive archive, not the test
  split.
- Added `scripts/prepare_dataset_view.py`. It creates
  `artifacts/dataset_view/`, a symlink-based view with the extra archive
  nesting removed and generated `tracking.npy` files. It does not copy raw
  images or modify `/data/dongzk/RiskBench`.
- Added `scripts/single_scene_baselines.py`. On
  `10_i-1_1_c_f_f_1_rl/ClearSunset_low_`, it reads the released
  `actors_data`, `ego_data`, and `bbox.json`, preserves continuous scores,
  and writes official ROI-tool-compatible boolean JSON.
- Random, Range, and the local constant-velocity Kalman adapter each ran for
  all 84 frames of that variant. Official ROI evaluation completed:
  Random F1 11.76%, Range F1 0.00%, and Kalman adapter F1 27.27%.
  These are data-contract smoke results, not paper reproduction numbers:
  the Kalman adapter still needs the upstream rectangle-collision logic.
- The official ROI evaluator also recomputed all available prediction JSON
  files under `artifacts/model_official/model`; the raw log is
  `artifacts/metrics/official_recomputed.log` and JSON summaries are under
  `artifacts/metrics/official_recomputed/`. These are reference numbers from
  author-provided predictions, not our own model inference.
- A GPU architecture smoke attempt reached the DSA/RRL model constructor and
  dataset loader, but `timm` attempted to fetch an ImageNet ResNet-50
  checkpoint from Hugging Face and failed because it is not cached. The
  official DSA/RRL training or inference therefore requires either the
  corresponding checkpoint/cache or an explicitly documented offline
  initialization choice.
- With `pretrained=False` forced only in a non-production smoke process, the
  DSA/RRL architecture completed a CUDA forward pass on five real frames:
  logits shape `(1, 5, 2)` and object-attention shape `(1, 5, 20)`. This
  confirms the GPU/data tensor path, but is not a trained-model result.

## Reporting and visualization checkpoint (2026-10-07)

- Added `scripts/make_report_assets.py`. It reads the original archive
  read-only and writes report assets under ignored `artifacts/visualization/`.
- The representative interactive scenario is
  `10_i-1_1_c_f_f_1_rl/ClearSunset_low_`: 84 frames, GT risky actor `32284`,
  GT critical frame 37, and ego instance ID `32176`.
- The scene figure combines the RGB front view with only the GT-risk
  bounding box, the ego/GT bird's-eye trajectories, and the GT-object score
  curves from the three local smoke adapters. The frame-37 GT is red; it does
  not indicate a model prediction.
- The report visualizer intentionally omits non-GT gray tracks and ordinary
  candidate boxes so the figure remains readable for presentations. Traffic
  lights and unrelated background actors are not shown.
- For interactive/obstacle data, the official confusion-matrix ground truth
  is the listed GT risk actor during the metadata behavior interval; the
  critical-point metadata is a single reference frame used by PIC/consistency,
  not the only positive frame. Collision evaluation uses the full recorded
  frame range. The risk-type figures show the interval as a shaded band and
  the critical point as a dashed line.
- Obstacle metadata can contain GT actor IDs that have no front-camera bbox
  at the selected critical frame. The report marks this explicitly instead
  of inventing a 2D box; such obstacle objects may also lack a world
  location in `actors_data`, so a trajectory cannot be plotted from the
  released fields.
- Fixed the local obstacle contract on 2026-10-07 using the official data
  conventions: class-21 front instance masks are converted to obstacle boxes,
  `actor_attribute.json` supplies static obstacle BEV geometry, and obstacle
  distance records are included in the Random/Range/Kalman smoke outputs.
- Dataset inventory currently counts 7,218 scenario variants:
  interactive 1,865; collision 1,933; obstacle 1,430; non-interactive 1,990.
  Filtering basic scenario names by the official test prefixes (`10`, `A6`,
  `B3`) yields 1,634 variants: 515 interactive, 420 collision, 305 obstacle,
  394 non-interactive. The script also records per-variant frame counts.
- The single-scene smoke metrics are produced by our lightweight adapters and
  official ROI evaluator, but they are not paper reproduction results. The
  Kalman adapter is not yet the official rectangle-collision implementation.
- `artifacts/metrics/official_recomputed/` contains official-evaluator metrics
  on author-provided prediction JSON. The generated
  `official_prediction_reference_interactive_f1.png` and
  `official_prediction_reference_metrics.csv` are reference-only, not fresh
  inference from our models.

Rebuild the report assets from the workspace root:

```bash
conda run -n riskbench python scripts/make_report_assets.py \
  --data-root /data/dongzk/RiskBench/RiskBench_Dataset \
  --metadata-root /data/dongzk/RiskBench/RiskBench_Dataset/metadata \
  --score-root artifacts/risk_type_scenes/roi \
  --roi-root artifacts/risk_type_scenes/roi \
  --official-metrics-root artifacts/metrics/official_recomputed \
  --output-dir artifacts/visualization
```

Key scene outputs are under `artifacts/visualization/scenes/`; the other
report outputs are `dataset_split_counts.png` and
`official_prediction_reference_interactive_f1.png`. The matching scene
summary is JSON; the full dataset inventory and official prediction reference
table are CSV/JSON so the figures can be regenerated and audited.

The final three representative figures are under
`artifacts/visualization/scenes/`:

- `scene_10_i-1_1_c_f_f_1_rl_ClearSunset_low_.png` (interactive)
- `scene_10_i-1_1_c_r_l_0_HardRainNoon_low_.png` (collision)
- `scene_10_i-1_0_r_sl_ClearSunset_low_.png` (obstacle)

The obstacle figure uses the official instance-segmentation/geometry path. On the representative obstacle
scene, the fixed local Range and local Kalman outputs both contain all four GT
obstacle IDs at the critical interval; official ROI evaluation reports
Recall 100.00% and F1 21.84% for this single-scene smoke test. This is a
contract check, not a paper reproduction result.

## Batch execution checkpoint (2026-10-07)

- Added `scripts/run_lightweight_test_split.py`. It scans only basic scenarios
  beginning with `10`, `A6`, or `B3`, combines per-scene outputs into the
  official `model_root/<method>/<data_type>.json` contract, and writes a
  `batch_manifest.json` with every scene/method status and runtime.
- A four-scene smoke run (one scene from each data type) completed all 12
  adapter jobs with zero errors. The merged Random, Range, and Kalman files
  were each accepted by the official ROI evaluator; the interactive metrics
  were F1 11.76%, 0.00%, and 27.27%, respectively.
- The batch script currently covers only the lightweight adapters. It does not
  claim learned-model inference, and the local Kalman path remains an
  exploratory constant-velocity adapter until the upstream rectangle-collision
  implementation is ported.

Smoke command:

```bash
conda run -n riskbench python scripts/run_lightweight_test_split.py \
  --data-root /data/dongzk/RiskBench/RiskBench_Dataset \
  --output-root artifacts/batch_smoke \
  --limit-per-type 1
```

Full test-prefix command:

```bash
conda run -n riskbench python scripts/run_lightweight_test_split.py \
  --data-root /data/dongzk/RiskBench/RiskBench_Dataset \
  --output-root artifacts/batch_lightweight_test
```

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

- Current branch is `main`; `origin` points to the user-authorized
  `kunmytry10/RiskBench_reproduce` GitHub repository.
- The official upstream checkout under `codes/RiskBench` and the paper under
  `paper/` are workspace inputs and remain untracked; do not stage them
  wholesale. Stage only the reproduction scripts and documentation intended
  for version control.
- The dataset, generated artifacts, logs, and model weights are excluded by
  `.gitignore` and must not be committed.
