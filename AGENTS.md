# RiskBench Reproduction Instructions

## Scope and filesystem safety

- The only project workspace that may be modified is `/home/gaoh/RiskBench_reproduce`.
- Treat `/data/dongzk/RiskBench` as read-only input data. Do not modify, move, rename, or delete files there.
- Do not modify, move, rename, or delete any directory outside this workspace, including any other user's home directory.
- Keep generated plots, reports, caches, logs, checkpoints, and experiment outputs inside this repository (prefer `artifacts/`, `results/`, or `logs/`).

## Environment

- Run the reproduction in an Anaconda/conda environment. Record the environment name, Python version, package versions, CUDA/PyTorch details, and relevant commands in `README.md`.
- Prefer the official implementation under `codes/` and document any necessary compatibility patch before applying it.
- Do not silently download or vendor large datasets or model weights into the repository; document external paths and download commands instead.

## Reproduction workflow

- First inspect and document the dataset schema, modalities, splits, labels, and the RiskBench input/output contract.
- Build reproducible dataset exploration and visualization scripts before running model metrics.
- Reproduce the paper's reported baselines, including random, range, and Kalman filter where supported by the official code, using the paper's split, preprocessing, horizons, metrics, and seeds.
- Save exact commands, configuration files, raw metric outputs, plots, and comparison tables under version-controlled project paths.
- Keep a chronological reproduction log in `README.md`; link each result to the commit and command that produced it.

## Git and documentation

- Initialize or use the Git repository in this directory only.
- Make small, descriptive commits at meaningful milestones. Never commit raw datasets, private credentials, or machine-specific secrets.
- Before any `git push`, inspect the remote URL and branch, and push only this repository's intended commits to the user-authorized GitHub remote.
- Update `README.md` whenever setup, data findings, experiment commands, results, or limitations change.

## Verification

- Before claiming success, rerun the documented commands in the conda environment and record their exit status and key outputs.
- Distinguish official results, independently reproduced results, and exploratory findings.

## Current Git and reproduction state

- GitHub SSH authentication was verified successfully:
  `ssh -T git@github.com` returned:
  `Hi kunmytry10! You've successfully authenticated, but GitHub does not provide shell access.`
- Reproduction repository remote:
  `git@github.com:kunmytry10/RiskBench_reproduce.git`
- Current branch:
  `main`
- Initial documentation commit:
  `ec6dcaa docs: record RiskBench reproduction plan`
- The initial commit has been pushed successfully to `origin/main`.
- Git identity:
  - name: `kunmytry10`
  - email: `13986110509@163.com`
- At the time of this update, `codes/` and `paper/` are present as untracked project inputs. Do not run `git add codes paper` blindly.
- The official source repository is under `codes/RiskBench`; preserve its own upstream Git metadata and do not rewrite its history.
- Dataset input root:
  `/data/dongzk/RiskBench`
- Dataset files are currently treated as read-only during exploration unless a later task explicitly requires a safe copy or generated derivative.
- GPU baseline confirmed by user:
  - 6 NVIDIA GPUs
  - 24 GB memory per GPU
  - driver `525.125.06`
  - driver-reported CUDA `12.0`
  - no running compute processes at the time of inspection
- No CUDA version has been changed. Future dependencies must be isolated in a dedicated Conda environment and must not modify other environments.
- Reproduction status:
  1. Git setup: complete.
  2. Dataset archive/schema reconnaissance: partially documented in `README.md`; no full exploration script committed yet.
  3. Visualization: not implemented yet.
  4. RiskBench input/output contract: identified from the official code and documented partially in `README.md`.
  5. Random, Range, and Kalman Filter metrics: not independently rerun yet.
- Next execution milestones:
  1. inspect `codes/RiskBench` entrypoints and dependencies;
  2. create an isolated Conda environment or record an existing compatible environment;
  3. add read-only dataset inspection and visualization scripts under this repository;
  4. run the official metric pipeline for Random, Range, and Kalman Filter;
  5. save raw outputs and comparison tables under repository artifact directories;
  6. update this file and `README.md`;
  7. commit and push each meaningful milestone.
- New-window continuation checklist:
  1. run `git status --short --branch`;
  2. run `git log --oneline --decorate -5`;
  3. run `git remote -v`;
  4. read this section and the reproduction log in `README.md`;
  5. verify that no command writes into another user's home directory.

