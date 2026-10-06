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
