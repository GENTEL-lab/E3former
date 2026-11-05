# E3former

Code for *Energy-Adaptive Equivariant State Space Models for Noise-Robust Protein Structure Representation*.

## Installation

Python 3.10 and a CUDA-capable GPU are recommended.

```bash
git clone https://github.com/GENTEL-lab/E3former.git
cd E3former

conda create -n e3former python=3.10 -y
conda activate e3former
bash scripts/install.sh
```

The installer uses PyTorch 2.1.0, CUDA 12.1 wheels, PyG 2.4.0, and e3nn 0.5.1.

## Data

Datasets: [GENTEL-Lab/E3former_data](https://huggingface.co/datasets/GENTEL-Lab/E3former_data).

```bash
python -m pip install -e '.[hub]'
python scripts/download.py --namespace GENTEL-Lab
```

Downloads are pinned to the release revision and checked with SHA256. To select
individual datasets, add `--only cath-af` (also accepts `cath`,
`ppbs`, and `ppbs-af`).

The script downloads and extracts the processed datasets into `data/`:

```text
data/
  cath/           processed/*.pt, chain_set_splits.json
  cathAF/         processed/*.pt, chain_set_splits.json
  binding_site/   processed/*.pt, labels/*_ids.txt
  bindingsiteAF/  processed/*.pt, labels/*_ids.txt
```

```bash
e3former check-data
```

For data stored elsewhere, use `--data-dir /path/to/data` or set `E3FORMER_DATA`.

## Training

```bash
e3former train --dataset cath    --features seq --epochs 25 --output runs/cath
e3former train --dataset cath-af --features seq --epochs 25 --output runs/cath-af
e3former train --dataset ppbs    --features seq --epochs 25 --output runs/ppbs
e3former train --dataset ppbs-af --features seq --epochs 25 --output runs/ppbs-af
```

The task and input dimensions are configured automatically. Feature choices are
`seq`, `angles` (C-alpha angles), and `backbone` (including backbone dihedrals).
Inverse folding uses sequence positions without amino-acid identity inputs.

Common options: `--epochs 25`, `--patience 10`, `--batch-size 8`, `--device cuda:1`,
`--learning-rate 0.001`, `--base-radius 9.5`, and `--energy-radius 2.5`.
Training writes `checkpoints/best.ckpt`, `checkpoints/last.ckpt`, and CSV logs
under the output directory. The best checkpoint minimizes validation loss.
Early stopping uses validation loss; `--patience 0` disables it.
Use `--resume runs/cath/checkpoints/last.ckpt` to resume training.

## Evaluation

Evaluate the best checkpoint from training:

```bash
e3former evaluate --checkpoint runs/cath/checkpoints/best.ckpt --output runs/eval-cath
e3former evaluate --checkpoint runs/cath-af/checkpoints/best.ckpt --output runs/eval-cath-af
e3former evaluate --checkpoint runs/ppbs/checkpoints/best.ckpt --output runs/eval-ppbs
e3former evaluate --checkpoint runs/ppbs-af/checkpoints/best.ckpt --output runs/eval-ppbs-af
```

The checkpoint supplies the model configuration and feature setting. Results are
written to `metrics.json`. Inverse folding reports residue-weighted perplexity
and recovery; binding-site prediction reports AUPRC (average precision).
Add `--save-predictions` to export predictions as CSV.

For PPBS, `--subset` accepts `70`, `homology`, `topology`, `none`, or `all`
(default). `all` pools residues from the four test splits. Evaluation uses the
complete selected split, including the final incomplete batch.

Training and evaluation accept `--seed`, `--batch-size`, and `--device`.

## Quick Check

```bash
e3former train --dataset cath-af --epochs 1 --batch-size 2 --limit-proteins 8 \
  --max-train-batches 2 --max-val-batches 2 --output runs/smoke
e3former evaluate --checkpoint runs/smoke/checkpoints/best.ckpt --batch-size 2 --limit-proteins 16 \
  --output runs/smoke-eval
```

## ProteinWorkshop Encoders

The original ProteinWorkshop encoders are available through `--encoder`, for
example `schnet`, `egnn`, `tfn`, `gcpnet`, `gear_net`, `gear_net_edge`, `gvp`, and
`mace`. Install optional dependencies with `pip install -e '.[baselines]'` when
using GearNet or ESM. Model implementations are in
`proteinworkshop/models/graph_encoders/`.

Built on [ProteinWorkshop](https://github.com/a-r-j/ProteinWorkshop).
Code is released under the [MIT license](LICENSE).
