# E3former

Official code for **Energy-Adaptive Equivariant State Space Models for Noise-Robust Protein Structure Representation** (**NeurIPS 2026**).


## Overview

E3former is an equivariant Transformer–state space model (SSM) hybrid for
residue-level protein representation learning from imperfect experimental and
AlphaFold-predicted structures. It combines two components:

- **Energy-aware adaptive graph:** a Lennard-Jones-shaped, distance-derived
  structural score adjusts residue neighborhoods to reduce sensitivity to local
  coordinate deviations. The score is a structural proxy, not a calibrated
  physical energy. The reported experiments sample adaptive radii during both
  training and inference.
- **Equivariant Elastic-SSM:** selected high-order tensor features from successive
  Transformer layers form a sequence along network depth. Parameter-sharing
  equivariant operations integrate these features into type-0 invariant summaries,
  which are combined with the final Transformer representation for prediction.

The paper evaluates inverse folding and protein–protein binding-site prediction
on CATH and PPBS, together with their AlphaFold-predicted variants, CATH-AF and
PPBS-AF.


## Installation

Python **3.10 is required** by the installer and package configuration.
A CUDA-capable GPU is recommended for training.

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

Dataset sizes reported in Appendix A.1, Table S1:

| Dataset | CLI name | Train | Validation | Test | Task |
| --- | --- | --- | --- | --- | --- |
| CATH | `cath` | 18,024 | 608 | 1,120 | Inverse folding |
| CATH-AF | `cath-af` | 16,468 | 559 | 1,015 | Inverse folding |
| PPBS | `ppbs` | 12,577 | 3,178 | 3,984 | Binding-site prediction |
| PPBS-AF | `ppbs-af` | 11,345 | 2,876 | 3,636 | Binding-site prediction |

CATH follows the CATH 4.2 40% nonredundant benchmark with topology-separated
splits and chains of at most 500 residues. PPBS follows the ScanNet split
protocol. The AlphaFold variants retain the original split assignments for
entries with available mapped predicted structures; PPBS-AF transfers interface
labels by sequence alignment.

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

Appendix A.2 reports training for at most 50 epochs with validation-based early
stopping, a learning rate of 0.001, and batch sizes of 16 or 32. The examples below
use batch size 32. The CLI defaults are 50 epochs, batch size 8, and seed 52.
These commands train the public implementation; the paper and release settings
are compared below.

```bash
e3former train --dataset cath    --features seq --epochs 50 --batch-size 32 --output runs/cath
e3former train --dataset cath-af --features seq --epochs 50 --batch-size 32 --output runs/cath-af
e3former train --dataset ppbs    --features seq --epochs 50 --batch-size 32 --output runs/ppbs
e3former train --dataset ppbs-af --features seq --epochs 50 --batch-size 32 --output runs/ppbs-af
```

The task and input dimensions are configured automatically. Feature names map to
the paper's settings as follows: `seq` → +Seq, `angles` → +κ, α (adding virtual
torsion and bond angles over C-alpha atoms), and `backbone` → +ϕ, ψ, ω (adding
backbone dihedrals while retaining the C-alpha angles). Inverse folding excludes
amino-acid identity inputs; binding-site prediction includes residue identity.

Other options include `--patience 10`, `--device cuda:1`, `--seed 52`,
`--learning-rate 0.001`, `--base-radius`, and `--energy-radius`. The radius
defaults are 9.5 / 2.5 for CATH and CATH-AF, and 7.0 / 3.0 for PPBS and PPBS-AF.
Training writes `checkpoints/best.ckpt`, `checkpoints/last.ckpt`, and CSV logs
under the output directory. The best checkpoint minimizes validation loss.
Early stopping uses validation loss; `--patience 0` disables it.
Use `--resume runs/cath/checkpoints/last.ckpt` to resume training.

### Paper and Release Settings

The result table above reports the paper's experiments. The current public
configuration has the following differences from Appendix A.2 that need to be
resolved for exact reproduction:

| Setting | Camera-ready paper | Current release |
| --- | --- | --- |
| Inverse-folding output classes | 20 residue types | 23 configured classes |
| Tensor degrees supplied to Elastic-SSM | L = 3, 4, 5 | L = 3, 4, 5, 6 |
| Dropout | 0 | Attention dropout 0.2; prediction-head dropout 0 |

See the [task configuration](e3former/config.py),
[encoder defaults](proteinworkshop/models/graph_encoders/e3former.py), and
[prediction-head configuration](proteinworkshop/config/decoder/residue_type.yaml).

## Evaluation

The [release checkpoint manifest](checkpoints/manifest.json) is currently empty.
Evaluate the best checkpoint produced by training:

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

This checks the training and evaluation workflow on a small subset; its metrics
are not the full-dataset results reported in the paper.

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

## Citation

If you find E3former useful in your research, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{zhang2026e3former,
  title     = {Energy-Adaptive Equivariant State Space Models for Noise-Robust Protein Structure Representation},
  author    = {Zhang, Zhongyue and Ma, Runze and Huang, Yanjie and Zheng, Shuangjia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
