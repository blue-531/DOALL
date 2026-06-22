<br/>
  <h1 align="center" style="font-size: 1.7rem">DO-ALL: Distill Once, Adapt Life-Long</h1>
  <p align="center">
  ECCV 2026
  </p>
  <p align="center">
    <a href="https://blue-531.github.io/">Hyun-Kurl Jang*</a>,
    <a href="https://jihun1998.github.io/">Jihun Kim*</a>,
    <a href="https://sangrockeg.github.io/">Hyeokjun Kweon*</a>,
    Kuk-Jin Yoon
  </p>
  <p align="center">
    * denotes equal contribution
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2606.20196">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
  </p>
</p>
<p align="center">
  <img src="assets/do_all_demo_v2.gif" width="90%" alt="DO-ALL demo">
</p>

Official code for **DO-ALL**, a plug-and-play module for Continual Test-Time Adaptation
(CTTA) that revisits source information through **Dataset Distillation (DD)**.

Before deployment, DO-ALL distills the source dataset into a compact set of **synthetic
anchors**. During adaptation, each target sample is matched to its nearest anchor, which guides the update via **anchor
replay**, **feature alignment**, and
**harm-adaptive blending** that rewinds unstable parameter groups toward the source model.
DO-ALL plugs into any base CTTA method *without changing its objective*.

## Installation

```bash
conda create -n doall python=3.10 -y && conda activate doall
pip install torch==2.3.1 torchvision==0.18.1   # match your CUDA build
pip install -r requirements.txt
```

## Data

| Dataset | Download | Expected location |
|---------|----------|-------------------|
| **CIFAR10-C / CIFAR100-C** | Auto-downloaded on first run | `./data/` (managed automatically) |
| **ImageNet-C** | [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF) | `./data/ImageNet-C/<corruption>/<severity>/...` |
| **CCC** | [CCC](https://github.com/oripress/CCC) | `./data/CCC` |

## Pre-trained model weights

All backbones are the standard public checkpoints used by the upstream benchmark and are
**auto-downloaded** on first run — no manual setup:

| Benchmark | Backbone | Weights (`MODEL.ARCH`) | Source |
|-----------|----------|------------------------|--------|
| ImageNet-C / CCC | ResNet-50 | `resnet50`, `IMAGENET1K_V1` | torchvision |
| CIFAR10-C | WRN-28-10 | `Standard` | RobustBench |
| CIFAR100-C | ResNeXt-29 | `Hendrycks2020AugMix_ResNeXt` | RobustBench |

## DD source anchors

The distilled source anchors are released on Google Drive:

[**Download**](https://drive.google.com/file/d/1YRb06F0teMukNQG8YnBmr2KiMw5apFa2/view?usp=sharing) — unpack into `./DD_anchor/`.

Each set is named `{dataset}_{DDmethod}_{backbone}`. Every distilled image carries its source soft label as a sibling
`.pt`. Point `--synpath` at an IPC folder whose `{dataset}_{backbone}` matches the benchmark you run.

| Anchor set | Benchmark | DD method |
|------------|-----------|-----------|
| `imagenet_WMDD_resnet50/IPC_10`  | ImageNet-C / CCC (ResNet-50) | WMDD |
| `imagenet_SRe2L_resnet50/IPC_10` | ImageNet-C / CCC (ResNet-50) | SRe²L |
| `imagenet_DELT_resnet50/IPC_10`  | ImageNet-C / CCC (ResNet-50) | DELT |
| `cifar100_WMDD_resnext/IPC_10`   | CIFAR100-C (ResNeXt-29) | WMDD |
| `cifar100_SRe2L_resnext/IPC_10`  | CIFAR100-C (ResNeXt-29) | SRe²L |
| `cifar100_DELT_resnext/IPC_10`   | CIFAR100-C (ResNeXt-29) | DELT |

Example: `--synpath ./DD_anchor/imagenet_WMDD_resnet50/IPC_10`.

## Quick start

```bash
# baseline (upstream CTTA method)
python test_time.py --cfg cfgs/imagenet_c/roid.yaml

# the SAME baseline + DO-ALL — one flag
python test_time.py --cfg cfgs/imagenet_c/roid.yaml \
    --do_all --synpath ./DD_anchor/imagenet_WMDD_resnet50/IPC_10
```

| Flag | Meaning |
|------|---------|
| `--do_all` | Enable the DO-ALL plug-in (sets `cfg.DOALL.ENABLED`). |
| `--synpath <root>/IPC_<N>` | DD anchor `ImageFolder` (`N` = images-per-class). Required with `--do_all`. |
| `--stride k` | Run the anchor branch every `k` steps for cheaper adaptation. |

### Reproduce Table 2 (ImageNet-to-ImageNet-C, ResNet-50, severity 5, continual)

```bash
SYN_ROOT=./DD_anchor/imagenet_WMDD_resnet50 GPU=0 bash run_table2.sh
# runs the 3 baselines + their 3 +DO-ALL variants; writes run_logs/<method>.log
```

## Acknowledgements

Built on [mariodoebler/test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation)
(EATA, ROID, RMT, and RobustBench utilities). We thank the authors for releasing their code.

## Citation

If you find this work useful, please cite:

```bibtex
@article{jang2026distill,
  title={Distill Once, Adapt Life-Long: Exploring Dataset Distillation for Continual Test-Time Adaptation},
  author={Jang, Hyun-Kurl and Kim, Jihun and Kweon, Hyeokjun and Yoon, Kuk-Jin},
  journal={arXiv preprint arXiv:2606.20196},
  year={2026}
}
```
