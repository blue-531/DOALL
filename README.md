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
</p>

---

Official code for **DO-ALL**, a plug-and-play module for Continual Test-Time Adaptation
(CTTA) that revisits source information through **Dataset Distillation (DD)**.

Before deployment, DO-ALL distills the source dataset into a compact set of **synthetic
anchors**. During adaptation, each target sample is matched to its nearest anchor, which guides the update via **anchor
replay**, **feature alignment**, and
**harm-adaptive blending** that rewinds unstable parameter groups toward the source model.
DO-ALL plugs into any base CTTA method *without changing its objective*.

This repo is built on the upstream
[test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation)
## Quick start

```bash
# baseline
python test_time.py --cfg cfgs/imagenet_c/roid.yaml

# the SAME baseline + DO-ALL (one flag)
python test_time.py --cfg cfgs/imagenet_c/roid.yaml --do_all --synpath ./WMDD_imagenet/IPC_10/23
```

`--do_all` enables the plug-in; `--synpath <root>/IPC_<N>/<exp>` points at the DD anchor
ImageFolder (`N` = images-per-class); `--stride k` runs the anchor branch every `k` steps
(Table 6). All three are also settable in a config under the `DOALL` node.

Reproduce all of Table 2 (3 baselines + 3 `+DO-ALL`):

```bash
bash run_table2.sh        # writes run_logs/<method>.log; prints mean error per run
```

## How DO-ALL plugs in

DO-ALL lives entirely in **[`methods/do_all.py`](methods/do_all.py)** (a `DOALL` class with its
own frozen source copy + DD anchor bank). Each baseline adds the *same* 3 lines:

```python
# __init__:
self.do_all = DOALL(self, cfg, num_classes) if cfg.DOALL.ENABLED else None
# top of forward_and_adapt(self, x):
if self.do_all is not None and self.do_all.should_run():
    self.do_all.anchor_update(x)          # anchor objective + harm-aware rewind
```

So `methods/{eata,roid,rmt}.py` are the upstream baselines plus this hook; nothing else
changes. See [`methods/OURS_METHOD.md`](methods/OURS_METHOD.md) for the full method and
hyper-parameters (KD T=5, weights `W_CE=1, W_MMD=10, W_MIX=1`, rewind `β_max=0.05, β_s=5`).

## Installation

```bash
conda create -n doall python=3.10 -y && conda activate doall
pip install torch==2.3.1 torchvision==0.18.1   # match your CUDA build
pip install -r requirements.txt
```

## Required data & assets

Paths resolve from `cfg.DATA_DIR` (default `./data`) and `--synpath`.

| Asset | Needed by | Where |
|-------|-----------|-------|
| **ImageNet-C** (test) | all | `./data/ImageNet-C/<corruption>/<severity>/...` — [zenodo 2235448](https://zenodo.org/records/2235448) |
| **ImageNet train** (source) | EATA*, RMT* | `./data/imagenet2012/` — EATA Fisher / RMT warm-up + prototypes. Not needed by ROID. |
| **DD source anchors** | any `--do_all` run | `--synpath <root>/IPC_<N>/<exp>` — ImageFolder of distilled images (WMDD / SRe²L / DELT, or a coreset with `CORESET` in the path). |

The ResNet-50 backbone is `torchvision IMAGENET1K_V1` (auto-downloaded). RMT also writes/reads a
warm-up checkpoint + prototypes under `./ckpt/` (regenerated from ImageNet-train if absent).



## Acknowledgements

Built on [mariodoebler/test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation). We thank the authors for releasing their code.
