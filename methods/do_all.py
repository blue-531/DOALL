"""
DO-ALL: Distill Once, Adapt Life-Long  —  modular plug-in for CTTA methods.

DO-ALL is attached to any base TTA method (EATA / ROID / RMT) and, at every (strided)
adaptation step, runs ONE anchor-guided update *before* the base method's own update:

  1. match each target sample to its nearest DD source anchor (cosine over frozen-source
     deep features);
  2. minimise an anchor objective  L = W_CE * KD(T=5) + W_MMD * MMD + W_MIX * MixUp  and
     take an optimizer step;
  3. harm-adaptive blending: rewind the parameter groups whose update is most harmful to
     the source anchors back toward the frozen source model.

The module is self-contained: it keeps its OWN frozen source copy, so the base method
needs no change beyond a 3-line hook.  See the paper for the full method and hyper-parameters.

Enable with ``--do_all`` (sets ``cfg.DOALL.ENABLED``). Anchors come from ``--synpath`` ->
``cfg.DOALL.SYNDATA_PATH`` (an ImageFolder of distilled images, path ``.../IPC_<N>/<exp>``).
"""

import os
import logging
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.synthetic_loading import get_synthetic_loader

logger = logging.getLogger(__name__)


# ----------------------------- helpers -----------------------------

def sample_lambda(batch_size: int, alpha: float, device) -> torch.Tensor:
    return torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)


def mmd_loss(src: torch.Tensor, tgt: torch.Tensor,
             bandwidths=(0.5, 1.0, 2.0), unbiased: bool = True,
             eps: float = 1e-12) -> torch.Tensor:
    """Multi-kernel RBF MMD^2 (U-statistic) between two [N, D] feature sets."""
    assert src.dim() == 2 and tgt.dim() == 2 and src.size(1) == tgt.size(1)
    Ns, Nt = src.size(0), tgt.size(0)

    def pdist2(x, y):
        x2 = (x * x).sum(dim=1, keepdim=True)
        y2 = (y * y).sum(dim=1, keepdim=True).t()
        return torch.clamp(x2 + y2 - 2.0 * x @ y.t(), min=0.0)

    dxx, dyy, dxy = pdist2(src, src), pdist2(tgt, tgt), pdist2(src, tgt)
    Kxx = Kyy = Kxy = 0.0
    for sigma in bandwidths:
        s2 = torch.as_tensor(sigma, dtype=src.dtype, device=src.device) ** 2
        Kxx = Kxx + torch.exp(-dxx / (2.0 * s2 + eps))
        Kyy = Kyy + torch.exp(-dyy / (2.0 * s2 + eps))
        Kxy = Kxy + torch.exp(-dxy / (2.0 * s2 + eps))

    if unbiased:
        z = torch.tensor(0.0, dtype=src.dtype, device=src.device)
        Kxx_sum = (Kxx.sum() - Kxx.diag().sum()) / (Ns * (Ns - 1) + eps) if Ns > 1 else z
        Kyy_sum = (Kyy.sum() - Kyy.diag().sum()) / (Nt * (Nt - 1) + eps) if Nt > 1 else z
        Kxy_sum = Kxy.mean()
    else:
        Kxx_sum, Kyy_sum, Kxy_sum = Kxx.mean(), Kyy.mean(), Kxy.mean()
    return torch.clamp(Kxx_sum + Kyy_sum - 2.0 * Kxy_sum, min=0.0)


def nn_by_cosine(features_test: torch.Tensor, features_src: torch.Tensor, topk: int = 1):
    ft = F.normalize(features_test, p=2, dim=1)
    fs = F.normalize(features_src, p=2, dim=1)
    S = ft @ fs.T
    if topk == 1:
        indices = S.argmax(dim=1, keepdim=True)
        sims = S.gather(1, indices)
    else:
        sims, indices = torch.topk(S, k=topk, dim=1, largest=True, sorted=True)
    return indices, sims


# ----------------------------- the plug-in -----------------------------

class DOALL:
    """Anchor-guided + harm-aware-rewind plug-in attached to a host TTAMethod.

    The host must expose ``.model`` (adapting model), ``.optimizer`` and ``.device``.
    """

    def __init__(self, host, cfg, num_classes: int):
        self.host = host
        self.model = host.model
        self.device = host.device
        self.num_classes = num_classes
        self.arch_name = cfg.MODEL.ARCH

        # own frozen source replica (independent of the base method)
        self.src_model = deepcopy(self.model).to(self.device).eval().requires_grad_(False)

        # ----- DD anchor set -----
        self.coreset_path = cfg.DOALL.SYNDATA_PATH
        if not self.coreset_path or not os.path.exists(self.coreset_path):
            raise FileNotFoundError(
                f"[DO-ALL] DD anchor path not found: {self.coreset_path!r}. "
                "Pass --synpath <path>/IPC_<N>/<exp> together with --do_all."
            )
        self.ipc = int(self.coreset_path.split("_")[-1])      # e.g. .../IPC_10 -> 10
        self.is_coreset = "CORESET" in self.coreset_path
        self.batch_size_dd = self.ipc * num_classes
        _, self.coreset_loader = get_synthetic_loader(
            data_root_dir=self.coreset_path,
            batch_size=self.batch_size_dd,
            workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
        )
        self.batch = next(iter(self.coreset_loader))          # whole DD set fits in one batch
        logger.info(f"[DO-ALL] ipc={self.ipc}, anchors={self.batch_size_dd}, is_coreset={self.is_coreset}")

        # ----- objective weights (from cfg.DOALL) -----
        self.lambda_anchor = float(cfg.DOALL.LAMBDA_ANCHOR)
        self.w_ce = float(cfg.DOALL.W_CE)
        self.w_mmd = float(cfg.DOALL.W_MMD)
        self.w_mix = float(cfg.DOALL.W_MIX)
        self.kd_temperature = 5.0

        # ----- harm-aware rewind config -----
        self.score_type = cfg.DOALL.SCORE_TYPE      # "first_order" | "second_order"
        self.group_by = cfg.DOALL.GROUP_BY          # "layer" | "param"
        self.percentile = float(cfg.DOALL.PERCENTILE)
        self.beta_max = float(cfg.DOALL.BETA_MAX)
        self.beta_scale = float(cfg.DOALL.BETA_SCALE)
        self.eps = 1e-8
        self._ema_sg2: Dict[str, torch.Tensor] = {}

        # ----- stride gating -----
        self.stride = max(1, int(cfg.DOALL.STRIDE))
        self._step = 0

        # ----- per-arch backbone split + cached anchor features -----
        self._build_layered_backbones()
        self._extract_anchor_feature()
        logger.info(f"[DO-ALL] enabled: lambda={self.lambda_anchor}, "
                    f"w=(ce {self.w_ce}, mmd {self.w_mmd}, mix {self.w_mix}), stride={self.stride}")

    # ------------------- public API used by the host -------------------
    def should_run(self) -> bool:
        run = (self._step % self.stride == 0)
        self._step += 1
        return run

    @torch.enable_grad()
    def anchor_update(self, x):
        """One anchor-guided optimizer step + harm-aware rewind, on the host's model."""
        imgs_test = x[0]
        model, optimizer = self.model, self.host.optimizer

        groups = self._named_param_groups(model, self.group_by)
        prev_params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

        # 1) match each test sample to its nearest anchor (frozen-source feature space)
        with torch.no_grad():
            features_test = self._extract_test_features_src(imgs_test)
        indices, _ = nn_by_cosine(features_test, self.total_feat)
        idx = indices[:, 0].cpu()

        imgs_src_full = self.batch[0]
        soft_label_src = self.batch[3]
        imgs_src = imgs_src_full[idx].to(self.device)

        # 2) anchor objective: KD + MMD + MixUp
        loss_mmd, outputs_src, outputs_tgt = self._anchor_forward_and_mmd(imgs_src, imgs_test)

        T = self.kd_temperature
        if self.is_coreset:
            tgt = soft_label_src.squeeze()[idx].to(self.device)
            loss_ce_src = F.kl_div(F.log_softmax(outputs_src / T, dim=-1), tgt, reduction="batchmean") * T
        else:
            tgt = F.softmax(soft_label_src.squeeze().to(self.device)[indices[:, 0]] / T, dim=-1)
            loss_ce_src = F.kl_div(F.log_softmax(outputs_src / T, dim=-1), tgt, reduction="batchmean") * (T * T)

        lam = sample_lambda(imgs_test.size(0), alpha=1.0, device=imgs_test.device)
        w = lam.view(-1, 1, 1, 1)
        imgs_mixed = w * imgs_test + (1 - w) * imgs_src
        outputs_mixed = model(imgs_mixed)
        mixed_logit = (w.view(-1, 1) * F.softmax(outputs_tgt, -1)
                       + (1 - w).view(-1, 1) * F.softmax(outputs_src, -1)).detach()
        loss_mix = F.kl_div(F.log_softmax(outputs_mixed, dim=-1), mixed_logit, reduction="batchmean")

        loss = self.lambda_anchor * (self.w_ce * loss_ce_src + self.w_mmd * loss_mmd + self.w_mix * loss_mix)
        loss.backward()

        # 3) capture anchor grads BEFORE the step, then step
        grads_S = {n: p.grad.detach().clone()
                   for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        optimizer.step()
        optimizer.zero_grad()

        # 4) harm-aware rewind toward the frozen source
        if self.score_type.lower() == "second_order":
            self._update_sg2_ema(grads_S)
        deltas = self._delta_params(model, prev_params)
        scores = self._score_groups(groups, grads_S, deltas)
        mask, _ = self._select_mask(scores, self.percentile, self.eps)
        self._blend_groups_toward_source(model, self.src_model, groups, scores, mask)
        model.zero_grad(set_to_none=True)

    # ------------------- backbone splitting -------------------
    def _build_layered_backbones(self):
        a, m, s = self.arch_name, self.model, self.src_model
        if a == "Hendrycks2020AugMix_ResNeXt":
            self.layer_1_src = nn.Sequential(s.conv_1_3x3, s.bn_1, s.stage_1)
            self.layer_2_src = nn.Sequential(s.stage_2)
            self.layer_3_src = nn.Sequential(s.stage_3)
            self.layer_1 = nn.Sequential(m.conv_1_3x3, m.bn_1, m.stage_1)
            self.layer_2 = nn.Sequential(m.stage_2)
            self.layer_3 = nn.Sequential(m.stage_3)
            self.classifier_head = nn.Sequential(m.avgpool, nn.Flatten(), m.classifier)
            self._n_stages = 3
        elif a == "resnet50":
            self.layer_1_src = nn.Sequential(s.normalize, s.model.conv1, s.model.bn1, s.model.relu, s.model.maxpool, s.model.layer1)
            self.layer_2_src = nn.Sequential(s.model.layer2)
            self.layer_3_src = nn.Sequential(s.model.layer3)
            self.layer_4_src = nn.Sequential(s.model.layer4)
            self.layer_1 = nn.Sequential(m.normalize, m.model.conv1, m.model.bn1, m.model.relu, m.model.maxpool, m.model.layer1)
            self.layer_2 = nn.Sequential(m.model.layer2)
            self.layer_3 = nn.Sequential(m.model.layer3)
            self.layer_4 = nn.Sequential(m.model.layer4)
            self.classifier_head = nn.Sequential(m.model.avgpool, nn.Flatten(), m.model.fc)
            self._n_stages = 4
        elif a == "Standard":
            self.layer_1_src = nn.Sequential(s.conv1, s.block1)
            self.layer_2_src = nn.Sequential(s.block2)
            self.layer_3_src = nn.Sequential(s.block3)
            self.layer_1 = nn.Sequential(m.conv1, m.block1)
            self.layer_2 = nn.Sequential(m.block2)
            self.layer_3 = nn.Sequential(m.block3)
            self.classifier_head = nn.Sequential(m.bn1, m.relu, nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten(), m.fc)
            self._n_stages = 3
        else:
            raise NotImplementedError(f"[DO-ALL] unsupported arch for backbone split: {a}")

    def _extract_test_features_src(self, x):
        """Frozen-source deep features (last stage, global-average-pooled)."""
        if self._n_stages == 4:
            f = self.layer_4_src(self.layer_3_src(self.layer_2_src(self.layer_1_src(x))))
        else:
            f = self.layer_3_src(self.layer_2_src(self.layer_1_src(x)))
        return f.mean(dim=(2, 3))

    def _extract_anchor_feature(self):
        """Pre-cache deep features of the whole DD anchor set (frozen source)."""
        with torch.no_grad():
            imgs_src = self.batch[0]
            feats = []
            chunk = self.num_classes
            for i in range(self.ipc):
                f = self._extract_test_features_src(imgs_src[chunk * i:chunk * (i + 1)].to(self.device))
                feats.append(f)
            self.total_feat = torch.cat(feats, 0)
        logger.info(f"[DO-ALL] anchor features cached: {tuple(self.total_feat.shape)}")

    def _anchor_forward_and_mmd(self, imgs_src, imgs_test):
        if self._n_stages == 4:
            f1s = self.layer_1(imgs_src);  f1t = self.layer_1(imgs_test)
            f2s = self.layer_2(f1s);       f2t = self.layer_2(f1t)
            f3s = self.layer_3(f2s);       f3t = self.layer_3(f2t)
            f4s = self.layer_4(f3s);       f4t = self.layer_4(f3t)
            outputs_src = self.classifier_head(f4s)
            outputs_tgt = self.classifier_head(f4t)
            loss_mmd = (mmd_loss(f1s.mean(dim=(2, 3)), f1t.mean(dim=(2, 3)))
                        + mmd_loss(f2s.mean(dim=(2, 3)), f2t.mean(dim=(2, 3)))
                        + mmd_loss(f3s.mean(dim=(2, 3)), f3t.mean(dim=(2, 3)))
                        + mmd_loss(f4s.mean(dim=(2, 3)), f4t.mean(dim=(2, 3))))
        else:
            f1s = self.layer_1(imgs_src);  f1t = self.layer_1(imgs_test)
            f2s = self.layer_2(f1s);       f2t = self.layer_2(f1t)
            f3s = self.layer_3(f2s);       f3t = self.layer_3(f2t)
            outputs_src = self.classifier_head(f3s)
            outputs_tgt = self.classifier_head(f3t)
            loss_mmd = (mmd_loss(f1s.mean(dim=(2, 3)), f1t.mean(dim=(2, 3)))
                        + mmd_loss(f2s.mean(dim=(2, 3)), f2t.mean(dim=(2, 3)))
                        + mmd_loss(f3s.mean(dim=(2, 3)), f3t.mean(dim=(2, 3))))
        return loss_mmd, outputs_src, outputs_tgt

    # ------------------- harm-aware rewind utilities -------------------
    @torch.no_grad()
    def _delta_params(self, model_now, prev) -> Dict[str, torch.Tensor]:
        return {n: (p.detach() - prev[n]).to(p.dtype)
                for n, p in model_now.named_parameters() if p.requires_grad and n in prev}

    def _update_sg2_ema(self, grads, momentum: float = 0.95):
        for n, g in grads.items():
            sg2 = g.detach() ** 2
            self._ema_sg2[n] = sg2 if n not in self._ema_sg2 else momentum * self._ema_sg2[n] + (1 - momentum) * sg2

    def _score_groups(self, groups, grads_S, deltas) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        use_second = (self.score_type.lower() == "second_order")
        for gname, plist in groups:
            s_val = 0.0
            for pname, _p in plist:
                if pname not in grads_S or pname not in deltas:
                    continue
                g, d = grads_S[pname], deltas[pname]
                term = (g * d).sum().item()
                if use_second:
                    h_diag = self._ema_sg2.get(pname, torch.zeros_like(d))
                    term += 0.5 * (h_diag * (d ** 2)).sum().item()
                s_val += max(0.0, term)
            scores[gname] = s_val
        return scores

    @staticmethod
    def _select_mask(scores: Dict[str, float], percentile: float, eps: float = 1e-8):
        if not scores:
            return [], 0.0
        vals = torch.tensor(list(scores.values()))
        pos_vals = vals[vals > 0]
        if pos_vals.numel() == 0:
            return [], 0.0
        thr = torch.quantile(pos_vals, torch.tensor(1.0 - percentile)).item()
        return [k for k, v in scores.items() if v > thr + eps], thr

    @torch.no_grad()
    def _blend_groups_toward_source(self, model, source_model, groups, scores, mask):
        if not mask:
            return
        mvals = torch.tensor([scores[g] for g in mask], dtype=torch.float32)
        mmin, mmax = float(mvals.min().item()), float(mvals.max().item())
        denom = max(self.eps, (mmax - mmin))
        src_named = dict(source_model.named_parameters())
        for gname, plist in groups:
            if gname not in mask:
                continue
            s = (scores[gname] - mmin) / denom
            beta = self.beta_max * torch.sigmoid(torch.tensor(self.beta_scale * (s - 0.5))).item()
            for pname, p in plist:
                src_p = src_named.get(pname, None)
                if src_p is not None:
                    p.copy_((1 - beta) * p + beta * src_p)

    @staticmethod
    def _named_param_groups(model, group_by: str = "layer"):
        groups: List[Tuple[str, List[Tuple[str, nn.Parameter]]]] = []
        if group_by == "param":
            return [(n, [(n, p)]) for n, p in model.named_parameters() if p.requires_grad]
        buckets: Dict[str, List[Tuple[str, nn.Parameter]]] = {}
        for full_name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            layer_name = full_name.rsplit(".", 1)[0] if "." in full_name else full_name
            buckets.setdefault(layer_name, []).append((full_name, p))
        return sorted(buckets.items(), key=lambda kv: kv[0])
