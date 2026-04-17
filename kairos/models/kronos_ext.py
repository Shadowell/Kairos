"""Kronos 的外生变量扩展版本（**方案 A**：保留 Tokenizer，旁路融合）。

设计要点
--------
1. **Tokenizer 不动**，仍然吃 6 维 OHLCV+amount；兼容 ``NeoQuasar/Kronos-*`` 预训练权重。
2. 在 Transformer 输入端加一条 **exog 通道**（``Linear -> SiLU -> Linear``），
   与 token embedding、time embedding **相加** 融合，可选择用 ``gate`` 做门控。
3. 额外提供一个 **ReturnHead**（方案 C）：分位回归头，直接输出未来 h 步收益的分位数，
   用于训练 ``CE + pinball`` 联合损失。
4. 仍然兼容 ``KronosPredictor`` 接口：只在 ``forward / decode_s1`` 多接一个 ``exog`` 参数。
5. 保留 ``PyTorchModelHubMixin``，``from_pretrained`` / ``push_to_hub`` 直接可用。

> **重要**：exog 的维度必须与 ``data_pipeline/build_features.py::EXOG_COLS`` 完全一致。
> 当前默认 ``n_exog = 32``。如果你扩展了特征，记得同步修改两边并重训。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from kairos.vendor.kronos.kronos import Kronos
from kairos.vendor.kronos.module import (
    DependencyAwareLayer,
    DualHead,
    HierarchicalEmbedding,
    RMSNorm,
    TemporalEmbedding,
    TransformerBlock,
)


# ---------------------------------------------------------------------------
# 外生通道
# ---------------------------------------------------------------------------
class ExogenousEncoder(nn.Module):
    """Linear -> SiLU -> Linear -> RMSNorm + 可学习 gate。"""

    def __init__(self, n_exog: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_exog, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = RMSNorm(d_model)
        # 零初始化门控，等价于第一步退化为原版 Kronos
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, exog: torch.Tensor) -> torch.Tensor:
        # exog: [B, T, n_exog]
        h = self.norm(self.proj(exog))
        return torch.tanh(self.gate) * h  # 门控 ∈ (-1, 1)


# ---------------------------------------------------------------------------
# 分位回归头（方案 C）
# ---------------------------------------------------------------------------
class QuantileReturnHead(nn.Module):
    """从 transformer 末层 hidden 预测未来 h 步收益的 n 个分位。"""

    def __init__(self, d_model: int, horizon: int = 1, n_quantiles: int = 9):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.norm = RMSNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, horizon * n_quantiles),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, d_model]  →  [B, T, horizon, n_quantiles]
        B, T, _ = hidden.shape
        out = self.fc(self.norm(hidden))
        return out.view(B, T, self.horizon, self.n_quantiles)

    @staticmethod
    def pinball_loss(
        pred: torch.Tensor,        # [B, T, h, q]
        target: torch.Tensor,      # [B, T, h]
        quantiles: torch.Tensor,   # [q]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = target.unsqueeze(-1)                         # [B,T,h,1]
        q = quantiles.view(1, 1, 1, -1)                       # [1,1,1,q]
        diff = target - pred                                  # [B,T,h,q]
        loss = torch.maximum(q * diff, (q - 1) * diff)
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).unsqueeze(-1)
            return loss.sum() / (mask.sum() + 1e-9)
        return loss.mean()


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class KronosWithExogenous(nn.Module, PyTorchModelHubMixin):
    """Kronos + 外生变量旁路通道 + 可选分位回归头。

    兼容 ``Kronos`` 的所有构造参数，额外新增:
        n_exog (int): 外生变量维度（与 EXOG_COLS 对齐，默认 32）
        use_return_head (bool): 是否启用分位回归头
        return_horizon (int): 回归头预测未来多少步
        n_quantiles (int): 回归头分位数量
    """

    def __init__(
        self,
        s1_bits: int,
        s2_bits: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        ffn_dropout_p: float,
        attn_dropout_p: float,
        resid_dropout_p: float,
        token_dropout_p: float,
        learn_te: bool,
        n_exog: int = 32,
        use_return_head: bool = True,
        return_horizon: int = 5,
        n_quantiles: int = 9,
    ):
        super().__init__()
        # 保存所有参数（便于 from_pretrained 还原）
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.learn_te = learn_te
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p
        self.n_exog = n_exog
        self.use_return_head = use_return_head
        self.return_horizon = return_horizon
        self.n_quantiles = n_quantiles
        self.s1_vocab_size = 2 ** s1_bits

        # ---- Kronos 原版结构（保持一致以便加载预训练权重）----
        self.token_drop = nn.Dropout(token_dropout_p)
        self.embedding = HierarchicalEmbedding(s1_bits, s2_bits, d_model)
        self.time_emb = TemporalEmbedding(d_model, learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim,
                             ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.dep_layer = DependencyAwareLayer(d_model)
        self.head = DualHead(s1_bits, s2_bits, d_model)

        # ---- 新增的外生通道 ----
        self.exog_encoder = ExogenousEncoder(n_exog, d_model,
                                             dropout=ffn_dropout_p)

        # ---- 可选的分位回归头 ----
        if use_return_head:
            self.return_head = QuantileReturnHead(
                d_model, horizon=return_horizon, n_quantiles=n_quantiles)
        else:
            self.return_head = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0,
                            std=self.embedding.d_model ** -0.5)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    # 从官方 Kronos 预训练权重初始化（只对公共子图）
    # ------------------------------------------------------------------
    @classmethod
    def from_kronos_pretrained(
        cls,
        kronos_repo_or_path: str,
        n_exog: int = 32,
        use_return_head: bool = True,
        return_horizon: int = 5,
        n_quantiles: int = 9,
    ) -> "KronosWithExogenous":
        """加载 NeoQuasar 官方权重，并把 exog/return_head 随机初始化。"""
        base: Kronos = Kronos.from_pretrained(kronos_repo_or_path)
        model = cls(
            s1_bits=base.s1_bits,
            s2_bits=base.s2_bits,
            n_layers=base.n_layers,
            d_model=base.d_model,
            n_heads=base.n_heads,
            ff_dim=base.ff_dim,
            ffn_dropout_p=base.ffn_dropout_p,
            attn_dropout_p=base.attn_dropout_p,
            resid_dropout_p=base.resid_dropout_p,
            token_dropout_p=base.token_dropout_p,
            learn_te=base.learn_te,
            n_exog=n_exog,
            use_return_head=use_return_head,
            return_horizon=return_horizon,
            n_quantiles=n_quantiles,
        )
        # 迁移公共层的权重
        shared_sd = {}
        base_sd = base.state_dict()
        own_sd = model.state_dict()
        for k, v in base_sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                shared_sd[k] = v
        missing = [k for k in own_sd if k not in shared_sd]
        model.load_state_dict(shared_sd, strict=False)
        print(f"[KronosWithExogenous] 载入 {len(shared_sd)}/{len(own_sd)} 层；"
              f"新初始化 {len(missing)} 层（exog / return_head）")
        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _embed(self, s1_ids, s2_ids, stamp=None, exog=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        if exog is not None:
            x = x + self.exog_encoder(exog)
        return self.token_drop(x)

    def forward(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: Optional[torch.Tensor] = None,
        exog: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = False,
        s1_targets: Optional[torch.Tensor] = None,
    ):
        x = self._embed(s1_ids, s2_ids, stamp, exog)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        hidden = self.norm(x)

        s1_logits = self.head(hidden)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1 = torch.multinomial(
                probs.view(-1, self.s1_vocab_size), 1
            ).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1)

        x2 = self.dep_layer(hidden, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)

        quantiles = self.return_head(hidden) if self.return_head is not None else None
        return s1_logits, s2_logits, quantiles

    # 供 auto_regressive_inference 使用
    def decode_s1(self, s1_ids, s2_ids, stamp=None, exog=None, padding_mask=None):
        x = self._embed(s1_ids, s2_ids, stamp, exog)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        hidden = self.norm(x)
        return self.head(hidden), hidden

    def decode_s2(self, context, s1_ids, padding_mask=None):
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)

    # ------------------------------------------------------------------
    # 渐进解冻 helper
    # ------------------------------------------------------------------
    def freeze_backbone(self, unfreeze_last_n: int = 2):
        """冻结 embedding + 前 (n_layers - unfreeze_last_n) 层 transformer，
        只训 exog_encoder、最后几层、norm、head、return_head。"""
        for p in self.embedding.parameters():
            p.requires_grad = False
        for p in self.time_emb.parameters():
            p.requires_grad = False
        n_freeze = max(0, self.n_layers - unfreeze_last_n)
        for i, blk in enumerate(self.transformer):
            req = i >= n_freeze
            for p in blk.parameters():
                p.requires_grad = req
        # 其余（norm / head / dep_layer / exog_encoder / return_head）默认可训
        print(f"[freeze] 冻结 embedding + 前 {n_freeze} 层 transformer；"
              f"解冻最后 {self.n_layers - n_freeze} 层 + exog + heads")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
