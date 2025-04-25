"""
model.py  ─ Edge-based Attention Aggregation Network
==================================================
Komponentene her bygger på idéer fra:

• Particle Transformer for Jet Tagging
  https://arxiv.org/abs/2202.03772
• ParticleNet (jet-tagging via particle clouds)
  https://doi.org/10.1103/physrevd.101.056019
• Dynamic Graph CNN for Point Clouds
  https://arxiv.org/abs/1801.07829
• Graph Attention Networks
  https://arxiv.org/abs/1710.10903
• Squeeze-and-Excitation Networks
  https://arxiv.org/pdf/1709.01507
• Deep Residual Learning
  https://arxiv.org/abs/1512.03385
• PointNet++ (multiskalafusjon)
  https://arxiv.org/abs/1706.02413
• Graph U-Net (hard top-k-pooling)
  https://arxiv.org/abs/1905.05178
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
def batch_distance_matrix_general(A: torch.Tensor,
                                  B: torch.Tensor) -> torch.Tensor:
    """
    Return squared Euclidean distance matrices for every batch element.

    Parameters
    ----------
    A, B : (N, P, D) tensor
        N – batch size, P – number of points, D – coordinate dims.

    Returns
    -------
    (N, P, P) tensor
        D[n, i, j] = ‖A[n, i] − B[n, j]‖²
    """
    r_A = torch.sum(A * A, dim=2, keepdim=True)        # ‖A‖²
    r_B = torch.sum(B * B, dim=2, keepdim=True)        # ‖B‖²
    m   = torch.bmm(A, B.transpose(1, 2))              # A · Bᵀ
    return r_A - 2 * m + r_B.transpose(1, 2)           # ‖A − B‖²


def knn(topk_idx: torch.Tensor,
        feats: torch.Tensor) -> torch.Tensor:
    """
    Gather K-nearest-neighbour features.

    Parameters
    ----------
    topk_idx : (N, P, K) long tensor
        Indices of neighbours per point.
    feats    : (N, P, C) float tensor
        Feature matrix.

    Returns
    -------
    (N, P, K, C) tensor with neighbour features.
    """
    N, P, K = topk_idx.shape
    batch   = torch.arange(N, device=feats.device).view(N, 1, 1).expand(N, P, K)
    return feats[batch, topk_idx]


# ---------------------------------------------------------------------
class EdgeConvBlock(nn.Module):
    r"""
    EdgeConv-basert blokk med opsjonell multi-head attention-pooling,
    SE-gating og radialavstand som ekstra kantattributt.

    • Kantfunksjonen følger Dynamic Graph CNN:
      h_ij = [x_i , x_j − x_i , r_ij]
    • Attention-pooling implementerer Graph Attention Networks-mekanismen.
    • SE-gating fra Squeeze-and-Excitation gir kanalre-vektlegging.
    • Residual shortcut i stil med ResNet.
    """

    def __init__(self, K: int, in_channels: int, channels: tuple[int, ...],
                 *,
                 with_bn: bool = True,
                 activation: nn.Module = nn.ReLU(),
                 pooling: str = 'attention',           # {'attention','average','max'}
                 num_heads: int = 1,
                 init_tau: float = 1.0,
                 se_ratio: int = 4,
                 attn_hidden: int = 32,
                 use_radial: bool = False):
        super().__init__()
        assert pooling in ('average', 'max', 'attention')
        assert channels[-1] % num_heads == 0

        self.K, self.pooling, self.H = K, pooling, num_heads
        self.use_radial = use_radial
        self.out_channels = channels[-1]
        self.head_dim = self.out_channels // self.H
        self.activation, self.with_bn = activation, with_bn

        # -------- kant-MLP --------
        edge_in = 2 * in_channels + (1 if use_radial else 0)
        conv_in = edge_in
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for out_c in channels:
            self.convs.append(nn.Conv2d(conv_in, out_c, 1,
                                        bias=not with_bn))
            if with_bn:
                self.bns.append(nn.BatchNorm2d(out_c))
            conv_in = out_c

        # -------- shortcut --------
        self.shortcut_conv = nn.Conv1d(in_channels, self.out_channels, 1,
                                       bias=not with_bn)
        if with_bn:
            self.shortcut_bn = nn.BatchNorm1d(self.out_channels)

        # -------- attention --------
        if pooling == 'attention':
            self.tau = nn.Parameter(torch.tensor(float(init_tau)))
            self.attn_mlp = nn.Sequential(
                nn.Conv2d(edge_in, attn_hidden, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(attn_hidden, self.H, 1)
            )

        # -------- SE-gating --------
        hidden = max(1, self.out_channels // se_ratio)
        self.se_fc1 = nn.Linear(self.out_channels, hidden, bias=False)
        self.se_fc2 = nn.Linear(hidden, self.out_channels, bias=False)

    # .................................................................
    def forward(self,
                points: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        points   : (N, P, D) tensor  – koordinater (brukes kun til nabo-søket)
        features : (N, P, C_in) tensor

        Returns
        -------
        (N, P, C_out) tensor  – oppdaterte punkt-features
        """
        N, P, _ = features.shape

        # dynamisk K-NN (fra Dynamic Graph CNN)
        dist = batch_distance_matrix_general(points, points)
        _, k_idx = torch.topk(-dist, k=self.K + 1, dim=2)
        k_idx = k_idx[:, :, 1:]                      # fjern selvideks

        knn_feat = knn(k_idx, features)             # nabo-features
        center   = features.unsqueeze(2).expand(-1, -1, self.K, -1)
        edge     = torch.cat([center, knn_feat - center], dim=-1)

        if self.use_radial:
            knn_pts = knn(k_idx, points)
            r = torch.norm(knn_pts - points.unsqueeze(2), dim=-1, keepdim=True)
            edge = torch.cat([edge, r], dim=-1)

        x = edge.permute(0, 3, 1, 2)                # (N, C, P, K)

        # ---------- attention-vekter ----------
        if self.pooling == 'attention':
            w = self.attn_mlp(x).view(N, self.H, P, self.K)
            w = torch.softmax(w / self.tau, dim=-1)

        # ---------- kant-MLP ----------
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if self.with_bn:
                x = self.bns[i](x)
            x = self.activation(x)

        # ---------- pooling over naboer ----------
        if self.pooling == 'attention':
            x = x.view(N, self.H, self.head_dim, P, self.K)
            x = (x * w.unsqueeze(2)).sum(dim=4).reshape(N, self.out_channels, P)
        elif self.pooling == 'average':
            x = x.mean(dim=3)
        else:  # max
            x = x.max(dim=3).values

        # ---------- SE-gating ----------
        s = torch.sigmoid(
            self.se_fc2(F.relu(self.se_fc1(x.mean(dim=2)), inplace=True))
        )
        x = x * s.unsqueeze(2)

        # ---------- residual ----------
        sc = self.shortcut_conv(features.permute(0, 2, 1))
        if self.with_bn:
            sc = self.shortcut_bn(sc)
        sc = self.activation(sc)
        return (x + sc).permute(0, 2, 1)            # (N, P, C_out)


# ---------------------------------------------------------------------
class EAAN(nn.Module):
    """
    Edge-based Attention Aggregation Network (EAAN)

    Hierarkisk arkitektur for partikkel-/punkt-skyer:
        • flere EdgeConv-blokker med dynamisk nabo-søking
        • valgfri multiskala-fusjon av blokkutganger (som PointNet++)
        • hard top-M pooling (fra Graph U-Net) før global aggregering
        • fullt koblet hode for flerklasse-klassifisering
    """

    # .................................................................

    def __init__(self, input_dims: int, num_classes: int,
                 *,
                 conv_params=((16, (64, 64, 64)),
                              (16, (128, 128, 128)),
                              (16, (256, 256, 256)),
                              (16, (256, 256, 512))),
                 fc_params=((512, 0.2),),
                 conv_pooling: str = 'attention',
                 num_heads: int = 4, init_tau: float = 0.7, se_ratio: int = 4,
                 attn_hidden: int = 32, use_radial: bool = True,
                 top_M: int = 32, use_fusion: bool = True,
                 use_fts_bn: bool = True, use_counts: bool = True):
        super().__init__()
        self.use_fts_bn, self.use_fusion, self.use_counts = \
            use_fts_bn, use_fusion, use_counts
        self.top_M = top_M

        # -------------------------------------------------- input BN
        if use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        # -------------------------------------------------- EdgeConv-blokker
        self.edge_convs = nn.ModuleList()
        for i, (K, chs) in enumerate(conv_params):
            in_ch = input_dims if i == 0 else conv_params[i - 1][1][-1]
            self.edge_convs.append(
                EdgeConvBlock(K, in_ch, chs,
                              pooling=conv_pooling,
                              num_heads=num_heads,
                              init_tau=init_tau,
                              se_ratio=se_ratio,
                              attn_hidden=attn_hidden,
                              use_radial=use_radial)
            )

        # -------------------------------------------------- multiskala-fusjon
        if use_fusion:
            total = sum(c[-1] for _, c in conv_params)
            fus   = max(128, min(1024, (total // 128) * 128))
            self.fusion_block = nn.Sequential(
                nn.Conv1d(total, fus, 1, bias=False),
                nn.BatchNorm1d(fus),
                nn.ReLU())
        else:
            fus = conv_params[-1][1][-1]

        # -------------------------------------------------- fullt koblet hode
        layers, in_fc = [], fus
        for units, drop in fc_params:
            layers += [nn.Linear(in_fc, units), nn.ReLU()]
            if drop:
                layers.append(nn.Dropout(drop))
            in_fc = units
        self.fc       = nn.Sequential(*layers)
        self.fc_final = nn.Linear(in_fc, num_classes)

    # .................................................................

    def forward(self,
                points: torch.Tensor,
                features: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        points   : (N, P, D) – relative (Δη, Δφ) eller xyz-koordinater
        features : (N, P, C_in) – partikkelfeatures
        mask     : (N, P, 1)   – 1 = gyldig partikkel, 0 = *padding*
                                 (beregnes automatisk hvis None)

        Returns
        -------
        (N, num_classes) logits
        """
        if mask is None:
            mask = (features.abs().sum(dim=2, keepdim=True) != 0).float()

        coord_shift = (mask == 0).float() * 1e9  # hindrer at padding tas som nabo

        if self.use_fts_bn:
            features = self.bn_fts(features.permute(0, 2, 1)) \
                       .permute(0, 2, 1) * mask

        # ---------------- EdgeConv-stabel -----------------
        outs = []
        for i, edge in enumerate(self.edge_convs):
            pts_in   = points if i == 0 else features
            features = edge(pts_in + coord_shift, features) * mask
            if self.use_fusion:
                outs.append(features)

        # ---------------- multiskala-fusjon ---------------
        if self.use_fusion:
            fused    = torch.cat(outs, 2).permute(0, 2, 1)
            features = self.fusion_block(fused).permute(0, 2, 1) * mask

        # ---------------- hard top-M pooling --------------
        if self.top_M and self.top_M < features.size(1):
            with torch.no_grad():
                score = features.norm(dim=2)            # Graph U-Net score
                _, idx = torch.topk(score, self.top_M, dim=1)
            row = torch.arange(features.size(0), device=features.device) \
                      .unsqueeze(1).expand_as(idx)
            features = features[row, idx, :]
            mask     = mask[row, idx, :]

        # ---------------- global aggregering --------------
        pooled = (features.sum(dim=1) / mask.sum(dim=1)) \
                 if self.use_counts else features.mean(dim=1)

        x = self.fc(pooled)
        return self.fc_final(x)
