"""
Machado K3 — versão separada do gerador machadiano usando a arquitetura do
lançamento do Kimi K3 (Moonshot AI, jul/2026), reduzida para escala de estudo.

O que foi trazido do Kimi K3 (referências no final do arquivo):

1.  KDA (Kimi Delta Attention): atenção linear com delta rule e *gate por canal*
    (decaimento diferente para cada dimensão do estado). Substitui o softmax na
    maioria das camadas.
2.  Hibridização 3:1 — três camadas KDA para cada camada de Gated MLA, com a
    última camada sempre global (MLA).
3.  NoPE — nenhuma codificação posicional (nem RoPE, nem embedding aprendido).
    A posição vem implicitamente do decaimento recorrente do KDA.
4.  Gated MLA — atenção global com compressão q-LoRA / kv-LoRA e gate sigmoide
    full-rank na saída.
5.  AttnRes (Attention Residuals) — cada camada lê uma combinação por softmax
    das saídas de todas as camadas anteriores, em vez do somatório com pesos
    unitários do residual clássico.
6.  Stable LatentMoE — roteamento no espaço latente (d/2), top-k de N experts,
    experts compartilhados, RMSNorm no agregado.
7.  SiTU-GLU — ativação com soft-cap duplo por tanh, limitando a magnitude da
    saída (|f| <= beta1*beta2) em vez do SwiGLU ilimitado.
8.  Quantile Balancing — balanceamento de carga sem loss auxiliar, via bias por
    expert derivado dos quantis dos scores do roteador.

As proporções (3:1, latente = d/2, inter ~ 0.86*d, beta1=4/beta2=25, 2 shared
experts) seguem o relatório técnico; apenas as dimensões absolutas foram
reduzidas para caber num corpus de ~11 MB.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuração
# ============================================================================

@dataclass
class KimiK3Config:
    vocab_size: int = 2000
    block_size: int = 256

    n_embd: int = 320
    n_layer: int = 8            # 8 camadas => 6 KDA + 2 MLA (padrão 3:1)
    mla_every: int = 4          # a cada 4 camadas uma é Gated MLA (a última inclusive)

    # --- KDA ---
    kda_heads: int = 5
    kda_head_dim: int = 64
    kda_conv_size: int = 4      # short conv causal antes de q/k/v
    kda_chunk_size: int = 64    # tamanho do chunk na forma chunkwise
    kda_max_decay: float = 0.25  # |log alpha| máximo por passo (estabilidade numérica)

    # --- Gated MLA ---
    mla_heads: int = 5
    mla_head_dim: int = 64
    q_lora_rank: int = 192
    kv_lora_rank: int = 128

    # --- Stable LatentMoE ---
    n_experts: int = 32
    n_experts_active: int = 4
    n_shared_experts: int = 2
    moe_latent_dim: int = 160   # = n_embd // 2, como no K3 (7168 -> 3584)
    expert_inter: int = 160     # ~0.86 * latente, arredondado
    shared_inter: int = 288     # ~0.86 * n_embd (por expert compartilhado)
    router_bias_rate: float = 1e-3   # taxa de atualização do Quantile Balancing

    # --- SiTU-GLU ---
    situ_beta1: float = 4.0
    situ_beta2: float = 25.0

    use_attn_res: bool = True   # False => residual clássico (soma com peso 1)
    dropout: float = 0.0

    def layer_types(self) -> List[str]:
        """Padrão 3:1 (KDA,KDA,KDA,MLA) com a última camada sempre global."""
        types = ["mla" if (i + 1) % self.mla_every == 0 else "kda"
                 for i in range(self.n_layer)]
        types[-1] = "mla"
        return types


# ============================================================================
# Blocos básicos
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


class SiTUGLU(nn.Module):
    """Sigmoid-Tanh Unit GLU (Kimi K3).

        f(x) = down[ b1*tanh(Wg x / b1) * sigmoid(Wg x) * b2*tanh(Wu x / b2) ]

    Perto da origem se comporta como SwiGLU; longe dela satura, com limite
    absoluto |ativação| <= b1*b2 (100 com b1=4, b2=25). É o que impede os
    outliers de ativação que estouram em baixa precisão.
    """

    def __init__(self, dim, inter, out_dim=None, beta1=4.0, beta2=25.0):
        super().__init__()
        out_dim = out_dim or dim
        self.beta1 = beta1
        self.beta2 = beta2
        self.gate = nn.Linear(dim, inter, bias=False)
        self.up = nn.Linear(dim, inter, bias=False)
        self.down = nn.Linear(inter, out_dim, bias=False)

    def forward(self, x):
        g = self.gate(x)
        act = self.beta1 * torch.tanh(g / self.beta1) * torch.sigmoid(g)
        u = self.beta2 * torch.tanh(self.up(x) / self.beta2)
        return self.down(act * u)


class ShortConv(nn.Module):
    """Convolução depthwise causal (kernel 4) aplicada a q/k/v do KDA."""

    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim,
                              padding=kernel_size - 1, bias=False)

    def forward(self, x):                      # (B, T, D)
        T = x.size(1)
        y = self.conv(x.transpose(1, 2))[..., :T]
        return y.transpose(1, 2)


class AttnRes(nn.Module):
    """Attention Residuals (Kimi Team, arXiv:2603.15031).

    Em vez de acumular as saídas das camadas anteriores com peso fixo 1, cada
    camada agrega essas saídas com pesos softmax dependentes do token:

        h_l = sum_i alpha_{i->l} v_i,
        alpha_{i->l} = softmax_i( w_l . RMSNorm(v_i) / sqrt(d) )

    `w_l` (pseudo-query) é inicializado em ZERO — condição destacada no paper —
    de modo que no início do treino os pesos são uniformes e a rede parte de um
    comportamento equivalente ao residual médio.
    """

    def __init__(self, dim, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.norm = RMSNorm(dim)
            self.w = nn.Parameter(torch.zeros(dim))
            self.scale = dim ** -0.5

    def forward(self, sources: List[torch.Tensor]) -> torch.Tensor:
        if not self.enabled:
            out = sources[0]
            for s in sources[1:]:
                out = out + s
            return out
        V = torch.stack(sources, dim=2)                 # (B, T, n, C)
        logits = (self.norm(V) @ self.w) * self.scale   # (B, T, n)
        alpha = torch.softmax(logits, dim=-1)
        return (alpha.unsqueeze(-1) * V).sum(dim=2)


# ============================================================================
# Kimi Delta Attention
# ============================================================================

def delta_rule_recurrent(q, k, v, beta, log_alpha, state=None):
    """Referência sequencial da recorrência do KDA (também usada no cache).

        S_t = (I - beta_t k_t k_t^T) diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
        o_t = q_t^T S_t

    `alpha_t` é um vetor (uma taxa de esquecimento por canal da chave) — é
    exatamente aí que o KDA difere do Gated DeltaNet, que usa um escalar.

    q, k:       (B, T, H, Dk)
    v:          (B, T, H, Dv)
    beta:       (B, T, H)
    log_alpha:  (B, T, H, Dk)
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, dtype=q.dtype, device=q.device) if state is None else state
    outs = []
    for t in range(T):
        a = torch.exp(log_alpha[:, t])                      # (B, H, Dk)
        k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]
        S = S * a.unsqueeze(-1)                             # diag(alpha) S
        kS = (k_t.unsqueeze(-1) * S).sum(dim=-2)            # (B, H, Dv)
        u = (v_t - kS) * beta[:, t].unsqueeze(-1)           # pseudo-valor
        S = S + k_t.unsqueeze(-1) * u.unsqueeze(-2)
        outs.append((q_t.unsqueeze(-1) * S).sum(dim=-2))
    return torch.stack(outs, dim=1), S


def delta_rule_chunkwise(q, k, v, beta, log_alpha, chunk_size=64, state=None):
    """Forma chunkwise da mesma recorrência (paralela dentro do chunk).

    Desenvolvendo a recorrência dentro de um chunk com A_t = prod_{s<=t} alpha_s:

        S_t   = diag(A_t) ( S_0 + sum_{s<=t} diag(A_s)^-1 beta_s k_s u_s^T )
        u_t   = v_t - S_0^T kbar_t - sum_{s<t} beta_s (ktil_s . kbar_t) u_s
        o_t   = qbar_t S_0 + sum_{s<=t} beta_s (qbar_t . ktil_s) u_s

    com kbar = A ⊙ k, qbar = A ⊙ q e ktil = k / A. O sistema em `u` é
    triangular inferior unitário e é resolvido de uma vez por chunk.
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    C = min(chunk_size, T)
    pad = (C - T % C) % C
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        log_alpha = F.pad(log_alpha, (0, 0, 0, 0, 0, pad))
    Tp = q.size(1)
    N = Tp // C

    def to_chunks(x):                       # (B, T, H, D) -> (B, H, N, C, D)
        return x.view(B, N, C, H, -1).permute(0, 3, 1, 2, 4)

    q, k, v = to_chunks(q), to_chunks(k), to_chunks(v)
    la = to_chunks(log_alpha)
    beta = beta.view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)   # (B,H,N,C,1)

    cum = la.cumsum(dim=-2)                                          # log A_t
    exp_cum = torch.exp(cum)
    exp_inv = torch.exp(-cum)
    q_bar = q * exp_cum
    k_bar = k * exp_cum
    k_til = k * exp_inv

    idx = torch.arange(C, device=q.device)
    strict = (idx.unsqueeze(-1) > idx.unsqueeze(0))                  # t > s
    causal = (idx.unsqueeze(-1) >= idx.unsqueeze(0))                 # t >= s
    eye = torch.eye(C, device=q.device, dtype=q.dtype)

    # matrizes intra-chunk (independentes do estado que entra)
    A_u = (k_bar @ k_til.transpose(-1, -2)) * beta.transpose(-1, -2)  # [t,s]
    A_u = A_u.masked_fill(~strict, 0.0) + eye
    A_o = (q_bar @ k_til.transpose(-1, -2)) * beta.transpose(-1, -2)
    A_o = A_o.masked_fill(~causal, 0.0)

    S = torch.zeros(B, H, Dk, Dv, dtype=q.dtype, device=q.device) if state is None else state
    outs = []
    for n in range(N):
        kb, qb = k_bar[:, :, n], q_bar[:, :, n]
        rhs = v[:, :, n] - kb @ S                                     # (B,H,C,Dv)
        u = torch.linalg.solve_triangular(A_u[:, :, n], rhs, upper=False, unitriangular=True)
        outs.append(qb @ S + A_o[:, :, n] @ u)
        # estado ao fim do chunk: diag(A_C) S_0 + sum_s beta_s (A_C/A_s ⊙ k_s) u_s^T
        decay_end = torch.exp(cum[:, :, n, -1:] - cum[:, :, n])       # (B,H,C,Dk), <= 1
        S = exp_cum[:, :, n, -1].unsqueeze(-1) * S \
            + (k[:, :, n] * decay_end * beta[:, :, n]).transpose(-1, -2) @ u

    out = torch.stack(outs, dim=2)                                    # (B,H,N,C,Dv)
    out = out.permute(0, 2, 3, 1, 4).reshape(B, Tp, H, Dv)
    return out[:, :T], S


class KimiDeltaAttention(nn.Module):
    """Camada KDA: delta rule com gate por canal + short conv + gate de saída."""

    def __init__(self, cfg: KimiK3Config):
        super().__init__()
        self.cfg = cfg
        H, D = cfg.kda_heads, cfg.kda_head_dim
        self.n_head, self.head_dim = H, D
        inner = H * D

        self.q_proj = nn.Linear(cfg.n_embd, inner, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, inner, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, inner, bias=False)
        self.q_conv = ShortConv(inner, cfg.kda_conv_size)
        self.k_conv = ShortConv(inner, cfg.kda_conv_size)
        self.v_conv = ShortConv(inner, cfg.kda_conv_size)

        self.b_proj = nn.Linear(cfg.n_embd, H, bias=True)            # beta_t
        self.a_proj = nn.Linear(cfg.n_embd, inner, bias=True)        # gate por canal
        self.a_scale = nn.Parameter(torch.zeros(H, D))               # escala do decaimento

        self.g_proj = nn.Linear(cfg.n_embd, inner, bias=False)       # gate sigmoide de saída
        self.o_norm = RMSNorm(D)                                     # RMSNorm por head
        self.o_proj = nn.Linear(inner, cfg.n_embd, bias=False)

        # bias negativo => beta pequeno no início (edições suaves no estado)
        nn.init.constant_(self.b_proj.bias, -1.0)
        # bias positivo em a_proj => softplus grande => alpha longe de 1 é evitado
        nn.init.constant_(self.a_proj.bias, 1.0)

    def _log_alpha(self, x, B, T):
        raw = F.softplus(self.a_proj(x)).view(B, T, self.n_head, self.head_dim)
        scale = F.softplus(self.a_scale) + 1e-4
        la = -raw * scale
        return la.clamp(min=-self.cfg.kda_max_decay)

    def forward(self, x, state=None, use_chunkwise=True):
        B, T, _ = x.shape
        H, D = self.n_head, self.head_dim

        q = F.silu(self.q_conv(self.q_proj(x))).view(B, T, H, D)
        k = F.silu(self.k_conv(self.k_proj(x))).view(B, T, H, D)
        v = F.silu(self.v_conv(self.v_proj(x))).view(B, T, H, D)
        q = F.normalize(q, p=2.0, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2.0, dim=-1, eps=1e-6)

        beta = torch.sigmoid(self.b_proj(x))                         # (B, T, H)
        log_alpha = self._log_alpha(x, B, T)

        if use_chunkwise and T > 1:
            out, state = delta_rule_chunkwise(q, k, v, beta, log_alpha,
                                              self.cfg.kda_chunk_size, state)
        else:
            out, state = delta_rule_recurrent(q, k, v, beta, log_alpha, state)

        out = self.o_norm(out).reshape(B, T, H * D)
        out = out * torch.sigmoid(self.g_proj(x))                    # gate full-rank
        return self.o_proj(out), state


# ============================================================================
# Gated MLA (atenção global, NoPE)
# ============================================================================

class GatedMLA(nn.Module):
    """Multi-head Latent Attention com q-LoRA/kv-LoRA, NoPE e gate sigmoide.

    Sem RoPE e sem embedding posicional: a informação de ordem chega pelas
    camadas KDA, cujo decaimento já é sensível à posição. Isso é o que permite
    ao K3 extrapolar contexto sem reescalar frequências.
    """

    def __init__(self, cfg: KimiK3Config):
        super().__init__()
        H, D = cfg.mla_heads, cfg.mla_head_dim
        self.n_head, self.head_dim = H, D
        self.scale = D ** -0.5
        inner = H * D

        self.q_down = nn.Linear(cfg.n_embd, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up = nn.Linear(cfg.q_lora_rank, inner, bias=False)

        self.kv_down = nn.Linear(cfg.n_embd, cfg.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.k_up = nn.Linear(cfg.kv_lora_rank, inner, bias=False)
        self.v_up = nn.Linear(cfg.kv_lora_rank, inner, bias=False)

        self.g_proj = nn.Linear(cfg.n_embd, inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.n_head, self.head_dim

        q = self.q_up(self.q_norm(self.q_down(x))).view(B, T, H, D).transpose(1, 2)
        c_kv = self.kv_norm(self.kv_down(x))
        k = self.k_up(c_kv).view(B, T, H, D).transpose(1, 2)
        v = self.v_up(c_kv).view(B, T, H, D).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = self.dropout(torch.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, H * D)

        out = out * torch.sigmoid(self.g_proj(x))
        return self.o_proj(out)


# ============================================================================
# Stable LatentMoE + Quantile Balancing
# ============================================================================

class LatentMoE(nn.Module):
    """MoE que roteia no espaço latente (d/2) em vez do residual completo.

    Fluxo (igual ao do K3, em escala menor):
        x -(down)-> latente -> router sigmoide, top-k de N experts
        experts SiTU-GLU rodam DENTRO do latente
        agregado -> RMSNorm -> (up)-> d, somado aos experts compartilhados

    Balanceamento por Quantile Balancing: sem loss auxiliar. Para cada expert
    calculamos o quantil (1 - k/N) dos seus scores; o bias de seleção é movido
    na direção que iguala esses limiares — quem está sobrecarregado tem limiar
    alto e perde bias.
    """

    def __init__(self, cfg: KimiK3Config):
        super().__init__()
        self.cfg = cfg
        E, dl = cfg.n_experts, cfg.moe_latent_dim
        self.n_experts, self.top_k = E, cfg.n_experts_active

        self.down = nn.Linear(cfg.n_embd, dl, bias=False)
        self.up = nn.Linear(dl, cfg.n_embd, bias=False)
        self.agg_norm = RMSNorm(dl)
        self.router = nn.Linear(dl, E, bias=False)

        inter = cfg.expert_inter
        self.w_gate = nn.Parameter(torch.empty(E, dl, inter))
        self.w_up = nn.Parameter(torch.empty(E, dl, inter))
        self.w_down = nn.Parameter(torch.empty(E, inter, dl))
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=0.02)

        self.shared = nn.ModuleList([
            SiTUGLU(cfg.n_embd, cfg.shared_inter, beta1=cfg.situ_beta1, beta2=cfg.situ_beta2)
            for _ in range(cfg.n_shared_experts)
        ])

        self.register_buffer("expert_bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E))
        self.last_load: Optional[torch.Tensor] = None

    def _expert_ffn(self, h, e):
        b1, b2 = self.cfg.situ_beta1, self.cfg.situ_beta2
        g = h @ self.w_gate[e]
        act = b1 * torch.tanh(g / b1) * torch.sigmoid(g)
        u = b2 * torch.tanh((h @ self.w_up[e]) / b2)
        return (act * u) @ self.w_down[e]

    @torch.no_grad()
    def _quantile_balance(self, scores):
        """Atualiza o bias por expert a partir dos quantis dos scores."""
        q = max(0.0, min(1.0, 1.0 - self.top_k / self.n_experts))
        # limiar efetivo de seleção do expert (score + bias já aplicado)
        thr = torch.quantile(scores.float(), q, dim=0) + self.expert_bias
        target = thr.mean()
        self.expert_bias += self.cfg.router_bias_rate * (target - thr)
        self.expert_bias -= self.expert_bias.mean()

    def forward(self, x):
        B, T, C = x.shape
        h = self.down(x)
        flat = h.reshape(-1, h.size(-1))
        N = flat.size(0)

        scores = torch.sigmoid(self.router(flat))                     # (N, E)
        _, topi = torch.topk(scores + self.expert_bias, self.top_k, dim=-1)
        w = scores.gather(-1, topi)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.zeros_like(flat)
        counts = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            hit = (topi == e)
            if not bool(hit.any()):
                continue
            tok, slot = hit.nonzero(as_tuple=True)
            counts[e] = tok.numel()
            ye = self._expert_ffn(flat[tok], e) * w[tok, slot].unsqueeze(-1)
            out = out.index_add(0, tok, ye)

        out = self.agg_norm(out.view(B, T, -1))
        y = self.up(out)
        for sh in self.shared:
            y = y + sh(x)

        self.last_load = counts / max(1.0, N * self.top_k)
        if self.training:
            self._quantile_balance(scores.detach())
            self.load.mul_(0.99).add_(0.01 * self.last_load)
        return y


# ============================================================================
# Bloco e modelo
# ============================================================================

class BlockK3(nn.Module):
    def __init__(self, cfg: KimiK3Config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.res_attn = AttnRes(cfg.n_embd, cfg.use_attn_res)
        self.norm_attn = RMSNorm(cfg.n_embd)
        self.attn = KimiDeltaAttention(cfg) if layer_type == "kda" else GatedMLA(cfg)
        self.res_ffn = AttnRes(cfg.n_embd, cfg.use_attn_res)
        self.norm_ffn = RMSNorm(cfg.n_embd)
        self.moe = LatentMoE(cfg)

    def forward(self, sources: List[torch.Tensor]):
        h = self.res_attn(sources)
        if self.layer_type == "kda":
            a, _ = self.attn(self.norm_attn(h))
        else:
            a = self.attn(self.norm_attn(h))
        sources = sources + [a]

        h2 = self.res_ffn(sources)
        f = self.moe(self.norm_ffn(h2))
        return sources + [f]


class GPTMachadoK3(nn.Module):
    """Modelo completo no estilo Kimi K3, em escala de estudo."""

    def __init__(self, cfg: KimiK3Config):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        types = cfg.layer_types()
        self.layer_types = types

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([BlockK3(cfg, t) for t in types])
        self.res_out = AttnRes(cfg.n_embd, cfg.use_attn_res)
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight        # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None and not getattr(m, "_keep_bias", False):
                pass
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size:]
        # NoPE: nenhuma soma de embedding posicional aqui.
        sources = [self.tok_emb(idx)]
        for blk in self.blocks:
            sources = blk(sources)
        x = self.ln_f(self.res_out(sources))
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self, active_only=False):
        total = sum(p.numel() for p in self.parameters())
        if not active_only:
            return total
        cfg = self.cfg
        per_expert = 3 * cfg.moe_latent_dim * cfg.expert_inter
        inactive = (cfg.n_experts - cfg.n_experts_active) * per_expert * cfg.n_layer
        return total - inactive

    def expert_load(self):
        """Fração de tokens recebida por cada expert (média móvel, por camada)."""
        return [blk.moe.load.clone() for blk in self.blocks]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


# ============================================================================
# Referências (pesquisadas em ago/2026)
# ============================================================================
# Kimi K3 Technical Report (Kimi Team)      https://arxiv.org/abs/2607.24653
# Attention Residuals (Kimi Team)           https://arxiv.org/abs/2603.15031
# Kimi Linear / KDA                         https://arxiv.org/abs/2510.26692
# Notas de arquitetura (KDA:MLA 3:1, 896 experts top-16, latente d/2,
# SiTU-GLU b1=4/b2=25, Quantile Balancing, NoPE, gate sigmoide full-rank).
