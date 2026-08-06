"""
Treino do Machado K3 (arquitetura Kimi K3 em escala reduzida).

    python train_k3.py --iters 8000                 # treino padrão
    python train_k3.py --quick                      # smoke test (poucos passos)
    python train_k3.py --optimizer muon             # Per-Head Muon (recipe do K3)

Usa o mesmo corpus (machado.txt) e o mesmo tokenizer BPE (tokenizer_bpe.pkl)
do modelo V3, para que a comparação entre as versões seja justa.
"""

import argparse
import math
import os
import pickle
import re
import time

import torch

from model_k3 import KimiK3Config, GPTMachadoK3


# ============================================================================
# Tokenizer BPE (mesmo arquivo .pkl usado pelo V3)
# ============================================================================

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self._cache = {}

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vocab = data["vocab"]
        self.merges = data["merges"]
        return self

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _encode_piece(self, piece: str):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        tokens = list(piece.encode("utf-8"))
        while len(tokens) >= 2:
            pair = min(
                zip(tokens, tokens[1:]),
                key=lambda p: self.merges.get(p, float("inf")),
                default=None,
            )
            if pair is None or pair not in self.merges:
                break
            idx = self.merges[pair]
            merged, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    merged.append(idx)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        self._cache[piece] = tokens
        return tokens

    def encode(self, text: str):
        """Codifica com pré-tokenização por palavra (com cache).

        O algoritmo BPE é o mesmo do app; a divisão prévia em palavras serve só
        para tornar viável codificar os 11 MB do corpus em segundos.
        """
        out = []
        for piece in re.findall(r"\s*\S+|\s+", text):
            out.extend(self._encode_piece(piece))
        return out

    def decode(self, tokens):
        raw = b"".join(self.vocab.get(int(i), b"?") for i in tokens)
        return raw.decode("utf-8", errors="replace")


# ============================================================================
# Per-Head Muon (o K3 usa Muon por cabeça; AdamW fica para embeddings/normas)
# ============================================================================

def newton_schulz(G, steps=5, eps=1e-7):
    """Ortogonalização aproximada de G via iteração de Newton-Schulz (quíntica)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X


class Muon(torch.optim.Optimizer):
    """Muon com suporte a ortogonalização por cabeça / por expert.

    `heads > 1` divide a matriz (H*D, in) em H blocos e ortogonaliza cada bloco
    separadamente — é o "Per-Head Muon" citado no relatório do K3. Tensores 3D
    (os experts empilhados do MoE) já são tratados em batch.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, heads=1):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, heads=heads))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(group["momentum"]).add_(g)
                upd = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                heads = group["heads"]
                shape = upd.shape
                if upd.dim() == 2 and heads > 1 and shape[0] % heads == 0:
                    upd = upd.view(heads, shape[0] // heads, shape[1])
                o = newton_schulz(upd, group["ns_steps"]).view(shape)
                scale = math.sqrt(max(1.0, shape[-2] / shape[-1]))
                p.add_(o.type_as(p), alpha=-group["lr"] * scale)
        return loss


def build_optimizer(model, kind, lr, weight_decay=0.1):
    if kind == "adamw":
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.95))]

    # muon: matrizes do corpo do modelo no Muon, o resto no AdamW
    muon_heads, muon_plain, adam_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.dim() >= 2
        is_embed = "tok_emb" in n or "lm_head" in n
        is_router = "router" in n
        if is_matrix and not is_embed and not is_router:
            head_like = any(t in n for t in ("q_proj", "k_proj", "v_proj", "q_up",
                                             "k_up", "v_up", "g_proj", "o_proj"))
            (muon_heads if head_like else muon_plain).append(p)
        else:
            adam_params.append(p)
    n_heads = model.cfg.kda_heads
    return [
        Muon(muon_heads, lr=lr * 2, heads=n_heads),
        Muon(muon_plain, lr=lr * 2, heads=1),
        torch.optim.AdamW(adam_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0),
    ]


# ============================================================================
# Dados
# ============================================================================

def load_data(corpus_path, tokenizer_path, cache_path, max_chars=None):
    tok = BPETokenizer().load(tokenizer_path)
    if cache_path and os.path.exists(cache_path) and max_chars is None:
        data = torch.load(cache_path)
        print(f"tokens carregados do cache: {len(data):,}")
        return data, tok

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    if max_chars:
        text = text[:max_chars]
    t0 = time.time()
    ids = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    print(f"corpus: {len(text):,} chars -> {len(data):,} tokens "
          f"({time.time() - t0:.1f}s, vocab {tok.vocab_size})")
    if cache_path and max_chars is None:
        torch.save(data, cache_path)
    return data, tok


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, splits, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


# ============================================================================
# Treino
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="machado.txt")
    ap.add_argument("--tokenizer", default="tokenizer_bpe.pkl")
    ap.add_argument("--out_dir", default="checkpoints_k3")
    ap.add_argument("--token_cache", default="checkpoints_k3/tokens.pt")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--n_embd", type=int, default=320)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_experts", type=int, default=32)
    ap.add_argument("--top_k_experts", type=int, default=4)
    ap.add_argument("--max_chars", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test rápido")
    args = ap.parse_args()

    if args.quick:
        args.iters, args.eval_every, args.eval_iters = 30, 10, 3
        args.n_embd, args.n_layer, args.batch_size = 128, 4, 4
        args.block_size, args.max_chars = 128, 200_000
        args.n_experts, args.top_k_experts = 8, 2
        args.token_cache = None

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)

    data, tok = load_data(args.corpus, args.tokenizer, args.token_cache, args.max_chars)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    cfg = KimiK3Config(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        kda_heads=max(1, args.n_embd // 64),
        mla_heads=max(1, args.n_embd // 64),
        q_lora_rank=max(32, args.n_embd // 2),
        kv_lora_rank=max(16, args.n_embd // 3),
        n_experts=args.n_experts,
        n_experts_active=args.top_k_experts,
        moe_latent_dim=args.n_embd // 2,
        expert_inter=args.n_embd // 2,
        shared_inter=int(args.n_embd * 0.9),
    )
    model = GPTMachadoK3(cfg).to(device)
    print(f"device={device} | camadas={cfg.layer_types()}")
    print(f"parâmetros: {model.num_params()/1e6:.2f}M totais, "
          f"{model.num_params(active_only=True)/1e6:.2f}M ativos por token")

    optimizers = build_optimizer(model, args.optimizer, args.lr)

    def set_lr(it):
        if it < args.warmup:
            lr = args.lr * (it + 1) / args.warmup
        else:
            r = (it - args.warmup) / max(1, args.iters - args.warmup)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * r))
        for opt in optimizers:
            base = 2.0 if isinstance(opt, Muon) else 1.0
            for g in opt.param_groups:
                g["lr"] = lr * base
        return lr

    best_val = float("inf")
    model.train()
    t0 = time.time()
    for it in range(args.iters):
        lr = set_lr(it)
        x, y = get_batch(splits["train"], args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()

        if it % args.eval_every == 0 or it == args.iters - 1:
            m = evaluate(model, splits, args.block_size, args.batch_size, device, args.eval_iters)
            load = model.expert_load()[0]
            imbalance = (load.max() / (load.mean() + 1e-9)).item()
            print(f"it {it:6d} | lr {lr:.2e} | train {m['train']:.4f} | val {m['val']:.4f} "
                  f"| carga máx/média {imbalance:.2f} | {time.time()-t0:.0f}s")
            if m["val"] < best_val:
                best_val = m["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "iteration": it,
                    "val_loss": best_val,
                    "architecture": "kimi_k3",
                }, os.path.join(args.out_dir, "best_model.pt"))

    print(f"\nmelhor val loss: {best_val:.4f}")
    ctx = torch.tensor([tok.encode("A vida é")], dtype=torch.long, device=device)
    sample = model.generate(ctx, 80, temperature=0.8, top_k=40)
    print("amostra:", tok.decode(sample[0].tolist())[:400])


if __name__ == "__main__":
    main()
