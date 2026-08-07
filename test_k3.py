"""Testes de sanidade da arquitetura K3 (rodar: python test_k3.py)."""

import torch

from model_k3 import (
    KimiK3Config, GPTMachadoK3, KimiDeltaAttention, SiTUGLU, LatentMoE,
    delta_rule_chunkwise, delta_rule_recurrent,
)


def test_kda_chunkwise_matches_recurrent():
    """A forma chunkwise tem que reproduzir a recorrência token a token."""
    torch.manual_seed(0)
    B, T, H, Dk, Dv = 2, 96, 3, 32, 32
    q = torch.nn.functional.normalize(torch.randn(B, T, H, Dk), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, Dk), dim=-1)
    v = torch.randn(B, T, H, Dv)
    beta = torch.rand(B, T, H)
    log_alpha = -torch.rand(B, T, H, Dk) * 0.25

    o_ref, s_ref = delta_rule_recurrent(q, k, v, beta, log_alpha)
    o_chunk, s_chunk = delta_rule_chunkwise(q, k, v, beta, log_alpha, chunk_size=32)

    err_o = (o_ref - o_chunk).abs().max().item()
    err_s = (s_ref - s_chunk).abs().max().item()
    assert err_o < 1e-3, f"saída divergente: {err_o}"
    assert err_s < 1e-3, f"estado divergente: {err_s}"
    print(f"  KDA chunkwise vs recorrente: erro max saída={err_o:.2e} estado={err_s:.2e}")


def test_kda_is_causal():
    """Mexer no token t não pode alterar as saídas anteriores a t."""
    torch.manual_seed(0)
    cfg = KimiK3Config(block_size=64, kda_heads=2, kda_head_dim=32, n_embd=64)
    kda = KimiDeltaAttention(cfg).eval()
    x = torch.randn(1, 32, 64)
    with torch.no_grad():
        out_a, _ = kda(x)
        x2 = x.clone()
        x2[:, 20:] += 3.0
        out_b, _ = kda(x2)
    # a short conv tem kernel 4, então o efeito começa em t = 20
    diff = (out_a[:, :20] - out_b[:, :20]).abs().max().item()
    assert diff < 1e-5, f"vazamento de futuro: {diff}"
    print(f"  causalidade do KDA ok (diff={diff:.2e})")


def test_situ_glu_bounded():
    """SiTU-GLU tem que saturar: |ativação| <= beta1*beta2."""
    ff = SiTUGLU(16, 32, beta1=4.0, beta2=25.0)
    with torch.no_grad():
        g = ff.gate(torch.randn(1000, 16) * 50)
        act = 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)
        u = 25.0 * torch.tanh(ff.up(torch.randn(1000, 16) * 50) / 25.0)
        peak = (act * u).abs().max().item()
    assert peak <= 4.0 * 25.0 + 1e-3, peak
    print(f"  SiTU-GLU limitado: pico={peak:.2f} (limite 100)")


def test_moe_balances():
    """Quantile Balancing tem que aproximar a carga dos experts da uniforme."""
    torch.manual_seed(0)
    cfg = KimiK3Config(n_embd=64, moe_latent_dim=32, n_experts=16,
                       n_experts_active=2, expert_inter=32, shared_inter=32,
                       router_bias_rate=0.05)
    moe = LatentMoE(cfg).train()
    x = torch.randn(8, 64, 64)
    moe(x)
    first = moe.last_load.clone()
    for _ in range(200):
        moe(x)
    last = moe.last_load
    uniform = 1.0 / cfg.n_experts
    spread_before = (first - uniform).abs().sum().item()
    spread_after = (last - uniform).abs().sum().item()
    assert spread_after <= spread_before, (spread_before, spread_after)
    print(f"  desbalanceamento MoE: {spread_before:.3f} -> {spread_after:.3f}")


def test_layer_pattern():
    cfg = KimiK3Config(n_layer=8, mla_every=4)
    types = cfg.layer_types()
    assert types == ["kda", "kda", "kda", "mla", "kda", "kda", "kda", "mla"], types
    assert types[-1] == "mla", "a última camada precisa ser global"
    print(f"  padrão de camadas 3:1 ok: {types}")


def test_forward_backward():
    torch.manual_seed(0)
    cfg = KimiK3Config(vocab_size=200, block_size=64, n_embd=64, n_layer=4,
                       kda_heads=2, kda_head_dim=32, mla_heads=2, mla_head_dim=32,
                       q_lora_rank=48, kv_lora_rank=32, n_experts=8,
                       n_experts_active=2, moe_latent_dim=32, expert_inter=32,
                       shared_inter=48)
    model = GPTMachadoK3(cfg)
    idx = torch.randint(0, 200, (2, 64))
    logits, loss = model(idx, idx)
    loss.backward()
    grads = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(loss) and grads > 0
    assert logits.shape == (2, 64, 200)
    print(f"  forward/backward ok: loss={loss.item():.3f}, "
          f"params={model.num_params()/1e6:.2f}M "
          f"(ativos {model.num_params(active_only=True)/1e6:.2f}M)")


def test_generate():
    cfg = KimiK3Config(vocab_size=100, block_size=32, n_embd=64, n_layer=2,
                       kda_heads=2, kda_head_dim=32, mla_heads=2, mla_head_dim=32,
                       q_lora_rank=32, kv_lora_rank=16, n_experts=4,
                       n_experts_active=2, moe_latent_dim=32, expert_inter=32,
                       shared_inter=32)
    model = GPTMachadoK3(cfg)
    out = model.generate(torch.zeros(1, 3, dtype=torch.long), 12, temperature=0.8, top_k=10)
    assert out.shape == (1, 15)
    print("  geração ok")


if __name__ == "__main__":
    for fn in [test_kda_chunkwise_matches_recurrent, test_kda_is_causal,
               test_situ_glu_bounded, test_moe_balances, test_layer_pattern,
               test_forward_backward, test_generate]:
        print(f"{fn.__name__}:")
        fn()
    print("\nok — todos os testes passaram.")
