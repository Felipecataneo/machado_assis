import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ============================================================================
# 1. ARQUITETURA DA PROVA (GPTExam - Baseado em R)
# ============================================================================

class HeadExam(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = torch.matmul(q, k.transpose(-2, -1)) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = torch.matmul(wei, v)
        return out

class MultiHeadAttentionExam(nn.Module):
    def __init__(self, n_head, n_embd, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([HeadExam(n_embd, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForwardExam(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
    def forward(self, x):
        return self.proj(F.relu(self.linear(x)))

class BlockExam(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttentionExam(n_head, n_embd, head_size, block_size)
        self.ffwd = FeedForwardExam(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = x + self.dropout(self.sa_head(self.ln1(x)))
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x

class GPTExam(nn.Module):
    def __init__(self, vocab_size, n_block=6, n_embd=384, block_size=256, n_head=6):
        super().__init__()
        self.block_size = block_size
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[BlockExam(n_head, n_embd, block_size) for _ in range(n_block)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, x):
        if x.size(1) > self.block_size: x = x[:, -self.block_size:]
        B, T = x.shape
        tok_emb = self.token_embeddings_table(x)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        return self.lm_head(x)

# ============================================================================
# 2. ARQUITETURA MODERNA V3 (GPTModern - RoPE + RMSNorm)
# Esta classe DEVE existir para carregar os pesos do seu "best_model_v3.pt"
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def get_embed(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin().unsqueeze(1), emb.cos().unsqueeze(1)

def apply_rotary(q, k, sin, cos):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class CausalSelfAttentionV3(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)) == 0)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        sin, cos = self.rotary.get_embed(T, x.device)
        q, k = apply_rotary(q, k, sin, cos)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForwardV3(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.SiLU(), # Swish
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class BlockV3(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttentionV3(n_embd, n_head, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.ff = FeedForwardV3(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTModern(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[BlockV3(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    
    def forward(self, x):
        if x.size(1) > self.block_size: x = x[:, -self.block_size:]
        x = self.tok_emb(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# ============================================================================
# 3. FUNÇÕES DE SUPORTE (Tokenizer & Loader)
# ============================================================================

class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
    def encode(self, text): return [self.stoi.get(c, 0) for c in text]
    def decode(self, idxs): return ''.join([self.itos.get(i, '?') for i in idxs])

@st.cache_resource
def load_models_comparison(path_exam, path_modern):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    tokenizer = None
    
    # 1. Carregar Modelo Prova (Fonte da Verdade para o Tokenizer)
    try:
        ckpt = torch.load(path_exam, map_location=device)
        tokenizer = CharTokenizer(ckpt['stoi'], ckpt['itos'])
        
        m_ex = GPTExam(vocab_size=ckpt['vocab_size'], n_block=ckpt['n_block'], 
                       n_embd=ckpt['n_embd'], block_size=ckpt['block_size'], n_head=ckpt['n_head'])
        m_ex.load_state_dict(ckpt['model_state_dict'])
        m_ex.to(device).eval()
        models['exam'] = m_ex
    except Exception as e:
        st.error(f"Erro ao carregar modelo da prova: {e}")
        return None, None, device

    # 2. Carregar Modelo Moderno (V3)
    try:
        ckpt_v3 = torch.load(path_modern, map_location=device)
        
        # Tenta inferir parâmetros ou usa defaults comuns se for apenas state_dict
        # Ajuste estes valores se seu modelo V3 tiver dimensões diferentes
        n_embd = 384
        n_head = 6
        n_layer = 6
        
        if isinstance(ckpt_v3, dict) and 'n_embd' in ckpt_v3:
             n_embd = ckpt_v3['n_embd']
             n_head = ckpt_v3['n_head']
             n_layer = ckpt_v3['n_block'] if 'n_block' in ckpt_v3 else 6

        m_v3 = GPTModern(vocab_size=ckpt['vocab_size'], n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=256)
        
        if isinstance(ckpt_v3, dict) and 'model_state_dict' in ckpt_v3:
            m_v3.load_state_dict(ckpt_v3['model_state_dict'])
        else:
            m_v3.load_state_dict(ckpt_v3) # Caso tenha salvo o state_dict direto
            
        m_v3.to(device).eval()
        models['modern'] = m_v3
    except Exception as e:
        st.warning(f"Não foi possível carregar o modelo moderno V3: {e}")
        models['modern'] = None

    return models, tokenizer, device

def generate_text(model, tokenizer, prompt, max_tokens, temp, top_k, device):
    if model is None: return "N/A"
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -model.block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temp
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(idx[0].tolist())

# ============================================================================
# 4. INTERFACE GRÁFICA
# ============================================================================

st.set_page_config(layout="wide", page_title="Comparativo Final: Prova vs SOTA", page_icon="⚔️")

st.title("⚔️ Comparativo de Arquiteturas Transformer")
st.markdown("Comparação entre o modelo exigido na **Prova II** (baseado em R) e uma variante **Moderna (V3)** otimizada.")

with st.sidebar:
    st.header("📁 Arquivos")
    path_exam = st.text_input("Modelo Prova (.pt)", "modelo_prova_leve.pt")
    path_modern = st.text_input("Modelo V3 (.pt)", "checkpoints_v3/best_model.pt") # Ponha o nome do seu arquivo aqui
    
    st.header("🎛️ Geração")
    tokens = st.slider("Tokens", 50, 800, 400)
    temp = st.slider("Temperatura", 0.1, 1.5, 0.7)
    top_k = st.slider("Top-K", 1, 100, 40)
    
    st.info("Nota: O 'Modelo V3' deve ter sido treinado com o mesmo vocabulário (machado.txt).")

models, tokenizer, device = load_models_comparison(path_exam, path_modern)

if models:
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"✅ Modelo Prova carregado")
    with c2:
        if models['modern']: st.success(f"✅ Modelo V3 carregado")
        else: st.error("❌ Modelo V3 não encontrado ou erro na arquitetura")

    prompt = st.text_area("✍️ Prompt Inicial:", "A vida é", height=100)
    
    if st.button("🚀 Gerar Comparação", type="primary", use_container_width=True):
        if not prompt: st.warning("Digite um prompt.")
        else:
            with st.spinner("Gerando textos..."):
                # Prova
                start = time.time()
                txt_exam = generate_text(models['exam'], tokenizer, prompt, tokens, temp, top_k, device)
                t_exam = time.time() - start
                
                # Moderno
                txt_modern = "Modelo não carregado."
                t_modern = 0
                if models['modern']:
                    start = time.time()
                    txt_modern = generate_text(models['modern'], tokenizer, prompt, tokens, temp, top_k, device)
                    t_modern = time.time() - start
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("📘 Resultado Prova (13k steps)")
                st.caption(f"Arquitetura: Pre-LN Standard | Tempo: {t_exam:.2f}s")
                st.code(txt_exam, language=None)
                
            with col_b:
                st.subheader("🚀 Resultado V3 (80k steps)")
                st.caption(f"Arquitetura: RoPE + RMSNorm | Tempo: {t_modern:.2f}s")
                st.code(txt_modern, language=None)