import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
from pathlib import Path
import time

# ============================================================================
# CLASSES DO MODELO V1 (500k iterações - Overfitting)
# ============================================================================

class HeadV1(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttentionV1(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([HeadV1(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForwardV1(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class BlockV1(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        self.sa = MultiHeadAttentionV1(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForwardV1(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTMachadoV1(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[BlockV1(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============================================================================
# CLASSES DO MODELO V3 (80k iterações - Moderno)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


def apply_rotary_pos_emb(q, k, sin, cos):
    q_ = (q * cos) + (rotate_every_two(q) * sin)
    k_ = (k * cos) + (rotate_every_two(k) * sin)
    return q_, k_


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def get_embed(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        sin = emb.sin().unsqueeze(1)
        cos = emb.cos().unsqueeze(1)
        return sin, cos


class MultiHeadAttentionRope(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        sin, cos = self.rotary.get_embed(T, x.device)
        q, k = apply_rotary_pos_emb(q, k, sin, cos)
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        attn_scores = attn_scores.masked_fill(~self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.permute(0,2,1,3).contiguous().view(B, T, C)
        out = self.out(out)
        out = self.dropout(out)
        return out


class FeedForwardV3(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        inner = 4 * n_embd
        self.fc1 = nn.Linear(n_embd, inner)
        self.fc2 = nn.Linear(n_embd, inner)
        self.fc_out = nn.Linear(inner, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = F.silu(x2) * x1
        x = self.fc_out(x)
        return self.dropout(x)


class BlockV3(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = MultiHeadAttentionRope(n_embd, n_head, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.ff = FeedForwardV3(n_embd, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTMachadoV3(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=4, block_size=256, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([BlockV3(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        x = tok
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, device=None):
        if device is None:
            device = next(self.parameters()).device
        idx = idx.to(device)
        
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            with torch.cuda.amp.autocast(enabled=False):
                logits = self(idx_cond)
                logits = logits.float()
                logits = logits[:, -1, :]
                logits = torch.clamp(logits, min=-100, max=100)
                
                if temperature > 0:
                    logits = logits / temperature
                
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.where(logits < v[:, [-1]], 
                                        torch.full_like(logits, -float('inf')), 
                                        logits)
                
                probs = F.softmax(logits, dim=-1)
                
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)
                
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
        
        return idx


# ============================================================================
# TOKENIZER (ÚNICO - FUNCIONA PARA AMBOS)
# ============================================================================

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = {}
            for pair in zip(tokens, tokens[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            
            idx = self.merges[pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def decode(self, tokens):
        bytes_list = b"".join(self.vocab.get(idx, b'?') for idx in tokens)
        return bytes_list.decode("utf-8", errors="replace")
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.merges = data['merges']


# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models_and_tokenizer(v1_path, v3_path, tokenizer_path):
    """Carrega ambos os modelos e o tokenizer (executado uma vez)"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Carregar tokenizer (único para ambos)
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    # Carregar V1
    checkpoint_v1 = torch.load(v1_path, map_location=device)
    config_v1 = checkpoint_v1['model_config']
    model_v1 = GPTMachadoV1(
        vocab_size=config_v1['vocab_size'],
        n_embd=config_v1['n_embd'],
        n_head=config_v1['n_head'],
        n_layer=config_v1['n_layer'],
        block_size=config_v1['block_size'],
        dropout=0.0
    )
    model_v1.load_state_dict(checkpoint_v1['model_state_dict'])
    model_v1.eval()
    model_v1.to(device)
    
    # Carregar V3
    checkpoint_v3 = torch.load(v3_path, map_location=device)
    # V3 não tem 'model_config' embutido, usar checkpoint state
    state_v3 = checkpoint_v3['model_state']
    # Inferir vocab_size do embedding
    vocab_size = state_v3['tok_emb.weight'].shape[0]
    n_embd = state_v3['tok_emb.weight'].shape[1]
    
    model_v3 = GPTMachadoV3(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=8,
        n_layer=4,
        block_size=256,
        dropout=0.0
    )
    model_v3.load_state_dict(state_v3)
    model_v3.eval()
    model_v3.to(device)
    
    info_v1 = checkpoint_v1.get('training_info', {})
    info_v3 = {
        'final_iteration': checkpoint_v3.get('iteration', 80000),
        'best_val_loss': checkpoint_v3.get('val_loss', 2.1)
    }
    
    return model_v1, model_v3, tokenizer, info_v1, info_v3, device


# ============================================================================
# GERAÇÃO DE TEXTO
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens, temperature, top_k, device):
    """Gera texto com um modelo específico"""
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    
    text = tokenizer.decode(generated[0].tolist())
    return text, len(generated[0])


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Comparativo Machado V1 vs V3",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #666; margin-bottom: 2rem; }
    .model-card { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
    .v1-card { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .v3-card { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
    .generated-text { 
        background-color: #f8f9fa; 
        padding: 1.5rem; 
        border-radius: 10px; 
        font-family: 'Georgia', serif; 
        line-height: 1.8; 
        font-size: 1.05rem;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">📚 Gerador Machado de Assis - Comparativo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modelo V1 (Baseline - 500k iter) vs V3 (Moderno - 80k iter)</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.subheader("Caminhos dos Modelos")
    v1_path = st.text_input("Modelo V1 (.pt)", "checkpoints/machado_model.pt")
    v3_path = st.text_input("Modelo V3 (.pt)", "checkpoints_v3/best_model.pt")
    tokenizer_path = st.text_input("Tokenizer (.pkl)", "tokenizer_bpe.pkl")
    
    st.divider()
    
    st.subheader("Parâmetros de Geração")
    max_tokens = st.slider("Tokens a gerar", 50, 500, 200, 50)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    top_k = st.slider("Top-K", 1, 100, 40, 5)
    
    st.divider()
    
    st.subheader("📝 Prompts Sugeridos")
    examples = [
        "A vida é",
        "Bentinho olhou pela janela",
        "Capitu tinha olhos de",
        "O amor é",
        "Brás Cubas pensava"
    ]
    
    for ex in examples:
        if st.button(f"💡 {ex}", key=ex):
            st.session_state.prompt = ex

# Carregar modelos
try:
    with st.spinner("🔄 Carregando modelos..."):
        model_v1, model_v3, tokenizer, info_v1, info_v3, device = load_models_and_tokenizer(
            v1_path, v3_path, tokenizer_path
        )
    
    # Info Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card v1-card">
            <h3>📊 Modelo V1 - Baseline</h3>
            <p><strong>Descrição:</strong> GPT com arquitetura padrão</p>
            <ul>
                <li><strong>Iterações:</strong> 500.000</li>
                <li><strong>Arquitetura:</strong> Attention Padrão + Positional Embeddings</li>
                <li><strong>Camadas:</strong> 6 layers, 384 dim, 6 heads</li>
                <li><strong>Parâmetros:</strong> ~15M</li>
                <li><strong>Train Loss:</strong> 1.56</li>
                <li><strong>Val Loss:</strong> 4.16</li>
                <li><strong>Gap (Overfitting):</strong> 2.60 ⚠️</li>
                <li><strong>Status:</strong> Overfitting severo - pode memorizar trechos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card v3-card">
            <h3>🚀 Modelo V3 - Moderno</h3>
            <p><strong>Descrição:</strong> GPT com técnicas state-of-the-art</p>
            <ul>
                <li><strong>Iterações:</strong> 80.000 (early stopping)</li>
                <li><strong>Arquitetura:</strong> RoPE + RMSNorm + SwiGLU</li>
                <li><strong>Camadas:</strong> 4 layers, 256 dim, 8 heads</li>
                <li><strong>Parâmetros:</strong> ~8M</li>
                <li><strong>Train Loss:</strong> ~1.8 (estimado)</li>
                <li><strong>Val Loss:</strong> ~2.1 (estimado)</li>
                <li><strong>Gap (Overfitting):</strong> ~0.3 ✅</li>
                <li><strong>Status:</strong> Boa generalização - mais criativo</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("ℹ️ **Tokenizer:** Único arquivo BPE (vocab_size=2000) funciona para ambos os modelos")
    
    # Input
    prompt = st.text_area(
        "Digite seu prompt:",
        value=st.session_state.get('prompt', "A vida"),
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_btn = st.button("✨ Gerar Comparação", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Limpar", use_container_width=True)
    
    if clear_btn:
        st.session_state.clear()
        st.rerun()
    
    # Geração
    if generate_btn and prompt:
        with st.spinner("🎨 Gerando textos com ambos os modelos..."):
            
            # V1
            start_v1 = time.time()
            text_v1, tokens_v1 = generate_text(model_v1, tokenizer, prompt, max_tokens, temperature, top_k, device)
            time_v1 = time.time() - start_v1
            
            # V3
            start_v3 = time.time()
            text_v3, tokens_v3 = generate_text(model_v3, tokenizer, prompt, max_tokens, temperature, top_k, device)
            time_v3 = time.time() - start_v3
            
            st.success("✅ Textos gerados com sucesso!")
            
            # Stats
            st.subheader("📊 Estatísticas de Geração")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("V1 - Caracteres", len(text_v1))
                st.metric("V3 - Caracteres", len(text_v3))
            
            with col2:
                st.metric("V1 - Palavras", len(text_v1.split()))
                st.metric("V3 - Palavras", len(text_v3.split()))
            
            with col3:
                st.metric("V1 - Tokens", tokens_v1)
                st.metric("V3 - Tokens", tokens_v3)
            
            with col4:
                st.metric("V1 - Tempo (s)", f"{time_v1:.2f}")
                st.metric("V3 - Tempo (s)", f"{time_v3:.2f}")
            
            # Textos
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Modelo V1 (500k iterações)")
                st.caption("⚠️ Atenção: pode ter memorizado trechos do corpus (overfitting)")
                st.markdown(f'<div class="generated-text">{text_v1}</div>', unsafe_allow_html=True)
                st.download_button(
                    "💾 Baixar V1",
                    text_v1,
                    f"machado_v1_{int(time.time())}.txt",
                    key="download_v1"
                )
            
            with col2:
                st.markdown("### 🚀 Modelo V3 (80k iterações)")
                st.caption("✅ Esperado: maior capacidade de generalização e criatividade")
                st.markdown(f'<div class="generated-text">{text_v3}</div>', unsafe_allow_html=True)
                st.download_button(
                    "💾 Baixar V3",
                    text_v3,
                    f"machado_v3_{int(time.time())}.txt",
                    key="download_v3"
                )
            
            # Análise
            st.divider()
            st.info("""
            **📝 Interpretação dos Resultados:**
            
            - **V1 (Gap 2.6):** Se o texto parecer muito similar a trechos conhecidos da obra, confirma overfitting
            - **V3 (Gap 0.3):** Deve mostrar mais criatividade mantendo o estilo machadiano
            - **Tokenizer:** O mesmo arquivo BPE funciona para ambos pois foi treinado no mesmo corpus
            """)

except FileNotFoundError as e:
    st.error(f"""
    ❌ **Arquivos não encontrados!**
    
    Verifique se existem:
    - `{v1_path}` (Modelo V1)
    - `{v3_path}` (Modelo V3)
    - `{tokenizer_path}` (Tokenizer)
    
    **Nota:** Use qualquer tokenizer_bpe.pkl gerado (são idênticos)
    """)

except Exception as e:
    st.error(f"❌ Erro ao carregar: {str(e)}")
    with st.expander("Detalhes do erro"):
        st.code(str(e))

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>
    🤖 V1: 500k iter (overfitting) | V3: 80k iter (moderno)<br>
    📚 Corpus: Obra completa de Machado de Assis | Tokenizer: BPE único
    </small>
</div>
""", unsafe_allow_html=True)