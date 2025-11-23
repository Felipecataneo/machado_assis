import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
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
        if x.size(1) > self.block_size: 
            x = x[:, -self.block_size:]
        B, T = x.shape
        tok_emb = self.token_embeddings_table(x)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        return self.lm_head(x)

# ============================================================================
# 2. ARQUITETURA MODERNA V3 (RoPE + RMSNorm + SwiGLU)
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
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size:]
        B, T = idx.shape
        tok = self.tok_emb(idx)
        x = tok
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ============================================================================
# 3. TOKENIZERS
# ============================================================================

class CharTokenizer:
    """Tokenizer simples char-level (usado na prova)"""
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
    
    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]
    
    def decode(self, idxs):
        return ''.join([self.itos.get(i, '?') for i in idxs])

class BPETokenizer:
    """Tokenizer BPE (usado no V3)"""
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
# 4. FUNÇÕES DE CARREGAMENTO
# ============================================================================

@st.cache_resource
def load_models_comparison(path_exam, path_modern, path_tokenizer_bpe):
    """Carrega ambos os modelos com seus respectivos tokenizers"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    tokenizers = {}
    infos = {}
    
    # 1. Carregar Modelo da Prova (Char-level)
    try:
        ckpt_exam = torch.load(path_exam, map_location=device)
        tokenizers['exam'] = CharTokenizer(ckpt_exam['stoi'], ckpt_exam['itos'])
        
        model_exam = GPTExam(
            vocab_size=ckpt_exam['vocab_size'], 
            n_block=ckpt_exam['n_block'], 
            n_embd=ckpt_exam['n_embd'], 
            block_size=ckpt_exam['block_size'], 
            n_head=ckpt_exam['n_head']
        )
        model_exam.load_state_dict(ckpt_exam['model_state_dict'])
        model_exam.to(device).eval()
        models['exam'] = model_exam
        
        infos['exam'] = {
            'vocab_size': ckpt_exam['vocab_size'],
            'n_embd': ckpt_exam['n_embd'],
            'n_head': ckpt_exam['n_head'],
            'n_layer': ckpt_exam['n_block'],
            'block_size': ckpt_exam['block_size'],
            'params': sum(p.numel() for p in model_exam.parameters()) / 1e6,
            'train_loss': ckpt_exam.get('train_loss', 'N/A'),
            'val_loss': ckpt_exam.get('val_loss', 'N/A')
        }
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo da prova: {e}")
        return None, None, None, device

    # 2. Carregar Modelo V3 (BPE)
    try:
        # Carregar tokenizer BPE
        tokenizer_bpe = BPETokenizer()
        tokenizer_bpe.load(path_tokenizer_bpe)
        tokenizers['modern'] = tokenizer_bpe
        
        # Carregar checkpoint V3
        ckpt_v3 = torch.load(path_modern, map_location=device)
        state_v3 = ckpt_v3['model_state']
        
        # Inferir parâmetros do state_dict
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
        model_v3.to(device).eval()
        models['modern'] = model_v3
        
        infos['modern'] = {
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': 8,
            'n_layer': 4,
            'block_size': 256,
            'params': sum(p.numel() for p in model_v3.parameters()) / 1e6,
            'train_loss': 'N/A',
            'val_loss': ckpt_v3.get('val_loss', 'N/A'),
            'iteration': ckpt_v3.get('iteration', 80000)
        }
        
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar modelo V3: {e}")
        models['modern'] = None
        tokenizers['modern'] = None
        infos['modern'] = None

    return models, tokenizers, infos, device

# ============================================================================
# 5. FUNÇÃO DE GERAÇÃO
# ============================================================================

def generate_text(model, tokenizer, prompt, max_tokens, temp, top_k, device):
    """Gera texto com um modelo específico"""
    if model is None or tokenizer is None:
        return "Modelo não disponível", 0
    
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -model.block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temp
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinmultinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    
    return tokenizer.decode(idx[0].tolist()), idx.size(1)

# ============================================================================
# 6. INTERFACE STREAMLIT
# ============================================================================

st.set_page_config(
    layout="wide", 
    page_title="Comparativo: Prova vs V3", 
    page_icon="⚔️"
)

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #666; margin-bottom: 2rem; }
    .model-card { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
    .exam-card { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .v3-card { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
    .generated-text { 
        background-color: #f8f9fa; 
        padding: 1.5rem; 
        border-radius: 10px; 
        font-family: 'Georgia', serif; 
        line-height: 1.8; 
        font-size: 1.05rem;
        white-space: pre-wrap;
        max-height: 600px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">⚔️ Comparativo de Arquiteturas Transformer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modelo da Prova II (Standard) vs Modelo V3 (State-of-the-art)</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Configuração de Arquivos")
    
    path_exam = st.text_input("Modelo Prova (.pt)", "modelo_prova_leve.pt")
    path_modern = st.text_input("Modelo V3 (.pt)", "checkpoints_v3/best_model.pt")
    path_tokenizer_bpe = st.text_input("Tokenizer BPE (.pkl)", "tokenizer_bpe.pkl")
    
    st.divider()
    
    st.header("🎛️ Parâmetros de Geração")
    max_tokens = st.slider("Tokens a gerar", 50, 800, 400, 50)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    top_k = st.slider("Top-K", 1, 100, 40, 5)
    
    st.divider()
    
    st.header("💡 Prompts Sugeridos")
    examples = [
        "A vida é",
        "Bentinho olhou pela janela",
        "Capitu tinha olhos de",
        "O amor é",
        "Brás Cubas pensava"
    ]
    
    for ex in examples:
        if st.button(f"📝 {ex}", key=ex):
            st.session_state.prompt = ex
    
    st.divider()
    st.info("""
    **⚠️ Nota Importante:**
    
    Os modelos usam tokenizers diferentes:
    - **Prova:** CharTokenizer (nível de caractere)
    - **V3:** BPETokenizer (subword tokens)
    
    Isso é esperado e não afeta a comparação qualitativa.
    """)

# Carregar modelos
models, tokenizers, infos, device = load_models_comparison(path_exam, path_modern, path_tokenizer_bpe)

if models and models.get('exam'):
    
    # Cards informativos
    col1, col2 = st.columns(2)
    
    with col1:
        info_exam = infos['exam']
        st.markdown(f"""
        <div class="model-card exam-card">
            <h3>📘 Modelo da Prova II</h3>
            <p><strong>Arquitetura:</strong> GPT Standard (baseado em R)</p>
            <ul>
                <li><strong>Tokenizer:</strong> Character-level ({info_exam['vocab_size']} chars)</li>
                <li><strong>Embedding:</strong> {info_exam['n_embd']} dim</li>
                <li><strong>Heads:</strong> {info_exam['n_head']}</li>
                <li><strong>Layers:</strong> {info_exam['n_layer']}</li>
                <li><strong>Context:</strong> {info_exam['block_size']} tokens</li>
                <li><strong>Parâmetros:</strong> {info_exam['params']:.2f}M</li>
                <li><strong>Técnicas:</strong> Standard Attention + Positional Embeddings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if models.get('modern') and infos.get('modern'):
            info_v3 = infos['modern']
            st.markdown(f"""
            <div class="model-card v3-card">
                <h3>🚀 Modelo V3 - Moderno</h3>
                <p><strong>Arquitetura:</strong> GPT com técnicas SOTA</p>
                <ul>
                    <li><strong>Tokenizer:</strong> BPE ({info_v3['vocab_size']} tokens)</li>
                    <li><strong>Embedding:</strong> {info_v3['n_embd']} dim</li>
                    <li><strong>Heads:</strong> {info_v3['n_head']}</li>
                    <li><strong>Layers:</strong> {info_v3['n_layer']}</li>
                    <li><strong>Context:</strong> {info_v3['block_size']} tokens</li>
                    <li><strong>Parâmetros:</strong> {info_v3['params']:.2f}M</li>
                    <li><strong>Técnicas:</strong> RoPE + RMSNorm + SwiGLU</li>
                    <li><strong>Iterações:</strong> {info_v3['iteration']:,}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Modelo V3 não carregado")
    
    # Input
    prompt = st.text_area(
        "✍️ Digite seu prompt:",
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
            
            # Modelo Prova
            start_exam = time.time()
            text_exam, tokens_exam = generate_text(
                models['exam'], 
                tokenizers['exam'], 
                prompt, 
                max_tokens, 
                temperature, 
                top_k, 
                device
            )
            time_exam = time.time() - start_exam
            
            # Modelo V3
            if models.get('modern') and tokenizers.get('modern'):
                start_v3 = time.time()
                text_v3, tokens_v3 = generate_text(
                    models['modern'], 
                    tokenizers['modern'], 
                    prompt, 
                    max_tokens, 
                    temperature, 
                    top_k, 
                    device
                )
                time_v3 = time.time() - start_v3
            else:
                text_v3 = "Modelo V3 não disponível"
                tokens_v3 = 0
                time_v3 = 0
            
            st.success("✅ Textos gerados com sucesso!")
            
            # Estatísticas
            st.subheader("📊 Estatísticas de Geração")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Prova - Caracteres", len(text_exam))
                if models.get('modern'):
                    st.metric("V3 - Caracteres", len(text_v3))
            
            with col2:
                st.metric("Prova - Palavras", len(text_exam.split()))
                if models.get('modern'):
                    st.metric("V3 - Palavras", len(text_v3.split()))
            
            with col3:
                st.metric("Prova - Tokens", tokens_exam)
                if models.get('modern'):
                    st.metric("V3 - Tokens", tokens_v3)
            
            with col4:
                st.metric("Prova - Tempo (s)", f"{time_exam:.2f}")
                if models.get('modern'):
                    st.metric("V3 - Tempo (s)", f"{time_v3:.2f}")
            
            # Textos gerados
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📘 Modelo da Prova")
                st.caption("Arquitetura padrão com positional embeddings")
                st.markdown(f'<div class="generated-text">{text_exam}</div>', unsafe_allow_html=True)
                st.download_button(
                    "💾 Baixar Texto Prova",
                    text_exam,
                    f"prova_{int(time.time())}.txt",
                    key="download_exam"
                )
            
            with col2:
                st.markdown("### 🚀 Modelo V3")
                st.caption("Arquitetura moderna com RoPE + RMSNorm + SwiGLU")
                if models.get('modern'):
                    st.markdown(f'<div class="generated-text">{text_v3}</div>', unsafe_allow_html=True)
                    st.download_button(
                        "💾 Baixar Texto V3",
                        text_v3,
                        f"v3_{int(time.time())}.txt",
                        key="download_v3"
                    )
                else:
                    st.error("Modelo V3 não disponível")
            
            # Análise comparativa
            st.divider()
            st.markdown("### 🔍 Análise Comparativa")
            
            st.info("""
            **📝 Diferenças de Arquitetura:**
            
            **Modelo da Prova (Standard GPT):**
            - ✅ Positional embeddings fixos (learned)
            - ✅ LayerNorm padrão
            - ✅ ReLU activation no FFN
            - ✅ Tokenização character-level (simples)
            
            **Modelo V3 (State-of-the-art):**
            - 🚀 RoPE (Rotary Position Embeddings) - melhor extrapolação de contexto
            - 🚀 RMSNorm - normalização mais eficiente
            - 🚀 SwiGLU - activation function superior
            - 🚀 BPE tokenization - vocabulário mais rico
            - 🚀 Weight tying (embedding = output layer)
            
            **Expectativas:**
            - V3 deve gerar texto com melhor coerência em sequências longas
            - V3 deve ter vocabulário mais rico (BPE vs char-level)
            - Ambos devem manter o estilo machadiano
            """)

else:
    st.error("""
    ❌ **Não foi possível carregar os modelos!**
    
    Verifique se os arquivos existem:
    - Modelo da Prova: `modelo_prova_leve.pt`
    - Modelo V3: `checkpoints_v3/best_model.pt    - Tokenizer BPE:tokenizer_bpe.pkl`
""")
Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>
    📚 Corpus: Obra completa de Machado de Assis<br>
    🤖 Prova: Standard GPT | V3: RoPE + RMSNorm + SwiGLU<br>
    ⚙️ Tokenizers independentes: CharTokenizer (Prova) | BPE (V3)
    </small>
</div>
""", unsafe_allow_html=True)
