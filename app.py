import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path
import time

# ============================================================================
# CLASSES DO MODELO (copiar do arquivo principal)
# ============================================================================

class Head(nn.Module):
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


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
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


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTMachado(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=128, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
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
        bytes_list = b"".join(self.vocab[idx] for idx in tokens if idx in self.vocab)
        return bytes_list.decode("utf-8", errors="replace")
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.merges = data['merges']


# ============================================================================
# FUNÇÕES DE CACHE
# ============================================================================

@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path):
    """Carrega modelo e tokenizer (executado uma única vez)"""
    
    # Carregar tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    # Carregar modelo
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['model_config']
    
    model = GPTMachado(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, checkpoint['training_info']


# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Gerador Machado de Assis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .generated-text {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        font-family: 'Georgia', serif;
        line-height: 1.8;
        font-size: 1.1rem;
    }
    .stats-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

st.markdown('<div class="main-title">📚 Gerador de Texto Machado de Assis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modelo GPT treinado na obra completa de Machado de Assis</div>', unsafe_allow_html=True)

# Sidebar com configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.subheader("Caminhos dos Arquivos")
    model_path = st.text_input(
        "Modelo (.pt)",
        value="checkpoints/machado_model.pt",
        help="Caminho para o arquivo machado_model.pt"
    )
    
    tokenizer_path = st.text_input(
        "Tokenizer (.pkl)",
        value="tokenizer_bpe.pkl",
        help="Caminho para o arquivo tokenizer_bpe.pkl"
    )
    
    st.divider()
    
    st.subheader("Parâmetros de Geração")
    
    max_tokens = st.slider(
        "Comprimento do Texto",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
        help="Número de tokens a gerar"
    )
    
    temperature = st.slider(
        "Temperature (Criatividade)",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Maior = mais criativo e aleatório"
    )
    
    top_k = st.slider(
        "Top-K (Diversidade)",
        min_value=1,
        max_value=100,
        value=40,
        step=5,
        help="Maior = mais diversidade lexical"
    )
    
    st.divider()
    
    # Exemplos de prompts
    st.subheader("📝 Prompts Sugeridos")
    example_prompts = [
        "A vida é",
        "Bentinho olhou pela janela e viu",
        "Capitu tinha olhos de",
        "A sociedade do Rio de Janeiro",
        "O amor é uma coisa",
        "Brás Cubas pensava em"
    ]
    
    for prompt in example_prompts:
        if st.button(f"💡 {prompt}", key=prompt):
            st.session_state.prompt = prompt

# Carregar modelo
try:
    with st.spinner("🔄 Carregando modelo..."):
        model, tokenizer, training_info = load_model_and_tokenizer(model_path, tokenizer_path)
    
    # Mostrar informações do modelo
    with st.expander("ℹ️ Informações do Modelo"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Iterações de Treino", f"{training_info['final_iteration']:,}")
        with col2:
            st.metric("Val Loss", f"{training_info['best_val_loss']:.4f}")
        with col3:
            params = sum(p.numel() for p in model.parameters())
            st.metric("Parâmetros", f"{params:,}")
    
    # Área de input do prompt
    prompt = st.text_area(
        "Digite seu prompt:",
        value=st.session_state.get('prompt', "A vida"),
        height=100,
        help="Digite o início do texto que você quer que o modelo continue"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        generate_button = st.button("✨ Gerar Texto", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Limpar", use_container_width=True)
    with col3:
        st.download_button(
            "💾 Baixar",
            data=st.session_state.get('generated_text', ''),
            file_name="texto_machado.txt",
            mime="text/plain",
            disabled=not st.session_state.get('generated_text')
        )
    
    if clear_button:
        st.session_state.generated_text = ""
        st.rerun()
    
    # Geração de texto
    if generate_button and prompt:
        with st.spinner("🎨 Gerando texto no estilo Machado de Assis..."):
            start_time = time.time()
            
            try:
                # Tokenizar prompt
                tokens = tokenizer.encode(prompt)
                idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                
                # Gerar
                with torch.no_grad():
                    generated = model.generate(
                        idx,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                
                # Decodificar
                generated_text = tokenizer.decode(generated[0].tolist())
                
                elapsed_time = time.time() - start_time
                
                # Salvar no session state
                st.session_state.generated_text = generated_text
                
                # Exibir resultado
                st.success(f"✅ Texto gerado em {elapsed_time:.2f} segundos!")
                
                # Estatísticas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Caracteres", len(generated_text))
                with col2:
                    st.metric("Palavras", len(generated_text.split()))
                with col3:
                    st.metric("Tokens", len(generated[0]))
                with col4:
                    st.metric("Tokens/seg", f"{len(generated[0])/elapsed_time:.1f}")
                
                # Texto gerado
                st.markdown("### 📖 Texto Gerado")
                st.markdown(f'<div class="generated-text">{generated_text}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Erro na geração: {str(e)}")
    
    elif st.session_state.get('generated_text'):
        # Mostrar texto anterior se existir
        st.markdown("### 📖 Último Texto Gerado")
        st.markdown(f'<div class="generated-text">{st.session_state.generated_text}</div>', unsafe_allow_html=True)

except FileNotFoundError as e:
    st.error(f"""
    ❌ **Arquivos não encontrados!**
    
    Verifique se os seguintes arquivos existem:
    - `{model_path}`
    - `{tokenizer_path}`
    
    **Como obter os arquivos:**
    1. Execute o treinamento: `python gpt_machado.py`
    2. Aguarde o término do treinamento
    3. Os arquivos serão criados automaticamente
    """)
    
except Exception as e:
    st.error(f"❌ Erro ao carregar modelo: {str(e)}")
    st.info("Certifique-se de que o modelo foi treinado corretamente.")

# Rodapé
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>
    🤖 Modelo GPT com BPE Tokenizer | Treinado na obra completa de Machado de Assis<br>
    📊 Parâmetros configuráveis para diferentes estilos de geração
    </small>
</div>
""", unsafe_allow_html=True)