# Machado K3 — versão com a arquitetura do Kimi K3

Terceira versão do gerador machadiano, separada das anteriores (Prova e V3).
Enquanto a V3 usa o pacote "moderno padrão" (RoPE + RMSNorm + SwiGLU), esta
versão reimplementa, em escala de estudo, a arquitetura anunciada no
lançamento do **Kimi K3** (Moonshot AI, julho de 2026).

| arquivo | papel |
| --- | --- |
| `model_k3.py` | arquitetura (KDA, Gated MLA, AttnRes, LatentMoE, SiTU-GLU) |
| `train_k3.py` | treino no corpus `machado.txt` + Per-Head Muon |
| `test_k3.py` | testes de sanidade (inclusive chunkwise × recorrente) |
| `app.py` | comparativo das três versões (a K3 aparece se houver checkpoint) |

## O que o Kimi K3 traz e o que foi implementado aqui

### 1. KDA — Kimi Delta Attention

Atenção **linear** com estado recorrente, delta rule e gate **por canal**:

```
S_t = (I − β_t k_t k_tᵀ) diag(α_t) S_{t−1} + β_t k_t v_tᵀ
o_t = q_tᵀ S_t
```

O Gated DeltaNet usa um escalar de esquecimento; o KDA usa um vetor `α_t`, de
modo que cada dimensão do estado decai a uma taxa diferente — algumas guardam
associações de longo prazo, outras funcionam como memória de trabalho.

Caminhos de entrada: projeção linear → convolução causal depthwise (kernel 4) →
SiLU → normalização L2 em `q`/`k`. `β_t = σ(W_β x)`. Saída: RMSNorm por cabeça,
gate sigmoide full-rank e projeção de saída.

Duas implementações equivalentes em `model_k3.py`:

* `delta_rule_recurrent` — recorrência token a token (referência, e usável para
  decodificação incremental com estado);
* `delta_rule_chunkwise` — forma por chunks, que resolve por chunk um sistema
  triangular inferior unitário em `u` e paraleliza o resto. `test_k3.py`
  confere as duas (erro máximo ~1e-7).

### 2. Hibridização 3:1 + NoPE

Três camadas KDA para cada camada de **Gated MLA**, com a última camada sempre
global — no K3 real são 93 camadas (69 KDA + 24 MLA); aqui o mesmo padrão em 6
ou 8 camadas. As camadas MLA usam compressão q-LoRA/kv-LoRA e **NoPE**: não há
RoPE nem embedding posicional em lugar nenhum do modelo. A posição chega
implicitamente pelo decaimento recorrente do KDA.

### 3. AttnRes — Attention Residuals

O residual pré-norm clássico soma as saídas de todas as camadas com peso fixo 1,
o que faz o hidden state crescer sem controle com a profundidade. O AttnRes
substitui isso por uma atenção sobre as saídas anteriores:

```
h_l = Σ_i α_{i→l} v_i,   α_{i→l} = softmax_i( w_l · RMSNorm(v_i) / √d )
```

`w_l` (pseudo-query, um vetor por camada) é inicializado em **zero**, condição
destacada no paper: os pesos começam uniformes. Custo: um RMSNorm e um vetor por
camada. `use_attn_res=False` na config volta ao residual clássico.

### 4. Stable LatentMoE

O roteamento acontece no **espaço latente** (metade da dimensão do residual),
não no residual inteiro — no K3 é 7168 → 3584, aqui `n_embd → n_embd/2`:

```
x —down→ latente → router sigmoide, top-k de N experts
        experts SiTU-GLU rodam dentro do latente
        agregado → RMSNorm → up → n_embd  (+ experts compartilhados full-width)
```

Proporções do K3 preservadas: top-k sobre um pool grande, 2 experts
compartilhados, intermediário ≈ 0,86 × dimensão. As escalas absolutas caem de
896 experts / top-16 para 16–32 experts / top-2–4.

### 5. SiTU-GLU

```
f(x) = down[ β₁·tanh(W_g x/β₁) · σ(W_g x) · β₂·tanh(W_u x/β₂) ]
```

com β₁ = 4 e β₂ = 25, os valores do relatório: perto da origem imita o SwiGLU,
longe dela satura em |ativação| ≤ β₁β₂ = 100. É o que evita os outliers de
ativação que estouram em baixa precisão — `test_k3.py` verifica o limite.

### 6. Quantile Balancing

Balanceamento de carga **sem loss auxiliar**. Para cada expert calculamos o
quantil (1 − k/N) dos seus scores de roteamento; o bias de seleção anda na
direção que iguala esses limiares efetivos, e a média dos bias é mantida em
zero. Expert sobrecarregado tem limiar alto e perde bias. Sem hiperparâmetro de
peso de loss auxiliar — só a taxa de atualização do bias.

### 7. Per-Head Muon (opcional, `--optimizer muon`)

Muon com ortogonalização por Newton-Schulz aplicada **por cabeça** nas matrizes
de atenção (e por expert nos tensores empilhados do MoE). Embeddings, normas e
o roteador ficam no AdamW.

## Checkpoint incluído

`checkpoints_k3/best_model.pt` — 5,66M parâmetros totais / 3,34M ativos por
token, 6 camadas (4 KDA + 2 Gated MLA), contexto 192, 16 experts com top-2 e 2
compartilhados. Treinado 2500 iterações em CPU (~46 min):

| modelo | val loss | parâmetros |
| --- | --- | --- |
| Prova (GPT padrão, char-level) | — | 10,8M |
| V3 (RoPE + RMSNorm + SwiGLU, 76k iterações) | 2,97 | 3,9M |
| **K3 (esta versão, 2,5k iterações)** | **2,57** | 5,66M (3,34M ativos) |

Comparação indicativa, não controlada: o K3 aqui usa contexto 192 contra 256 do
V3, e o corpus/tokenizer são os mesmos. Vale como sinal de que a arquitetura
treina bem em escala pequena, não como benchmark.

A carga dos experts, que começa concentrada (máx/média ≈ 6,9 na iteração 200),
converge para ≈ 2,0 com o Quantile Balancing — sem nenhuma loss auxiliar.

Para treinar mais (ou maior):

```bash
python train_k3.py --iters 20000 --n_embd 320 --n_layer 8 --block_size 256
```

## Diferenças de escala em relação ao K3 real

| | Kimi K3 | Machado K3 |
| --- | --- | --- |
| parâmetros | 2,8T (104B ativos) | ~5–30M (~3–20M ativos) |
| camadas | 93 (69 KDA + 24 MLA) | 6–8, mesmo padrão 3:1 |
| hidden / latente MoE | 7168 / 3584 | `n_embd` / `n_embd`÷2 |
| experts | 896, top-16, 2 shared | 16–32, top-2–4, 2 shared |
| contexto | 1M tokens | 192–256 tokens |
| multimodal, RL de long-horizon | sim | fora do escopo |

O que **não** foi replicado: entrada visual, quantização MXFP4, kernels
chunkwise em Triton, e todo o pipeline de RL. O objetivo é a arquitetura.

## Uso

```bash
pip install -r requirements.txt

python test_k3.py                       # sanidade da arquitetura
python train_k3.py --iters 8000         # treino (checkpoints_k3/best_model.pt)
python train_k3.py --optimizer muon     # com Per-Head Muon
python train_k3.py --quick              # smoke test em poucos minutos
streamlit run app.py                    # comparativo Prova × V3 × K3
```

Parâmetros úteis: `--n_embd`, `--n_layer`, `--n_experts`, `--top_k_experts`,
`--block_size`, `--batch_size`, `--lr`, `--max_chars`.

O treino reaproveita `machado.txt` e o mesmo `tokenizer_bpe.pkl` do V3, para que
a comparação entre as versões seja justa. A codificação do corpus usa
pré-tokenização por palavra com cache (o algoritmo BPE é o mesmo; a divisão só
torna viável tokenizar 11 MB em segundos).

## Referências

* [Kimi K3: Open Frontier Intelligence — Technical Report](https://arxiv.org/abs/2607.24653)
* [Attention Residuals (Kimi Team)](https://arxiv.org/abs/2603.15031)
* [Kimi Linear: An Expressive, Efficient Attention Architecture (KDA)](https://arxiv.org/abs/2510.26692)
* [Notas de arquitetura do K3 — Sebastian Raschka](https://sebastianraschka.com/blog/2026/kimi-k3-architecture-notes.html)
* [Kimi K3 Architecture: KDA, AttnRes and 896 MoE Experts](https://poyo.ai/hub/kimi-k3-architecture)
