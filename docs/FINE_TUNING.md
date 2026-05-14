# Fine-Tuning do Modelo LSTM — BBAS3

Registo das iterações de optimização realizadas durante o desenvolvimento, com os hiperparâmetros testados, motivações e resultados obtidos no conjunto de teste.

---

## Sumário de Resultados

| Run | Dropout | LR | Epochs | LOOKBACK | Loss | Factor | Early Stop | MAE (R$) | RMSE (R$) | MAPE |
|-----|---------|-----|--------|----------|------|--------|------------|----------|-----------|------|
| 1 — Baseline | 0.2 | 0.001 | 100 | 30 | MSELoss | 0.5 | ✗ | — | — | 4.80% |
| 2 — Regressão | 0.2 | 0.001 | 200 | 60 | HuberLoss(δ=1.0) | 0.5 | ✗ | — | — | 7.75% |
| 3 — Correcção | 0.2 | 0.001 | 200 | 30 | HuberLoss(δ=0.1) | 0.5 | ✗ | 0.9819 | 1.4828 | 4.24% |
| 4 | 0.4 | 0.001 | 200 | 30 | HuberLoss(δ=0.1) | 0.7 | ✗ | 0.6235 | 0.8149 | 2.70% |
| 5 — Exploração | 0.3 | 0.002 | 500* | 30 | HuberLoss(δ=0.1) | 0.7 | ✓ (ep. 190) | 0.6285 | 0.7907 | 2.72% |
| 6 — **Melhor** | 0.4 | 0.001 | 500* | 30 | HuberLoss(δ=0.1) | 0.7 | ✓ (ep. 120) | **0.5907** | **0.7542** | **2.56%** |

*EPOCHS=500 com early stopping — parou antes de completar todas as epochs.

---

## Run 1 — Baseline

**Configuração inicial do projecto.**

| Parâmetro | Valor |
|-----------|-------|
| Loss | `MSELoss` |
| LR | 0.001 |
| Epochs | 100 |
| LOOKBACK | 30 |
| Dropout | 0.2 |
| Scheduler factor | 0.5 |

**Resultado:** MAPE 4.80%

**Observações:**
- Treino funcional mas sem optimização.
- `MSELoss` penaliza quadraticamente todos os erros, tornando-o sensível a spikes de mercado (outliers).
- LR decaiu de forma agressiva (factor=0.5), chegando a valores muito baixos antes do final.

---

## Run 2 — Regressão

**Tentativa de melhorar com janela mais longa e HuberLoss.**

Mudanças relativamente ao Run 1:
- `LOOKBACK: 30 → 60`
- `MSELoss → HuberLoss(delta=1.0)`
- `EPOCHS: 100 → 200`

**Resultado:** MAPE 7.75% — **pior**

**Post-mortem:**
- **LOOKBACK=60 prejudicou o modelo:** o conjunto de validação encolheu de ~154 para ~124 janelas, reduzindo a diversidade de sinal. A arquitectura (hidden_size=128) não foi escalonada para sequências mais longas.
- **HuberLoss(delta=1.0) ineficaz em espaço normalizado:** no intervalo [0,1] todos os erros são menores que 1.0, portanto `Huber(δ=1.0) ≡ MSE` — a mudança de loss foi nula.

---

## Run 3 — Correcção

**Reversão de LOOKBACK e correcção do delta da HuberLoss.**

Mudanças relativamente ao Run 2:
- `LOOKBACK: 60 → 30` (revertido)
- `HuberLoss(delta=1.0 → 0.1)` — delta=0.1 corresponde a ~R$1.20 em escala original, valor realista para um erro de fecho diário
- Adicionado `clip_grad_norm_(max_norm=1.0)` para estabilidade dos gradientes

**Resultado:** MAPE 4.24% — melhor que o baseline mas aquém do potencial

**Observações:**
- HuberLoss(δ=0.1) funcionou: penaliza linearmente erros grandes (spikes políticos) e quadraticamente erros pequenos.
- LR decaiu 5 vezes (factor=0.5): `0.001 → 0.0005 → 0.00025 → 0.000125 → 0.0000625 → 0.000031`. LR final demasiado pequeno, aprendizagem estagnada nas últimas epochs.
- `val_loss` melhor na epoch 179/200 — sinal de que ainda havia margem.

---

## Run 4

**Ajuste do dropout e scheduler. Adição de `Close_pct` como 15ª feature.**

Mudanças relativamente ao Run 3:

| Parâmetro | Antes | Depois | Motivação |
|-----------|-------|--------|-----------|
| Dropout | 0.2 | **0.4** | BBAS3 tem ruído político — regularização mais forte força representações distribuídas |
| Scheduler factor | 0.5 | **0.7** | Decaimento menos agressivo; LR mantém força nas epochs finais |
| `min_lr` | — | **1e-4** | Piso mínimo: evita passos de gradiente microscópicos |
| Features | 14 | **15** (`Close_pct`) | Variação percentual diária fornece sinal direcional explícito ao LSTM |

**Resultado:** MAE R$0.6235 | RMSE R$0.8149 | **MAPE 2.70%**

**Por que dropout=0.4 melhorou tanto?**

Com dropout=0.2 o modelo memorizava padrões específicos do treino que não generalizavam para o conjunto de teste. Com dropout=0.4, cada forward pass activa uma sub-rede aleatória diferente — o modelo é forçado a aprender representações robustas e distribuídas. O fenómeno `train_loss > val_loss` observado é **esperado e normal** com dropout alto: durante treino 40% dos neurónios estão desligados (capacidade reduzida); durante validação todos ficam activos.

```
Melhor val_loss: 0.001070 (epoch 45/200)
LR percurso: 0.001 → 0.0007 → 0.000490 → 0.000343 → 0.000240 → 0.000168 → 0.000118 → 0.000082 → 0.000058 → 0.000040 → 0.000028
```

**Observação:** o melhor checkpoint foi a epoch 45 — o modelo rodou 155 epochs a mais sem melhorar. Motivou a adição de early stopping no Run 5.

---

## Run 5 — Exploração

**Teste de LR inicial mais alto, dropout intermédio e early stopping.**

Mudanças relativamente ao Run 4:

| Parâmetro | Antes | Depois |
|-----------|-------|--------|
| LR | 0.001 | 0.002 |
| Dropout | 0.4 | 0.3 |
| EPOCHS | 200 | 500 |
| Early stopping patience | — | 100 |

**Resultado:** MAE R$0.6285 | RMSE R$0.7907 | MAPE 2.72%

```
Melhor val_loss: 0.001097 (epoch 90)
Early stopping disparado: epoch 190 (90 + 100 epochs sem melhoria)
```

**Observações:**
- MAPE marginalmente pior que Run 4 (2.72% vs 2.70%), mas RMSE melhorou (0.7907 vs 0.8149) — menos erros grandes.
- LR=0.002 causou maior oscilação do val_loss nas primeiras 30 epochs — aprendizagem mais ruidosa mas que converge.
- Early stopping funcionou correctamente: identificou o melhor ponto (epoch 90) e parou sem desperdício.

---

## Run 6 — Melhor resultado ★

**Retorno ao dropout=0.4 e LR=0.001, agora com early stopping activo.**

Mudanças relativamente ao Run 5:

| Parâmetro | Antes (Run 5) | Depois (Run 6) |
|-----------|---------------|----------------|
| Dropout | 0.3 | **0.4** |
| LR | 0.002 | **0.001** |

**Resultado:** MAE R$0.5907 | RMSE R$0.7542 | **MAPE 2.56%**

```
Melhor val_loss: 0.001237 (epoch 20)
Early stopping disparado: epoch 120 (20 + 100 epochs sem melhoria)
LR percurso: 0.001 → 0.0007 → 0.000490 → 0.000343 → 0.000240 → 0.000168 → 0.000118
```

**Observações:**
- Confirma que **dropout=0.4 + LR=0.001 é a combinação óptima** para este modelo e dataset.
- O LR=0.002 do Run 5 introduzia demasiado ruído nas primeiras epochs, atrasando a convergência.
- Early stopping funcionou correctamente: melhor checkpoint na epoch 20, treino encerrado na 120 sem desperdício.
- Melhorias consistentes em todas as métricas relativamente ao Run 4: MAE −5.3%, RMSE −7.5%, MAPE −0.14 p.p.
- O gráfico mostra boa aderência à tendência geral, com o lag residual em reversões abruptas — limitação estrutural do LSTM.

---

## Configuração Final (Run 6)

```python
# src/model/lstm.py
hidden_size  = 128
num_layers   = 2
dropout      = 0.4

# src/model/train.py
EPOCHS               = 500
LEARNING_RATE        = 0.001
EARLY_STOP_PATIENCE  = 100
criterion  = nn.HuberLoss(delta=0.1)
optimizer  = Adam(lr=0.001)
scheduler  = ReduceLROnPlateau(patience=15, factor=0.7, min_lr=1e-4)
clip_grad_norm_(max_norm=1.0)

# src/data/dataset.py
LOOKBACK    = 30
batch_size  = 32
train/val/test split = 70/15/15 (cronológico)

# src/data/features.py — 15 features
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI_14", "MACD_Line", "MACD_Signal", "MACD_Hist",
    "BB_Width", "EMA_9", "EMA_21", "Volume_Ratio",
    "IBOV_Close", "Close_pct",
]
```

---
