# Anomaly Detection sul dataset R&D delle LHC Olympics 2020

Progetto per il corso di Deep Learning (Prof. S. Carrazza, Universita' degli Studi di Milano).

Implementazione e studio del metodo di *anomaly detection* risonante proposto in
**J. Collins, K. Howe, B. Nachman, "Anomaly Detection for Resonant New Physics with
Machine Learning"**, [arXiv:1805.02664](https://arxiv.org/abs/1805.02664).

L'idea: se un segnale di nuova fisica e' localizzato in una variabile risonante
($m_{JJ}$) su un fondo liscio, si puo' addestrare un classificatore a distinguere
la *signal region* dalle *sidebands* usando solo variabili di sottostruttura.
Per il lemma di Neyman-Pearson questo classificatore approssima una riscalatura
monotona del likelihood ratio segnale/fondo, quindi individua le stesse superfici
di decisione di un classificatore supervisionato -- senza aver mai visto
un'etichetta vera.

---

## Come riprodurre

```bash
git clone <url> cwola-hunting && cd cwola-hunting

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_data.sh        # ~80 MB da Zenodo
python scripts/prepare_data.py       # -> data/processed/*.parquet

jupyter lab notebooks/               # eseguire i notebook in ordine
```

Test rapido del codice, senza bisogno dei dati (usa eventi sintetici):

```bash
python tests/test_features.py
```

---

## I dati

Dataset R&D delle LHC Olympics 2020 (Kasieczka, Nachman, Shih),
[Zenodo record 6466204](https://zenodo.org/records/6466204),
DOI `10.5281/zenodo.6466204`, licenza CC-BY-4.0.

1M eventi QCD dijet come fondo e 100k eventi $W' \to XY$ con $X, Y \to qq$,
masse $m_{W'} = 3.5$ TeV, $m_X = 500$ GeV, $m_Y = 100$ GeV. Generazione con
Pythia8 + Delphes 3.4.1, senza pileup ne' MPI, con trigger su un singolo fat-jet
$R = 1$ a $p_T > 1.2$ TeV.

| File | Dim. | Uso |
|---|---|---|
| `events_anomalydetection_v2.features.h5` | 74 MB | dataset di lavoro: jet gia' clusterizzati, 15 colonne |
| `events_anomalydetection_Z_XY_qqq.features.h5` | 5 MB | segnale alternativo 3-prong, per il test di model-agnosticity |
| `pythia_RnD_*.cmnd`, `delphes_card_RnD.dat` | pochi kB | documentazione della simulazione, citata nel report |
| `events_anomalydetection_v2.h5` | 2.9 GB | costituenti grezzi; **opzionale**, solo per l'estensione CNN / deep sets |

`data/raw/` e' immutabile: nessuno script ci scrive dentro. Tutto cio' che il
codice produce finisce in `data/processed/`, che si puo' cancellare e rigenerare
in qualsiasi momento.

---

## Struttura

```
config/default.yaml     finestre SR/SB, path, seed, iperparametri
src/                    funzioni pure: nessun side effect, nessuna scrittura su disco
  config.py             caricamento YAML, risoluzione dei path
  io.py                 lettura degli .h5 LHCO, schema delle colonne, reader a blocchi
  features.py           m_JJ, ordinamento in massa, tau21/tau32, standardizzazione
scripts/                orchestrazione: leggono la config, chiamano src/, scrivono su disco
  download_data.sh      download da Zenodo + verifica md5
  prepare_data.py       raw/*.h5 -> processed/*.parquet, con controlli di sanita'
notebooks/              analisi ed esposizione dei risultati
tests/                  test su dati sintetici, girano senza il dataset reale
results/runs/           un JSON per run: config + metriche
results/figures/        PDF vettoriali per il report
```

---

## Variabili

**Risonante** (definisce le finestre, non e' mai un input della rete):

$$m_{JJ} = \sqrt{(E_1+E_2)^2 - |\vec p_1 + \vec p_2|^2}$$

**Ausiliarie $Y$** (gli input del classificatore):

$$Y = \left(m_{J_A},\ \Delta m_J = m_{J_B} - m_{J_A},\ \tau_{21}^{J_A},\ \tau_{21}^{J_B}\right)$$

con i due jet ordinati **per massa** ($m_{J_A} < m_{J_B}$) e non per $p_T$:
elimina l'ambiguita' di etichettatura e riduce la correlazione residua con $m_{JJ}$.
Il set esteso aggiunge $\tau_{32}$ per i segnali a 3 prong.

**Target del training CWoLa:** $y = 1$ se $m_{JJ} \in$ SR, $y = 0$ se $m_{JJ} \in$ SB.
La label di verita' segnale/fondo **non entra mai in training**: e' usata solo per
valutare a posteriori (ROC, AUC) e per costruire il mix a $S/B$ controllato.

---

## Stato

- [x] Fase 0 - lettura dati, feature, controlli di correlazione
- [ ] Fase 1 - benchmark pienamente supervisionato (upper bound)
- [ ] Fase 2 - CWoLa con k-fold
- [ ] Fase 3 - bump hunt e p-valori
- [ ] Fase 4 - confronto con autoencoder
- [ ] Fase 5 - scansione in S/B
