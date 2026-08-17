# Deep Learning per la ricerca di particelle esotiche

Riproduzione ed estensione di **Baldi, Sadowski & Whiteson (2014)**, *Searching for Exotic Particles in High-Energy Physics with Deep Learning* ([arXiv:1402.4735](https://arxiv.org/abs/1402.4735)).

Progetto d'esame per il corso di Deep Learning — Università degli Studi di Milano.

---

## 1. Il problema fisico

Ai collisori adronici la scoperta di nuove particelle è un problema di **classificazione segnale/fondo**: eventi rarissimi di interesse vanno separati da un fondo enormemente più abbondante con stati finali identici. Il rapporto di verosimiglianza è per il teorema di Neyman–Pearson la quantità ottimale di discriminazione, ma non è esprimibile analiticamente: si ricorre quindi a simulazioni Monte Carlo e a classificatori di machine learning.

Il benchmark **HIGGS** riguarda la produzione di bosoni di Higgs esotici,

$$ gg \to H^0 \to W^\mp H^\pm \to W^\mp W^\pm h^0 \to W^\mp W^\pm b\bar{b} $$

contro un fondo di coppie di quark top che produce lo stesso stato finale $W^\mp W^\pm b\bar{b}$ ma con cinematica diversa. Gli eventi sono generati con MadGraph5 + Pythia + Delphes a 8 TeV, con $m_{H^0} = 425$ GeV e $m_{H^\pm} = 325$ GeV. Il dataset pubblico contiene 11 milioni di eventi.

## 2. La tesi del paper

Ogni evento è descritto da due gruppi di variabili:

- **21 feature low-level** — le misure dirette del rivelatore: momento trasverso, pseudorapidità e angolo azimutale del leptone e dei quattro jet, b-tagging, energia trasversa mancante.
- **7 feature high-level** — masse invarianti ricostruite a mano ($m_{jj}$, $m_{jjj}$, $m_{\ell\nu}$, $m_{j\ell\nu}$, $m_{bb}$, $m_{Wbb}$, $m_{WWbb}$), costruite dai fisici per catturare l'esistenza degli stati intermedi risonanti. Sono funzioni non lineari e non banali delle low-level.

La pratica consolidata in fisica delle alte energie era di alimentare classificatori *shallow* (reti a un solo strato nascosto, boosted decision trees) con le feature high-level, perché i modelli poco profondi non riescono a ricostruirle da soli.

Il risultato centrale del paper è che **una rete profonda sulle sole feature low-level raggiunge le prestazioni di una rete profonda su tutte le feature** (AUC 0.880 contro 0.885), mentre la rete shallow sulle low-level resta molto indietro (0.733). La rete profonda, in altre parole, **scopre da sola l'informazione contenuta nelle masse invarianti**, rendendo superflua la costruzione manuale delle feature.

| Tecnica | Low-level | High-level | Complete |
|---|---|---|---|
| BDT | 0.73 | 0.78 | 0.81 |
| NN shallow | 0.733 | 0.777 | 0.816 |
| DN (5 layer) | **0.880** | 0.800 | **0.885** |

*AUC sul benchmark HIGGS, Tabella I del paper.*

## 3. Obiettivo di riproduzione

Riprodurre la griglia completa `{shallow, deep} × {low-level, high-level, complete}` sul dataset HIGGS, con la configurazione originale:

- MLP a 5 strati nascosti da 300 unità, attivazione `tanh`
- inizializzazione normale con $\sigma$ = 0.1 (primo strato), 0.05 (strati intermedi), 0.001 (output)
- SGD con learning rate 0.05, decadimento esponenziale per batch, momentum in rampa lineare 0.9 → 0.99 sulle prime 200 epoche
- weight decay $10^{-5}$, mini-batch di 100 eventi, early stopping su validation set
- valutazione via ROC / AUC su 500.000 eventi di test, media su più inizializzazioni casuali

## 4. Estensioni proposte

### 4.1 Confronto tra stack metodologici (2014 vs oggi)

Il paper fotografa lo stato dell'arte del 2014. A parità di architettura e di dati, si confronta lo stack originale con quello moderno:

| | Stack 2014 | Stack moderno |
|---|---|---|
| Attivazione | `tanh` | `ReLU` / `GELU` |
| Ottimizzatore | SGD + momentum | Adam / AdamW |
| Regolarizzazione | weight decay | dropout + weight decay |
| Normalizzazione | solo degli input | batch normalization |
| Arresto | early stopping | early stopping + LR scheduling |

La domanda: **quanto del divario shallow/deep osservato nel 2014 dipende dall'architettura profonda e quanto dalle difficoltà di ottimizzazione dell'epoca?** Il paper attribuisce esplicitamente il problema alla diffusione del gradiente con attivazioni saturanti; se lo stack moderno migliora la rete shallow più della profonda, parte del risultato originale è imputabile alla tecnica di training e non solo alla capacità rappresentativa.

### 4.2 Learning curve: AUC in funzione di $N_{\text{train}}$

Il paper addestra su un numero fisso di eventi (2.6M) e non indaga la dipendenza dalla dimensione del campione. L'estensione misura l'AUC in funzione di $N_{\text{train}}$ (da $10^4$ a $2.6 \times 10^6$) separatamente sui tre set di feature.

L'ipotesi da verificare: la "scoperta automatica delle feature" è un fenomeno che **richiede dati**. Ci si attende che a piccolo $N_{\text{train}}$ le feature high-level restino vantaggiose — la conoscenza fisica incorporata a mano compensa la scarsità di esempi — e che il vantaggio si annulli solo oltre una certa soglia. Individuare quella soglia quantifica il valore informativo del dominio expertise in termini di eventi simulati equivalenti, che è una domanda di interesse pratico: generare eventi Monte Carlo costa tempo di calcolo.

### 4.3 Linear probing delle rappresentazioni interne

Il paper afferma che la rete profonda "scopre" le feature high-level, ma lo argomenta **indirettamente**, per via delle prestazioni: la rete sulle low-level fa bene quanto quella su tutto, quindi *deve* aver ricostruito l'informazione. La Supplementary Table 4 mostra separatamente che una rete può essere addestrata a *calcolare* le high-level dalle low-level, ma si tratta di una rete diversa, addestrata a quello scopo.

Il **linear probing** verifica la tesi in modo diretto. La procedura:

1. Si addestra la rete profonda sul task di classificazione con le sole feature low-level.
2. Si congelano i pesi e si estraggono le attivazioni di ciascuno strato nascosto $h^{(1)}, \dots, h^{(5)}$ su un campione di eventi.
3. Per ogni strato e per ciascuna delle 7 feature high-level si addestra una **regressione lineare** (il "probe") dalle attivazioni al valore della feature.
4. Si misura $R^2$ (o MSE) del probe.

L'idea è che un classificatore lineare non può creare informazione, solo leggerla. Se una regressione lineare sulle attivazioni dello strato $k$ predice accuratamente $m_{bb}$, allora quella quantità è **linearmente decodificabile** dalla rappresentazione interna: la rete l'ha effettivamente costruita e resa esplicita, non è un artefatto della metrica di classificazione.

Cosa si può leggere dai risultati:

- **profilo per profondità** — se $R^2$ cresce con lo strato, si osserva la costruzione progressiva della feature; se satura presto, la ricostruzione avviene nei primi strati e i successivi fanno altro.
- **profilo per feature** — non tutte le masse invarianti hanno lo stesso ruolo. È ragionevole attendersi che $m_{WWbb}$ e $m_{Wbb}$, che codificano direttamente le masse ipotizzate $H^0$ e $H^\pm$, siano meglio rappresentate di $m_{\ell\nu}$, che assume lo stesso valore in segnale e fondo e ha quindi potere discriminante nullo. La rete dovrebbe ricostruire ciò che le serve, non ciò che è fisicamente definibile.
- **controllo** — lo stesso probe applicato alla rete shallow e a una rete non addestrata (pesi casuali) fornisce il riferimento: quanta parte della decodificabilità è dovuta all'apprendimento e quanta alla semplice proiezione casuale in dimensione alta.

Il linear probing collega dunque il risultato del paper a una domanda di interpretabilità, e trasforma un'affermazione sulle prestazioni in una misura sulle rappresentazioni.

## 5. Dati

Dataset **HIGGS** dallo UCI Machine Learning Repository: 11.000.000 eventi, 29 colonne (label, 21 low-level, 7 high-level). Gli ultimi 500.000 eventi costituiscono il test set secondo la convenzione standard.

## 6. Struttura del repository

```
.
├── data/           # dataset grezzo e array preprocessati (non versionati)
├── src/            # preparazione dati, modelli, training, valutazione
├── notebooks/      # esplorazione, esperimenti, figure
├── results/        # metriche, curve ROC, checkpoint
```

## 7. Deviazioni consapevoli dal paper

Elencate qui e discusse nella relazione:

- **Standardizzazione** — il paper calcola media e deviazione standard sull'intero insieme train+test. Qui le statistiche sono stimate esclusivamente sul training set, per evitare data leakage.
- **BDT** — il paper usa TMVA; qui si impiega un'implementazione equivalente in scikit-learn, quindi i valori assoluti non sono direttamente confrontabili.
- **Budget computazionale** — numero di inizializzazioni casuali ed epoche di training possono essere ridotti rispetto all'originale; ogni riduzione è documentata insieme ai risultati.
- **Pre-training** — il pre-training con autoencoder non è riprodotto: il paper stesso riporta che non produceva miglioramenti apprezzabili.

## 8. Riferimenti

1. P. Baldi, P. Sadowski, D. Whiteson, *Searching for Exotic Particles in High-Energy Physics with Deep Learning*, Nature Communications 5, 4308 (2014). arXiv:1402.4735
2. Dataset HIGGS, UCI Machine Learning Repository.
