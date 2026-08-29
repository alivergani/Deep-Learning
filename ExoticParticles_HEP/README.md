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

Il risultato centrale del paper è che **una rete profonda sulle sole feature low-level raggiunge le prestazioni di una rete profonda su tutte le feature** (AUC 0.880 contro 0.885), mentre la rete shallow sulle low-level resta molto indietro (0.733).

| Tecnica | Low-level | High-level | Complete |
|---|---|---|---|
| BDT | 0.73 | 0.78 | 0.81 |
| NN shallow | 0.733 | 0.777 | 0.816 |
| DN profonda | **0.880** | 0.800 | **0.885** |

*AUC sul benchmark HIGGS, Tabella I del paper.*

## 3. Riproduzione

### 3.1 Architettura

Il paper parla di "rete a 5 strati". Il conteggio dei parametri riportato nell'articolo (279.901 per la rete profonda con 28 input) torna solo **contando lo strato di uscita fra i cinque**, cioè con 4 strati nascosti. Verificato ricalcolando i parametri; il codice usa questa interpretazione.

- **Rete profonda**: 4 strati nascosti da 300 unità, attivazione `tanh` → 279.901 parametri (28 input)
- **Rete shallow**: 1 strato nascosto da 10.000 unità → 300.001 parametri (28 input)
- Uscita: un singolo logit, senza sigmoide (applicata dentro `BCEWithLogitsLoss`, numericamente più stabile)

### 3.2 Stack di ottimizzazione originale

- inizializzazione normale con $\sigma$ = 0.1 (primo strato), 0.05 (strati intermedi), 0.001 (uscita)
- SGD, learning rate iniziale 0.05, diviso per 1.0000002 **a ogni batch** (non a ogni epoca)
- momentum in rampa lineare 0.9 → 0.99 sulle prime 200 epoche
- weight decay $10^{-5}$, mini-batch di 100 eventi
- early stopping con pazienza 10, attivo solo dopo che il momentum ha raggiunto il valore massimo

### 3.3 Griglia

Le sei configurazioni del paper — `{deep, shallow} × {low, high, complete}` — con 5 semi ciascuna.

## 4. Estensioni

### 4.1 Confronto tra stack di ottimizzazione (2014 vs oggi)

A parità di **architettura, dati e numero di eventi**, si confronta lo stack originale con uno moderno:

| | Stack 2014 | Stack moderno |
|---|---|---|
| Attivazione | `tanh` | `ReLU` |
| Inizializzazione | gaussiana con $\sigma$ fissate a mano | He, $\sigma = \sqrt{2/n_{\text{in}}}$ |
| Ottimizzatore | SGD + rampa di momentum | AdamW (lr $10^{-3}$, wd $10^{-4}$) |
| Learning rate | decadimento esponenziale per batch | `ReduceLROnPlateau` (fattore 0.5, pazienza 3) |
| Arresto | early stopping (dopo la rampa) | early stopping |

Attivazione e ottimizzatore non sono scelti indipendentemente: si sceglie *lo stack*. Una $\sigma$ tarata sulla `tanh` è sbagliata per la `ReLU` e viceversa, quindi l'inizializzazione segue automaticamente l'attivazione.

**La domanda non è quale stack raggiunga l'AUC più alta.** Con dati e capacità fissati ci si attende che entrambi arrivino più o meno allo stesso punto. La quantità interessante è la **velocità di convergenza**: quante epoche servono per arrivare a una data AUC. Il paper riporta tempi di training dell'ordine delle settimane di GPU; misurare quanto di quel costo dipendesse dalla tecnica dell'epoca, e non dal problema, è un risultato presentabile e onesto.

Una scelta metodologica importante riguarda l'early stopping. Nello stack 2014 l'arresto è bloccato fino alla fine della rampa di momentum (epoca 200): prima di allora una stasi non significa convergenza. Nello stack moderno la rampa non esiste, quindi il vincolo va rimosso — se restasse, la rete non potrebbe fermarsi prima dell'epoca 200 anche avendo già convergito, e proprio la velocità di convergenza è ciò che si vuole misurare.

Lo stack moderno viene addestrato con la **sola architettura profonda**, sui tre set di feature: le difficoltà di ottimizzazione che ReLU e He risolvono riguardano la propagazione del segnale attraverso più strati, e su una rete a un solo strato nascosto il confronto sarebbe poco informativo.

**Passaggio da iperparametri a prescrizioni.** Il filo conduttore del confronto è che lo stack moderno sostituisce numeri scelti a mano con regole derivate: la $\sigma$ di inizializzazione non si sceglie, si calcola dal numero di ingressi; il learning rate non segue una curva fissata in anticipo, ma reagisce alla perdita di validation. Questo comporta che nello stack moderno il validation set influenza il training e non solo la selezione finale — cosa che l'early stopping fa comunque in entrambi gli stack, ma vale la pena dichiararla.

### 4.2 Linear probing delle rappresentazioni interne

Il paper afferma che la rete profonda "scopre" le feature high-level, ma lo argomenta **indirettamente**, per via delle prestazioni: la rete sulle low-level fa bene quanto quella su tutto, quindi *deve* aver ricostruito l'informazione. La Supplementary Table 4 mostra separatamente che una rete può essere addestrata a *calcolare* le high-level dalle low-level, ma si tratta di una rete diversa, addestrata a quello scopo.

Il **linear probing** verifica la tesi in modo diretto:

1. Si addestra la rete profonda sul task di classificazione con le sole feature low-level.
2. Si congelano i pesi e si estraggono le attivazioni di ciascuno strato nascosto su un campione di eventi del validation set.
3. Per ogni strato si addestra una **regressione ridge** dalle attivazioni (300 numeri) alle 7 masse invarianti.
4. Si misura $R^2$ su eventi che il probe non ha visto.

Il vincolo della linearità è il punto centrale: un modello lineare non può creare informazione, solo leggerla. Se una regressione lineare sulle attivazioni predice accuratamente $m_{bb}$, quella quantità è **linearmente decodificabile** dalla rappresentazione interna — la rete l'ha costruita e resa esplicita.

**I riferimenti sono indispensabili.** Una combinazione lineare di 300 quantità può fittare parecchio anche senza che ci sia nulla di interessante. Si calcolano quindi sempre:

- **probe sulle variabili grezze in ingresso** — il livello zero: quanto è già decodificabile senza rete;
- **probe su una rete non addestrata**, stessa architettura, pesi casuali — quanto si guadagna per la sola proiezione in 300 dimensioni attraverso una non-linearità.

Il risultato è la **differenza** rispetto a questi due, e l'andamento con la profondità.

Cosa si può leggere dai risultati:

- **profilo per profondità** — se $R^2$ cresce con lo strato, si osserva la costruzione progressiva; se satura presto, la ricostruzione avviene nei primi strati e i successivi fanno altro.
- **profilo per feature** — non tutte le masse hanno lo stesso ruolo. È ragionevole attendersi che $m_{WWbb}$ e $m_{Wbb}$, che codificano le masse ipotizzate $H^0$ e $H^\pm$, siano meglio rappresentate di $m_{\ell\nu}$, che assume lo stesso valore in segnale e fondo e ha quindi potere discriminante nullo. La rete dovrebbe ricostruire ciò che le serve, non ciò che è fisicamente definibile.
- **confronto con il BDT** — la curva di apprendimento del BDT (§6) mostra dall'esterno la stessa difficoltà che il probing misura dall'interno.

Parametri: ridge con $\alpha = 1$ (le 300 attivazioni sono fortemente correlate, una regressione non regolarizzata sarebbe instabile), 200.000 eventi per stimare i coefficienti e 50.000 per misurare, presi dal **validation set** — il test resta intatto.

## 5. Configurazioni

Nove in tutto:

| Stack | Architettura | Feature set | Semi |
|---|---|---|---|
| 2014 | deep, shallow | low, high, complete | 5 |
| moderno | deep | low, high, complete | 1 |

Il seme unico dello stack moderno è **uno dei cinque** usati per la baseline: così i due stack vedono gli stessi dati nello stesso ordine, e la differenza non è attribuibile allo split o all'ordine dei batch. Non ai pesi iniziali, che sono necessariamente diversi — è proprio la modifica sotto esame.

## 6. Riferimento BDT

Implementato con `HistGradientBoostingClassifier` di scikit-learn. Con 2.6M eventi l'early stopping interno non scatta mai (la validazione interna è così grande che anche miglioramenti minuscoli risultano significativi), quindi il valore riportato dipenderebbe dal numero di alberi scelto — che il paper non specifica. Invece di sceglierlo arbitrariamente si misura **tutta la curva** AUC contro numero di alberi, fino a 3000, e si mostra dove cade il valore del paper.

Risultato: high e complete saturano entro ~500 alberi, mentre **low è ancora in salita a 3000**. È la firma della difficoltà del problema sulle variabili grezze: il BDT sta approssimando a scalini una struttura — le masse invarianti — che nelle altre configurazioni è servita già pronta. Lo stesso fenomeno che il linear probing va a guardare dall'interno della rete.

I valori ottenuti sono sistematicamente **sopra** quelli del paper (≈0.764 / 0.794 / 0.84 contro 0.73 / 0.78 / 0.81), verosimilmente per la maggiore qualità dell'implementazione moderna rispetto a TMVA. Il riferimento è quindi più severo, non più indulgente.

## 7. Dati e preprocessing

Dataset **HIGGS** dallo UCI Machine Learning Repository: 11.000.000 eventi, 29 colonne (label, 21 low-level, 7 high-level).

**Suddivisione**, come nel paper: 2.600.000 eventi di training (i primi del file), 500.000 di validation, 500.000 di test (gli ultimi, per convenzione standard del dataset). I restanti 7.4M non vengono usati: usare più dati renderebbe il confronto con il paper non interpretabile.

**Standardizzazione**, secondo lo schema dei Methods:

- colonne **strettamente positive** (pT, energia mancante, masse invarianti): $x / \bar{x}$, media 1. Non si centra a zero perché per quelle grandezze lo zero è un estremo fisico del dominio, non un valore centrale.
- **tutte le altre** ($\eta$, $\phi$, b-tag): $(x - \bar{x}) / \sigma$, media 0 e deviazione 1.

Le statistiche si calcolano **solo sul training set** e si applicano tal quali a validation e test.

Due osservazioni emerse dalla verifica:

- Il dataset UCI è già stato normalizzato dagli autori (medie ≈1 per le positive, ≈0 per le angolari), ma ricalcolare le statistiche non è ridondante: gli autori hanno normalizzato sugli 11M completi, qui si usano i primi 2.6M, e sulle masse invarianti — che hanno code lunghe — la differenza arriva al 5% ($m_{\ell\nu}$ ha media 1.05).
- I **b-tag** hanno media ≈1 nel file UCI, quindi gli autori li hanno trattati come variabili positive. Qui il loro minimo è 0, la regola empirica li classifica come "da standardizzare" e finiscono centrati a zero. Su variabili discrete la differenza è irrilevante, ma è il caso in cui la regola "minimo > 0" incontra il suo limite. Verificato che la classificazione delle colonne coincide con l'attesa fisica su tutti e tre i feature set (6, 7 e 13 colonne positive).

## 8. Struttura del repository

```
.
├── data/
│   ├── processed/          # 11M eventi, array .npy (non versionati)
│   └── processed_small/    # 100k eventi, per prove e ottimizzazione
├── src/
│   ├── features.py         # nomi e indici delle colonne
│   ├── prepare_data.py     # da CSV grezzo ad array .npy
│   ├── check_data.py       # controllo di sanità (una tantum)
│   ├── data.py             # divisione, standardizzazione, scelta feature
│   ├── models.py           # le due architetture, i due stack
│   ├── train.py            # ciclo di addestramento, i due ottimizzatori
│   ├── evaluate.py         # curve ROC, AUC, tabelle
│   ├── esperimenti.py      # lancia una configurazione su più semi
│   ├── bdt.py              # riferimento BDT
│   ├── bdt_curva.py        # AUC in funzione del numero di alberi
│   └── probing.py          # linear probing delle attivazioni
├── notebooks/              # esplorazione e figure
├── results/
│   ├── riproduzione/       # stack 2014
│   ├── moderno/            # stack moderno
│   ├── bdt/
│   └── bdt_auc/
├── results_small/          # stessa struttura, per le prove
├── runs/                   # log TensorBoard (non versionati)
└── runs_small/
```

## 9. Uso

```bash
# una configurazione, tutti i semi
python src/esperimenti.py deep low

# solo alcuni semi (per parallelizzare in tmux)
python src/esperimenti.py deep low 0 1
python src/esperimenti.py deep low 2 3 4

# stack moderno
python src/esperimenti.py deep low moderno 0

# prova rapida su dati ridotti
python src/esperimenti.py deep low moderno small 0

# rigenera il riepilogo dopo che tutti i processi sono finiti
python src/esperimenti.py deep low

# linear probing su un modello già addestrato
python src/probing.py deep low moderno 0

# monitoraggio
tensorboard --logdir=runs
```

L'argomento `small` cambia **insieme** cartella dei dati, cartella dei risultati, cartella dei log e parametri ridotti: è un unico interruttore, per non poter lanciare un training vero con i dati di prova o viceversa.

Ogni seme già completato viene saltato, quindi si può rilanciare dopo un'interruzione senza perdere lavoro.

## 10. Deviazioni consapevoli dal paper

- **Standardizzazione** — il paper calcola media e deviazione standard sull'intero insieme train+test. Qui le statistiche sono stimate esclusivamente sul training set, per evitare data leakage.
- **BDT** — il paper usa TMVA; qui `HistGradientBoostingClassifier`, quindi i valori assoluti non sono direttamente confrontabili (e risultano migliori).
- **Numero di semi** — 5 per la riproduzione, 1 per lo stack moderno. Con un solo seme la deviazione standard non è definita e viene riportata come tale, non come zero.
- **Nessuna cross-validation** — con 500.000 eventi di validation e altrettanti di test le stime di AUC sono già molto precise; la variabilità rilevante è quella fra inizializzazioni, catturata dai semi.
- **Selezione dell'epoca** — i pesi finali sono quelli dell'epoca a perdita di validation minima. Questo introduce un lieve bias ottimistico sull'AUC di test. Nota: perdita minima e AUC massima non cadono necessariamente sulla stessa epoca; il criterio è comunque identico per i due stack, quindi il confronto resta pulito.
- **Ottimizzazione degli iperparametri** — solo per lo stack moderno, su `processed_small`, limitata a learning rate e weight decay. L'architettura resta ai valori del paper: ottimizzarla renderebbe non interpretabile il confronto fra stack, e l'architettura ottima dipende dalla quantità di dati, quindi non si trasferirebbe dalla scala ridotta. Che il solo stack moderno venga ottimizzato è uno sbilanciamento voluto: si confrontano due *pratiche* complete, non due ottimizzatori nudi.
- **Budget computazionale** — numero di semi ed epoche possono essere ridotti; ogni riduzione è documentata insieme ai risultati.
- **Pre-training** — il pre-training con autoencoder non è riprodotto: il paper stesso riporta che non produceva miglioramenti apprezzabili.

## 11. Punti critici del benchmark

Da discutere nella presentazione, indipendentemente dai risultati:

- il rapporto segnale/fondo è 50/50 per costruzione, molto lontano dalle proporzioni reali di un esperimento;
- la simulazione usa DELPHES, una simulazione veloce del rivelatore, non una simulazione completa;
- non sono incluse incertezze sistematiche, che in un'analisi reale dominano il risultato.

## 12. Riferimenti

1. P. Baldi, P. Sadowski, D. Whiteson, *Searching for Exotic Particles in High-Energy Physics with Deep Learning*, Nature Communications 5, 4308 (2014). arXiv:1402.4735
2. K. He et al., *Delving Deep into Rectifiers*, ICCV 2015. arXiv:1502.01852
3. Dataset HIGGS, UCI Machine Learning Repository.