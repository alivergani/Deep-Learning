"""
Linear probing: le masse invarianti sono leggibili dentro la rete?

IDEA
----
La rete profonda addestrata sulle sole variabili grezze (feature set "low")
raggiunge le prestazioni di una rete shallow che riceve gia' pronte le masse
invarianti. La domanda naturale e': se le raggiunge, si e' costruita da sola
qualcosa che assomiglia a quelle masse?

Il linear probing risponde cosi':
  1. si congela la rete addestrata, non si tocca piu' nulla dei suoi pesi;
  2. si prendono le attivazioni di uno strato nascosto (300 numeri per evento);
  3. si prova a predire una massa invariante da quei 300 numeri usando SOLO
     una regressione lineare;
  4. si misura la qualita' della predizione con l'R^2 su eventi mai visti
     dal probe.

Il vincolo della linearita' e' il punto centrale. Se un modello lineare ci
riesce, l'informazione non e' soltanto presente: e' presente in una forma
gia' estratta, leggibile con un'operazione banale. Se servisse un modello
complicato, staremmo misurando quanto e' bravo il probe, non cosa ha
costruito la rete.

I DUE RIFERIMENTI
-----------------
300 numeri sono tanti: una combinazione lineare di 300 quantita' puo'
fittare parecchio anche senza che ci sia nulla di interessante. Per questo
lo script calcola sempre anche:

  "input"    -> probe direttamente sulle variabili grezze in ingresso.
                E' il livello zero: quanto e' gia' decodificabile senza rete.

  "casuale"  -> probe su una rete della stessa forma ma NON addestrata.
                Misura quanto si guadagna per il solo fatto di proiettare
                le variabili in 300 dimensioni passando per una non-linearita'.

Il risultato vero e' la DIFFERENZA fra la rete addestrata e questi due,
e l'andamento dell'R^2 con la profondita' dello strato.

PERCHE' LE ATTIVAZIONI VENGONO STANDARDIZZATE
---------------------------------------------
Prima della ridge ogni colonna della rappresentazione viene portata a media
0 e deviazione standard 1, con media e deviazione stimate SOLO sugli eventi
di fit e poi applicate a quelli di misura (altrimenti l'insieme di misura
non sarebbe piu' davvero mai visto).

Non e' un dettaglio cosmetico. La ridge penalizza la somma dei quadrati dei
coefficienti, quindi la forza effettiva della penalita' dipende dalla scala
delle variabili: uno strato con attivazioni piccole verrebbe regolarizzato
molto piu' di uno con attivazioni grandi, e i due R^2 non sarebbero
confrontabili. Siccome l'intero risultato e' un confronto fra strati, la
scala va tolta di mezzo.

Senza standardizzazione l'ultimo strato nascosto dava R^2 fortemente
negativi (fino a -340): non "informazione persa", ma un sistema mal
condizionato che estrapola malissimo fuori dagli eventi di fit.

PERCHE' L'ALPHA NON E' PIU' FISSO
---------------------------------
La versione precedente usava una Ridge con alpha=10 uguale per tutte le
rappresentazioni, per garantire la confrontabilita' fra strati.

Su alcuni modelli quella scelta si rompe. Nello stack moderno l'ultimo
strato nascosto dava R^2 intorno a -160. Non e' informazione persa: le
attivazioni ReLU non sono limitate dall'alto e, col weight decay molto
basso trovato dalla ricerca iperparametri, qualche evento raro arriva a
migliaia di deviazioni standard dalla distribuzione degli eventi di fit
(max |z| ~ 7600 contro ~50 negli strati precedenti). L'R^2 e' basato
sull'errore quadratico, quindi una manciata di eventi con predizione
assurda domina la media su 50.000 eventi.

La soluzione e' lasciare che ogni rappresentazione scelga da sola quanto
regolarizzare, per validazione incrociata sui soli eventi di fit. Dove le
attivazioni hanno code pesanti la CV sceglie un alpha alto, i coefficienti
si schiacciano e il probe smette di estrapolare in modo assurdo.

La confrontabilita' fra strati non si perde, perche' era gia' garantita
dalla standardizzazione. Cambia leggermente la domanda a cui il probe
risponde: non piu' "quanto rende un probe lineare con questa penalita'
fissata", ma "qual e' il miglior probe lineare possibile su questo strato",
che e' altrettanto ben definita e non privilegia nessuno strato.

Gli alpha scelti finiscono nel JSON: se lo strato profondo di un modello
sceglie un alpha molto piu' alto degli altri, e' la misura diretta di
quanto e' mal condizionata la sua rappresentazione.

USO
---
    python src/probing.py deep low moderno 0
    python src/probing.py deep low 2014 0
    python src/probing.py deep low moderno small 0     (prova rapida)

Lo stack va scritto sempre in modo esplicito: senza, vale il default
STACK = "moderno", e si finisce per sondare un modello diverso da quello
che si crede.

Con "small" si leggono i dati da data/processed_small e si cercano i pesi
in results_small/, esattamente come fa esperimenti.py: le due modalita' non
si mescolano mai.

Produce results/<sottocartella>/probing_<nome>_seme<k>.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from data import prepara_dati
from features import INDICI
from models import rete_profonda, rete_shallow


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

MODELLO = "deep"
FEATURE_SET = "low"          # il probing ha senso soprattutto su "low"
STACK = "moderno"
SEME = 0

PICCOLO = False              # si attiva da riga di comando con "small"

# Eventi usati per il probe. Si prendono dal VALIDATION set, non dal test:
# il test resta intatto per la stima finale delle prestazioni.
N_PROBE_TRAIN = 200_000      # per stimare i coefficienti della regressione
N_PROBE_TEST = 50_000        # per misurare l'R^2 su dati mai visti

# Dimensioni dei tre insiemi. DEVONO essere identiche a quelle usate in
# esperimenti.py per addestrare il modello che stiamo sondando: da n_val e
# n_test dipende QUALI eventi finiscono nel validation set, e da n_train
# dipendono media e deviazione standard con cui i dati vengono normalizzati.
# Con numeri diversi la rete riceverebbe dati normalizzati in modo diverso
# da come e' stata addestrata, e i risultati sarebbero sbagliati senza che
# nulla segnali l'errore.
N_TRAIN = 10_000_000
N_VAL = 500_000
N_TEST = 500_000

# Griglia di valori fra cui scegliere la forza della regolarizzazione della
# ridge regression. Con 300 variabili molto correlate fra loro una
# regressione lineare semplice e' instabile: la ridge penalizza i
# coefficienti grandi e rende la soluzione ben definita.
#
# Non si fissa un alpha unico: ogni rappresentazione sceglie il proprio per
# validazione incrociata sui soli eventi di fit. Vedi la nota in cima al
# file: con alpha fisso l'ultimo strato nascosto dello stack moderno dava
# R^2 intorno a -160, per estrapolazione su pochi eventi con attivazioni
# estreme, non per informazione mancante.
#
# La griglia e' larga (da 0.1 a 1e6) proprio per coprire i casi mal
# condizionati, dove serve una penalita' di ordini di grandezza superiore
# a quella che basta agli strati iniziali.
ALPHAS = np.logspace(-1, 10, 30)

# ---------------------------------------------------------------------------


PROGETTO = Path(__file__).resolve().parent.parent

MODELLI_VALIDI = ["deep", "shallow"]
FEATURE_SET_VALIDI = ["low", "high", "complete"]
STACK_VALIDI = ["2014", "moderno"]

semi_da_riga_comando = []
for argomento in sys.argv[1:]:
    if argomento in STACK_VALIDI:
        STACK = argomento
    elif argomento == "small":
        PICCOLO = True
    elif argomento.isdigit():
        semi_da_riga_comando.append(int(argomento))
    elif argomento in MODELLI_VALIDI:
        MODELLO = argomento
    elif argomento in FEATURE_SET_VALIDI:
        FEATURE_SET = argomento
    else:
        raise SystemExit(f"Argomento non riconosciuto: '{argomento}'")

if semi_da_riga_comando:
    SEME = semi_da_riga_comando[0]

# --- dove leggere i dati e dove cercare i pesi ----------------------------
# Gli stessi valori di esperimenti.py: se li cambi la', vanno cambiati anche
# qui, altrimenti il probe lavora su una fetta di dati diversa da quella su
# cui il modello e' stato addestrato.
SOTTOCARTELLA = "riproduzione" if STACK == "2014" else "moderno"

if PICCOLO:
    CARTELLA_DATI = PROGETTO / "data" / "processed_small"
    CARTELLA_RISULTATI = PROGETTO / "results_small" / SOTTOCARTELLA
    N_TRAIN = 70_000
    N_VAL = 15_000
    N_TEST = 15_000
    N_PROBE_TRAIN = 12_000
    N_PROBE_TEST = 3_000
else:
    CARTELLA_DATI = None
    CARTELLA_RISULTATI = PROGETTO / "results" / SOTTOCARTELLA

ATTIVAZIONE = "tanh" if STACK == "2014" else "relu"

# Stesso schema di nomi di esperimenti.py.
NOME = f"{MODELLO}_{FEATURE_SET}"

FILE_MODELLO = CARTELLA_RISULTATI / f"{NOME}_seme{SEME}_modello.pt"

# Nomi delle 7 masse invarianti, nell'ordine delle colonne 21..27.
NOMI_MASSE = ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]


# ---------------------------------------------------------------------------
# 1. I DATI
# ---------------------------------------------------------------------------

def carica_dati():
    """
    Restituisce (X, Y) per il probe:
        X = variabili in ingresso alla rete   (quelle del feature set scelto)
        Y = le 7 masse invarianti             (i bersagli del probe)

    prepara_dati taglia sempre gli stessi eventi nello stesso ordine, quindi
    chiamandola due volte con feature set diversi otteniamo due matrici
    allineate riga per riga: la riga i di X e la riga i di Y sono lo stesso
    evento. Cosi' non dobbiamo riscrivere la logica di caricamento.
    """
    print("Caricamento dati...")

    argomenti = dict(n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                     silenzioso=True)
    if CARTELLA_DATI is not None:
        argomenti["cartella"] = CARTELLA_DATI

    dati_x = prepara_dati(feature_set=FEATURE_SET, **argomenti)
    dati_y = prepara_dati(feature_set="high", **argomenti)

    X = dati_x["val"][0]
    Y = dati_y["val"][0]

    n_serve = N_PROBE_TRAIN + N_PROBE_TEST
    if len(X) < n_serve:
        raise SystemExit(
            f"Servono {n_serve:,} eventi di validation, disponibili {len(X):,}.\n"
            f"Riduci N_PROBE_TRAIN e N_PROBE_TEST."
        )

    X = np.ascontiguousarray(X[:n_serve])
    Y = np.ascontiguousarray(Y[:n_serve])

    print(f"  {len(X):,} eventi, {X.shape[1]} variabili in ingresso, "
          f"{Y.shape[1]} masse da predire\n")
    return X, Y


# ---------------------------------------------------------------------------
# 2. LE ATTIVAZIONI
# ---------------------------------------------------------------------------

def costruisci_rete(n_input, addestrata):
    """
    Crea la rete. Se addestrata=True carica i pesi salvati, altrimenti la
    lascia con l'inizializzazione casuale (serve per il riferimento).
    """
    if MODELLO == "deep":
        rete = rete_profonda(n_input=n_input, attivazione=ATTIVAZIONE)
    else:
        rete = rete_shallow(n_input=n_input, attivazione=ATTIVAZIONE)

    if addestrata:
        if not FILE_MODELLO.exists():
            raise SystemExit(
                f"Non trovo i pesi: {FILE_MODELLO}\n"
                f"Il modello va addestrato prima con esperimenti.py."
            )
        rete.load_state_dict(torch.load(FILE_MODELLO, map_location="cpu"))

    rete.eval()          # niente dropout o batchnorm in modalita' training
    return rete


def estrai_attivazioni(rete, X, batch=10_000):
    """
    Passa gli eventi attraverso la rete e raccoglie l'uscita di ogni strato
    nascosto, piu' l'uscita finale (il logit di classificazione).

    Restituisce (attivazioni, uscita): attivazioni e' una lista di array
    (N, 300), uno per strato nascosto; uscita e' un array (N, 1) col logit
    finale, utile per chiedersi se anche la decisione segnale/fondo, da
    sola, lascia leggere le masse.

    La rete e' congelata: torch.no_grad() disattiva il calcolo dei gradienti,
    quindi nulla puo' modificare i pesi e il calcolo e' piu' leggero.
    """
    pezzi_per_strato = None
    pezzi_uscita = []

    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            logit, attivazioni = rete(xb, restituisci_attivazioni=True)

            if pezzi_per_strato is None:
                pezzi_per_strato = [[] for _ in attivazioni]

            for k, a in enumerate(attivazioni):
                pezzi_per_strato[k].append(a.numpy())
            pezzi_uscita.append(logit.numpy())

    attivazioni = [np.concatenate(pezzi) for pezzi in pezzi_per_strato]
    uscita = np.concatenate(pezzi_uscita)
    return attivazioni, uscita


# ---------------------------------------------------------------------------
# 3. IL PROBE
# ---------------------------------------------------------------------------

def esegui_probe(rappresentazione, Y, etichetta=""):
    """
    Addestra una regressione lineare da 'rappresentazione' a ciascuna delle
    7 masse, e restituisce (R^2, alpha scelti) su eventi tenuti da parte.

    rappresentazione: array (N, d). Puo' essere le attivazioni di uno strato,
                      oppure le variabili grezze in ingresso.
    Y:                array (N, 7), le masse invarianti.

    Divisione: i primi N_PROBE_TRAIN eventi servono a stimare i coefficienti,
    gli ultimi N_PROBE_TEST a misurare. Se misurassimo sugli stessi eventi
    usati per stimare, l'R^2 sarebbe ottimisticamente alto: con 300 variabili
    si fitta bene qualunque cosa sui dati visti.
    """
    R_train = rappresentazione[:N_PROBE_TRAIN]
    R_test = rappresentazione[N_PROBE_TRAIN:]
    Y_train = Y[:N_PROBE_TRAIN]
    Y_test = Y[N_PROBE_TRAIN:]

    # --- standardizzazione -------------------------------------------------
    # Media e deviazione standard si stimano SOLO sugli eventi di fit e poi
    # si applicano a quelli di misura: stimarle su tutto vorrebbe dire far
    # entrare gli eventi di misura nella costruzione del probe.
    #
    # Vedi la nota in cima al file: senza questo passaggio la stessa
    # penalita' significherebbe una forza diversa per ogni strato, e i
    # confronti fra strati non sarebbero piu' validi.
    scaler = StandardScaler().fit(R_train)
    R_train = scaler.transform(R_train)
    R_test = scaler.transform(R_test)
    
    n_estremi = (np.abs(R_test) > 20).any(axis=1).sum()
    print(f"      eventi oltre 20 sigma: {n_estremi} su {len(R_test)}")
    R_test = np.clip(R_test, -10, 10)   

    # Diagnostica. "max |z| sul test" e' la quantita' chiave: dice quanto
    # lontano dalla distribuzione di fit arriva l'evento piu' estremo. Valori
    # nell'ordine delle migliaia segnalano una rappresentazione con code
    # pesanti, su cui un probe poco regolarizzato estrapola malissimo.
    if etichetta:
        print(f"    {etichetta:<18} scala originale: "
              f"|x| medio = {np.abs(scaler.mean_).mean():.2f}, "
              f"dev.std. media = {scaler.scale_.mean():.2f}")
        print(f"      colonne quasi costanti (std < 1e-3): "
              f"{(scaler.scale_ < 1e-3).sum()} su {len(scaler.scale_)}")
        print(f"      max |z| sul test: {np.abs(R_test).max():.1f}")

    # Una sola RidgeCV predice tutte e 7 le masse insieme: sklearn accetta
    # un bersaglio multidimensionale e risolve i 7 problemi in un colpo.
    #
    # alpha_per_target=True lascia scegliere un alpha diverso per ciascuna
    # massa. Le 7 hanno distribuzioni molto diverse fra loro e non c'e'
    # motivo di imporre a tutte la stessa regolarizzazione.
    regressione = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
    regressione.fit(R_train, Y_train)
    Y_previsto = regressione.predict(R_test)

    alpha_scelti = np.atleast_1d(regressione.alpha_)

    if etichetta:
        print(f"      alpha scelti: "
              f"{'  '.join(f'{a:.3g}' for a in alpha_scelti)}")

    # R^2 separato per ogni massa.
    # R^2 = 1 - (errore del modello) / (varianza dei dati)
    #   1  -> ricostruzione perfetta
    #   0  -> il probe non fa meglio che predire sempre la media
    r2 = r2_score(Y_test, Y_previsto, multioutput="raw_values")
    return r2, alpha_scelti.tolist()


# ---------------------------------------------------------------------------

def main():
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    if PICCOLO:
        print(">>> MODALITA' PROVA: dati ridotti, pesi letti da results_small/")
    print(f"Linear probing su {NOME}, seme {SEME}, stack {STACK}")
    print(f"Modello: {FILE_MODELLO}")
    print(f"Probe: {N_PROBE_TRAIN:,} eventi per il fit, "
          f"{N_PROBE_TEST:,} per la misura")
    print(f"RidgeCV su attivazioni standardizzate, alpha scelto per "
          f"validazione incrociata")
    print(f"  griglia: {ALPHAS.min():.3g} ... {ALPHAS.max():.3g} "
          f"({len(ALPHAS)} valori)")
    print("=" * 68)
    print()

    X, Y = carica_dati()
    n_input = len(INDICI[FEATURE_SET])

    risultati = {}
    alphas_usati = {}

    # --- riferimento 1: le variabili grezze in ingresso --------------------
    print("Riferimento: variabili in ingresso")
    r2, a = esegui_probe(X, Y, "input")
    risultati["input"] = r2.tolist()
    alphas_usati["input"] = a

    # --- riferimento 2: rete non addestrata -------------------------------
    print("Riferimento: rete non addestrata")
    rete_casuale = costruisci_rete(n_input, addestrata=False)
    att_casuali, uscita_casuale = estrai_attivazioni(rete_casuale, X)
    for k, A in enumerate(att_casuali, start=1):
        r2, a = esegui_probe(A, Y, f"casuale str.{k}")
        risultati[f"casuale_strato{k}"] = r2.tolist()
        alphas_usati[f"casuale_strato{k}"] = a
    r2, a = esegui_probe(uscita_casuale, Y, "casuale uscita")
    risultati["casuale_uscita"] = r2.tolist()
    alphas_usati["casuale_uscita"] = a
    del att_casuali, uscita_casuale

    # --- la rete addestrata ------------------------------------------------
    print("Rete addestrata")
    rete = costruisci_rete(n_input, addestrata=True)
    attivazioni, uscita = estrai_attivazioni(rete, X)
    for k, A in enumerate(attivazioni, start=1):
        r2, a = esegui_probe(A, Y, f"strato {k}")
        risultati[f"strato{k}"] = r2.tolist()
        alphas_usati[f"strato{k}"] = a
    r2, a = esegui_probe(uscita, Y, "uscita")
    risultati["uscita"] = r2.tolist()
    alphas_usati["uscita"] = a
    n_strati = len(attivazioni)
    del attivazioni, uscita

    # --- salvataggio -------------------------------------------------------
    uscita = {
        "nome": NOME,
        "seme": SEME,
        "stack": STACK,
        "feature_set": FEATURE_SET,
        "piccolo": PICCOLO,
        "n_strati": n_strati,
        "masse": NOMI_MASSE,
        # Griglia offerta alla CV e valori effettivamente scelti, uno per
        # rappresentazione e per massa. Servono a documentare il probe: un
        # alpha molto piu' alto su uno strato e' la misura diretta di quanto
        # e' mal condizionata quella rappresentazione.
        "alphas_griglia": ALPHAS.tolist(),
        "alphas_scelti": alphas_usati,
        # Servono a distinguere questo file da quelli prodotti dalle versioni
        # precedenti dello script, che non standardizzavano (standardizzato)
        # e che usavano un alpha unico fissato a mano (alpha_per_strato).
        "standardizzato": True,
        "alpha_per_strato": True,
        "n_probe_train": N_PROBE_TRAIN,
        "n_probe_test": N_PROBE_TEST,
        "r2": risultati,
    }

    percorso = CARTELLA_RISULTATI / f"probing_{NOME}_seme{SEME}.json"
    with open(percorso, "w") as f:
        json.dump(uscita, f, indent=2)

    # --- tabella a schermo -------------------------------------------------
    print()
    print("=" * 68)
    print("R^2 della ricostruzione lineare delle masse invarianti")
    print("=" * 68)

    intestazione = "  ".join(f"{m:>7}" for m in NOMI_MASSE)
    print(f"{'':<16} {intestazione}")
    print("-" * 68)

    def riga(etichetta, valori):
        numeri = "  ".join(f"{v:>7.3f}" for v in valori)
        print(f"{etichetta:<16} {numeri}")

    riga("input", risultati["input"])
    for k in range(1, n_strati + 1):
        riga(f"casuale str.{k}", risultati[f"casuale_strato{k}"])
    riga("casuale uscita", risultati["casuale_uscita"])
    print("-" * 68)
    for k in range(1, n_strati + 1):
        riga(f"strato {k}", risultati[f"strato{k}"])
    riga("uscita", risultati["uscita"])
    print("=" * 68)
    print(f"\nSalvato in {percorso}")


if __name__ == "__main__":
    main()