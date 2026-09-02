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

USO
---
    python src/probing.py deep low moderno 0
    python src/probing.py deep low 0
    python src/probing.py deep low moderno small 0     (prova rapida)

Con "small" si leggono i dati da data/processed_small e si cercano i pesi
in results_small/, esattamente come fa esperimenti.py: le due modalita' non
si mescolano mai.

Produce results/probing_<nome>_seme<k>.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

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

# Forza della regolarizzazione della ridge regression.
# Con 300 variabili molto correlate fra loro una regressione lineare
# semplice e' instabile: la ridge penalizza i coefficienti grandi e
# rende la soluzione ben definita.
ALPHA = 10

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
    nascosto.

    Restituisce una lista: un array (N, 300) per ogni strato.

    La rete e' congelata: torch.no_grad() disattiva il calcolo dei gradienti,
    quindi nulla puo' modificare i pesi e il calcolo e' piu' leggero.
    """
    pezzi_per_strato = None

    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            _, attivazioni = rete(xb, restituisci_attivazioni=True)

            if pezzi_per_strato is None:
                pezzi_per_strato = [[] for _ in attivazioni]

            for k, a in enumerate(attivazioni):
                pezzi_per_strato[k].append(a.numpy())

    return [np.concatenate(pezzi) for pezzi in pezzi_per_strato]


# ---------------------------------------------------------------------------
# 3. IL PROBE
# ---------------------------------------------------------------------------

def esegui_probe(rappresentazione, Y):
    """
    Addestra una regressione lineare da 'rappresentazione' a ciascuna delle
    7 masse, e restituisce l'R^2 su eventi tenuti da parte.

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

    # Una sola Ridge predice tutte e 7 le masse insieme: sklearn accetta
    # un bersaglio multidimensionale e risolve i 7 problemi in un colpo.
    regressione = Ridge(alpha=ALPHA)
    regressione.fit(R_train, Y_train)
    Y_previsto = regressione.predict(R_test)

    # R^2 separato per ogni massa.
    # R^2 = 1 - (errore del modello) / (varianza dei dati)
    #   1  -> ricostruzione perfetta
    #   0  -> il probe non fa meglio che predire sempre la media
    return r2_score(Y_test, Y_previsto, multioutput="raw_values")


# ---------------------------------------------------------------------------

def main():
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    if PICCOLO:
        print(">>> MODALITA' PROVA: dati ridotti, pesi letti da results_small/")
    print(f"Linear probing su {NOME}, seme {SEME}")
    print(f"Modello: {FILE_MODELLO.name}")
    print(f"Probe: {N_PROBE_TRAIN:,} eventi per il fit, "
          f"{N_PROBE_TEST:,} per la misura")
    print("=" * 68)
    print()

    X, Y = carica_dati()
    n_input = len(INDICI[FEATURE_SET])

    risultati = {}

    # --- riferimento 1: le variabili grezze in ingresso --------------------
    print("Riferimento: variabili in ingresso")
    risultati["input"] = esegui_probe(X, Y).tolist()

    # --- riferimento 2: rete non addestrata -------------------------------
    print("Riferimento: rete non addestrata")
    rete_casuale = costruisci_rete(n_input, addestrata=False)
    att_casuali = estrai_attivazioni(rete_casuale, X)
    for k, A in enumerate(att_casuali, start=1):
        risultati[f"casuale_strato{k}"] = esegui_probe(A, Y).tolist()
    del att_casuali

    # --- la rete addestrata ------------------------------------------------
    print("Rete addestrata")
    rete = costruisci_rete(n_input, addestrata=True)
    attivazioni = estrai_attivazioni(rete, X)
    for k, A in enumerate(attivazioni, start=1):
        risultati[f"strato{k}"] = esegui_probe(A, Y).tolist()
    n_strati = len(attivazioni)
    del attivazioni

    # --- salvataggio -------------------------------------------------------
    uscita = {
        "nome": NOME,
        "seme": SEME,
        "stack": STACK,
        "feature_set": FEATURE_SET,
        "piccolo": PICCOLO,
        "n_strati": n_strati,
        "masse": NOMI_MASSE,
        "alpha": ALPHA,
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
    print("-" * 68)
    for k in range(1, n_strati + 1):
        riga(f"strato {k}", risultati[f"strato{k}"])
    print("=" * 68)
    print(f"\nSalvato in {percorso.name}")


if __name__ == "__main__":
    main()