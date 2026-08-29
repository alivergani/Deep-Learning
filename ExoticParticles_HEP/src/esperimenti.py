"""
Lancia UNA configurazione (modello + feature set + stack) su piu' semi.

Si sceglie la configurazione modificando le variabili qui sotto, oppure
passandola da riga di comando:

    python src/esperimenti.py deep low
    python src/esperimenti.py deep low moderno 0
    python src/esperimenti.py deep low moderno small 0     (prova rapida)

Ogni seme gia' completato viene saltato, quindi si puo' rilanciare lo
script dopo un'interruzione senza perdere il lavoro fatto.

I due stack scrivono in cartelle diverse, quindi non si sovrascrivono:

    results/riproduzione/deep_low_seme0_modello.pt    stack 2014
    results/moderno/deep_low_seme0_modello.pt         stack moderno

Con "small" cambiano INSIEME la cartella dei dati e quella dei risultati:
i risultati di prova finiscono in results_small/ e non possono mescolarsi
con quelli veri.

Prodotti nella cartella scelta:
    <nome>_seme<k>_modello.pt      pesi della rete
    <nome>_seme<k>_storia.json     perdita e AUC epoca per epoca
    <nome>_seme<k>_roc.npz         curva ROC sul test set
    <nome>_riepilogo.json          AUC finali di tutti i semi

I log per TensorBoard vanno invece in runs/<stack>/<nome>_seme<k>/.
Per guardarli:  tensorboard --logdir=runs
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data import prepara_dati
from evaluate import calcola_curva, salva_curva
from features import INDICI
from models import rete_profonda, rete_shallow
from train import addestra, valuta


# ---------------------------------------------------------------------------
# CONFIGURAZIONE - le uniche righe da modificare
# ---------------------------------------------------------------------------

MODELLO = "deep"             # "deep" oppure "shallow"
FEATURE_SET = "low"          # "low", "high" oppure "complete"

# Lo stack di addestramento:
#   "2014"    -> tanh + inizializzazione del paper + SGD con rampa di momentum
#   "moderno" -> ReLU + inizializzazione di He + AdamW con riduzione a plateau
# Attivazione e ottimizzatore vanno insieme: si sceglie lo stack, non i
# singoli pezzi.
STACK = "2014"

SEMI = [0, 1, 2, 3, 4]       # 5 inizializzazioni casuali, come nel paper
# Per lo stack moderno basta un seme solo, purche' sia uno di questi:
# cosi' i due stack vedono gli stessi dati nello stesso ordine.

# Dataset ridotto per le prove. Si attiva SOLO da riga di comando con
# l'argomento "small": non e' una variabile da cambiare a mano, perche'
# dimenticarsela accesa significherebbe lanciare un training vero sui dati
# di prova (o viceversa) senza accorgersene.
PICCOLO = False

# Parametri dei dati (dataset completo)
N_TRAIN = 2_600_000
N_VAL = 500_000
N_TEST = 500_000

# Parametri di addestramento (dataset completo)
BATCH = 100
EPOCHE_RAMPA = 200           # usato solo dallo stack 2014
MAX_EPOCHE = 300
PAZIENZA = 10

N_THREAD = 2                 # 0 = lascia decidere a PyTorch

# Se True, scrive i log per TensorBoard in runs/.
TENSORBOARD = True

# ---------------------------------------------------------------------------


PROGETTO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Argomenti da riga di comando (tutti facoltativi).
#
# Si riconoscono dal valore, quindi l'ordine non conta:
#     "deep" / "shallow"            -> il modello
#     "low" / "high" / "complete"   -> il feature set
#     "2014" / "moderno"            -> lo stack
#     "small"                       -> dataset ridotto, risultati separati
#     numeri                        -> i semi da addestrare
#
# Esempi:
#     python src/esperimenti.py                      usa i valori scritti sopra
#     python src/esperimenti.py deep low             tutti i semi della lista
#     python src/esperimenti.py deep low 0 1         solo i semi 0 e 1
#     python src/esperimenti.py deep low moderno 0   stack moderno, seme 0
#     python src/esperimenti.py deep low small 0     prova rapida
#
# Per parallelizzare, due sessioni tmux:
#     python src/esperimenti.py deep low 0 1
#     python src/esperimenti.py deep low 2 3 4
#
# Attenzione: "2014" e' anche un numero, ma viene letto come stack.
# Se ti servisse davvero il seme 2014, cambia la lista SEMI qui sopra.
# --------------------------------------------------------------------------

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
        raise SystemExit(
            f"Argomento non riconosciuto: '{argomento}'\n"
            f"Attesi: {MODELLI_VALIDI}, {FEATURE_SET_VALIDI}, "
            f"{STACK_VALIDI}, 'small', oppure numeri (i semi)."
        )

if semi_da_riga_comando:
    SEMI = semi_da_riga_comando


# Le due impostazioni che dipendono dallo stack.
if STACK == "2014":
    ATTIVAZIONE = "tanh"
    OTTIMIZZATORE = "sgd"
elif STACK == "moderno":
    ATTIVAZIONE = "relu"
    OTTIMIZZATORE = "adamw"
else:
    raise SystemExit(f"STACK deve essere uno di {STACK_VALIDI}")

# --- dove leggere i dati e dove scrivere i risultati ----------------------
# I percorsi cambiano insieme: e' l'unico modo per essere sicuri che i
# risultati di prova non finiscano mai fra quelli veri.
# La sottocartella dice gia' lo stack, quindi il nome del file non lo ripete.
SOTTOCARTELLA = "riproduzione" if STACK == "2014" else "moderno"

if PICCOLO:
    CARTELLA_DATI = PROGETTO / "data" / "processed_small"
    CARTELLA_RISULTATI = PROGETTO / "results_small" / SOTTOCARTELLA
    N_TRAIN = 70_000
    N_VAL = 15_000
    N_TEST = 15_000
    BATCH = 1000
    MAX_EPOCHE = 20
    EPOCHE_RAMPA = 5
else:
    CARTELLA_DATI = None     # None = data/processed, il default di data.py
    CARTELLA_RISULTATI = PROGETTO / "results" / SOTTOCARTELLA

# I log di TensorBoard stanno fuori da results/: sono file di monitoraggio,
# non risultati da conservare o sincronizzare. Le prove su dati ridotti
# vanno in runs_small/, per non sporcare i log dei run veri.
if TENSORBOARD:
    radice_log = "runs_small" if PICCOLO else "runs"
    CARTELLA_LOG = PROGETTO / radice_log / SOTTOCARTELLA
else:
    CARTELLA_LOG = None

NOME = f"{MODELLO}_{FEATURE_SET}"


def costruisci_modello(n_input):
    """Crea la rete richiesta dalla configurazione."""
    if MODELLO == "deep":
        return rete_profonda(n_input=n_input, attivazione=ATTIVAZIONE)
    if MODELLO == "shallow":
        return rete_shallow(n_input=n_input, attivazione=ATTIVAZIONE)
    raise ValueError("MODELLO deve essere 'deep' oppure 'shallow'")


def main():
    if N_THREAD > 0:
        torch.set_num_threads(N_THREAD)

    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    if PICCOLO:
        print(">>> MODALITA' PROVA: dati ridotti")
    print(f"Configurazione : {NOME}")
    print(f"Stack          : {STACK}  ({ATTIVAZIONE} + {OTTIMIZZATORE})")
    print(f"Semi           : {SEMI}")
    print(f"Eventi train   : {N_TRAIN:,}")
    print(f"Batch          : {BATCH}")
    print(f"Epoche massime : {MAX_EPOCHE}")
    print(f"Risultati in   : {CARTELLA_RISULTATI.relative_to(PROGETTO)}/")
    if CARTELLA_LOG is not None:
        print(f"Log TensorBoard: {CARTELLA_LOG.relative_to(PROGETTO)}/")
    print("=" * 60)
    print()

    # I dati si caricano UNA VOLTA SOLA e si riusano per tutti i semi:
    # cambia l'inizializzazione della rete, non i dati.
    argomenti = dict(feature_set=FEATURE_SET,
                     n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST)
    if CARTELLA_DATI is not None:
        argomenti["cartella"] = CARTELLA_DATI

    dati = prepara_dati(**argomenti)
    X_test, y_test = dati["test"]
    n_input = len(INDICI[FEATURE_SET])
    print()

    for seme in SEMI:
        base = CARTELLA_RISULTATI / f"{NOME}_seme{seme}"
        file_modello = Path(str(base) + "_modello.pt")

        # --- se questo seme e' gia' stato fatto, lo saltiamo ---------------
        if file_modello.exists():
            with open(str(base) + "_storia.json") as f:
                salvato = json.load(f)
            print(f"[seme {seme}] gia' completato, AUC = {salvato['auc_test']:.4f}")
            continue

        print(f"\n{'-' * 60}")
        print(f"[seme {seme}] inizio")
        print(f"{'-' * 60}")

        t0 = time.time()

        modello = costruisci_modello(n_input)

        # Una sottocartella di log per ogni seme: TensorBoard le mostra come
        # curve distinte, sovrapponibili nello stesso grafico.
        if CARTELLA_LOG is not None:
            logdir = str(CARTELLA_LOG / f"{NOME}_seme{seme}")
        else:
            logdir = None

        storia = addestra(
            modello, dati,
            batch=BATCH,
            epoche_rampa=EPOCHE_RAMPA,
            max_epoche=MAX_EPOCHE,
            pazienza=PAZIENZA,
            ottimizzatore=OTTIMIZZATORE,
            seme=seme,
            logdir=logdir,
        )

        # --- valutazione finale sul TEST set -------------------------------
        # Il test set si tocca solo qui, una volta per seme.
        perdita_test, auc_test = valuta(modello, X_test, y_test)
        print(f"\n[seme {seme}] TEST: perdita {perdita_test:.5f}, AUC {auc_test:.4f}")

        # --- salvataggi ----------------------------------------------------
        torch.save(modello.state_dict(), file_modello)

        storia["auc_test"] = float(auc_test)
        storia["perdita_test"] = float(perdita_test)
        storia["minuti"] = round((time.time() - t0) / 60, 1)
        storia["stack"] = STACK
        storia["n_epoche"] = len(storia["perdita_val"])
        storia["piccolo"] = PICCOLO
        with open(str(base) + "_storia.json", "w") as f:
            json.dump(storia, f)

        curva = calcola_curva(modello, X_test, y_test,
                              nome=f"{MODELLO} {FEATURE_SET} {STACK} (seme {seme})")
        salva_curva(curva, str(base) + "_roc.npz")

    # --- riepilogo della configurazione -------------------------------------
    # Rileggiamo TUTTI i semi presenti su disco, non solo quelli addestrati da
    # questo processo: cosi' il riepilogo resta corretto anche se piu' processi
    # girano in parallelo, ciascuno su un sottoinsieme di semi.
    #
    # I due stack non si confondono perche' stanno in cartelle diverse.
    semi_trovati = []
    valori_trovati = []
    epoche_trovate = []

    for file_storia in sorted(CARTELLA_RISULTATI.glob(f"{NOME}_seme*_storia.json")):
        numero = int(file_storia.name.split("_seme")[1].split("_")[0])
        with open(file_storia) as f:
            info = json.load(f)
        semi_trovati.append(numero)
        valori_trovati.append(float(info["auc_test"]))
        epoche_trovate.append(len(info["perdita_val"]))

    ordine = np.argsort(semi_trovati)
    semi_trovati = [semi_trovati[i] for i in ordine]
    valori = np.asarray([valori_trovati[i] for i in ordine])
    epoche = [epoche_trovate[i] for i in ordine]

    # Deviazione standard campionaria (ddof=1): stiamo stimando la
    # dispersione da un campione di semi, non descrivendo una popolazione.
    # Con un solo seme non e' definita, e mettiamo None invece di 0:
    # uno zero si scambierebbe per "nessuna variabilita'".
    if len(valori) > 1:
        dev_std = float(valori.std(ddof=1))
    else:
        dev_std = None

    riepilogo = {
        "modello": MODELLO,
        "feature_set": FEATURE_SET,
        "stack": STACK,
        "attivazione": ATTIVAZIONE,
        "ottimizzatore": OTTIMIZZATORE,
        "piccolo": PICCOLO,
        "n_input": n_input,
        "n_train": N_TRAIN,
        "batch": BATCH,
        "semi": semi_trovati,
        "auc_test": [float(v) for v in valori],
        "auc_medio": float(valori.mean()),
        "auc_dev_std": dev_std,
        "n_epoche": epoche,
    }

    with open(CARTELLA_RISULTATI / f"{NOME}_riepilogo.json", "w") as f:
        json.dump(riepilogo, f, indent=2)

    print()
    print("=" * 60)
    if dev_std is None:
        print(f"{NOME}: AUC = {valori.mean():.4f}  (un solo seme)")
    else:
        print(f"{NOME}: AUC = {valori.mean():.4f} ({dev_std:.4f})")
    print(f"semi inclusi  : {semi_trovati}")
    print(f"valori singoli: {np.round(valori, 4).tolist()}")
    print(f"epoche         : {epoche}")
    print("=" * 60)


if __name__ == "__main__":
    main()