"""
Ottimizzazione degli iperparametri dello stack moderno, con TPE (hyperopt).

COSA CERCA
----------
Due soli iperparametri, entrambi di AdamW:
    lr             il learning rate
    weight_decay   la regolarizzazione

L'architettura NON viene ottimizzata: resta ai valori del paper. Due motivi.
Il primo e' che il confronto fra stack ha senso solo a parita' di modello:
se lo stack moderno avesse anche una rete diversa, non si saprebbe piu' a
quale delle due modifiche attribuire la differenza. Il secondo e' che la
dimensione ottima di una rete dipende da quanti dati ha, quindi quella
trovata su un sottoinsieme non varrebbe sui 2.6 milioni di eventi veri.

Learning rate e weight decay invece si trasferiscono ragionevolmente bene
fra scale diverse, ed e' per questo che ha senso cercarli su un
sottoinsieme.

SU QUANTI DATI
--------------
Non su processed_small (100.000 eventi), ma su un sottoinsieme del dataset
vero. Con 100.000 eventi il rapporto fra parametri e dati e' 4:1 e ogni
configurazione, senza eccezioni, va in overfitting: la ricerca finisce per
scegliere in un regime che non somiglia a quello dei training finali. Con
un milione il rapporto e' 1:3.5 e il problema si attenua molto.

Il batch e' quello del paper (100). E' importante: il learning rate ottimo
dipende fortemente dal batch, e cercarlo con un batch diverso da quello
dei training veri produrrebbe un valore non trasferibile.

COSA MINIMIZZA
--------------
1 - AUC di validation, presa all'epoca che l'early stopping selezionerebbe
davvero (quella a perdita di validation minima). Non il massimo dell'AUC
lungo tutto il training: quello sarebbe un valore che la pipeline vera non
restituirebbe mai, e ottimizzarlo significherebbe tarare gli iperparametri
su un modello diverso da quello che poi si usa.

Il test set non viene mai toccato.

PERCHE' UN FILE A PARTE
-----------------------
esperimenti.py ripete una configurazione fissa su piu' semi; qui invece la
configurazione cambia ad ogni giro, e la successiva dipende da com'e' andata
la precedente. Sono due cicli con logiche opposte, meglio tenerli separati.

USO
---
    pip install hyperopt          (se non e' gia' installato)

    python src/ottimizza.py              # deep low, 25 tentativi
    python src/ottimizza.py high         # deep high
    python src/ottimizza.py 30           # 30 tentativi

Produce results_small/ottimizzazione_<feature_set>_<n>k_batch<b>.json
Il nome contiene la configurazione, cosi' ricerche fatte con dati o batch
diversi restano tutte su disco e sono confrontabili fra loro.
"""

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from data import prepara_dati
from features import INDICI
from models import rete_profonda
from train import addestra


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

FEATURE_SET = "low"
N_TENTATIVI = 25

# Lo spazio di ricerca.
#
# hp.loguniform campiona in scala logaritmica: significa che 1e-4 e 1e-3
# hanno la stessa probabilita' di 1e-3 e 1e-2. E' la scala giusta per
# questi due parametri, perche' quello che conta e' l'ordine di grandezza,
# non la differenza assoluta: passare da 0.001 a 0.002 cambia molto,
# passare da 0.051 a 0.052 non cambia nulla.
#
# Vuole gli estremi gia' come logaritmi, da qui i np.log().
SPAZIO = {
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
    # L'intervallo del weight decay arriva molto in basso di proposito:
    # se la ricerca converge verso 1e-6 la risposta e' "la regolarizzazione
    # non serve", che e' essa stessa un risultato da riportare.
    "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),
}

# Un solo seme, uguale per tutti i tentativi: cosi' la differenza fra due
# configurazioni viene dagli iperparametri e non dall'inizializzazione.
SEME = 0

# Parametri della ricerca. Vedi la nota "SU QUANTI DATI" in cima al file.
N_TRAIN = 1_000_000
N_VAL = 200_000
N_TEST = 100_000        # non usato: la ricerca non tocca mai il test set
BATCH = 100             # lo stesso dei training finali, deve esserlo
MAX_EPOCHE = 60         # piu' dati significa piu' epoche per convergere
PAZIENZA = 8

# I primi tentativi TPE li fa a caso, per farsi un'idea dello spazio prima
# di iniziare a sfruttare quello che ha imparato. Con meno di una decina di
# punti il suo modello probabilistico non ha abbastanza informazione.
TENTATIVI_CASUALI = 10

# Se True, ogni tentativo scrive anche i log per TensorBoard.
TENSORBOARD = True

# ---------------------------------------------------------------------------


PROGETTO = Path(__file__).resolve().parent.parent
CARTELLA_DATI = PROGETTO / "data" / "processed"
CARTELLA_RISULTATI = PROGETTO / "results_small"
CARTELLA_LOG = PROGETTO / "runs_small" / "ottimizzazione"

FEATURE_SET_VALIDI = ["low", "high", "complete"]

for argomento in sys.argv[1:]:
    if argomento in FEATURE_SET_VALIDI:
        FEATURE_SET = argomento
    elif argomento.isdigit():
        N_TENTATIVI = int(argomento)
    else:
        raise SystemExit(
            f"Argomento non riconosciuto: '{argomento}'\n"
            f"Attesi: {FEATURE_SET_VALIDI} oppure un numero (i tentativi)."
        )


# I dati si caricano UNA VOLTA SOLA, fuori dalla funzione obiettivo:
# ricaricarli ad ogni tentativo sarebbe la parte piu' lenta di tutte.
print("Caricamento dati...")
DATI = prepara_dati(feature_set=FEATURE_SET,
                    cartella=CARTELLA_DATI,
                    n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST)
N_INPUT = len(INDICI[FEATURE_SET])
print()

# Contatore dei tentativi, serve solo per stampare e per i nomi dei log.
contatore = {"n": 0}


def obiettivo(parametri):
    """
    Addestra una rete con gli iperparametri proposti e restituisce il
    valore da minimizzare.

    hyperopt chiama questa funzione una volta per tentativo, passandole un
    dizionario campionato dallo spazio di ricerca. Il valore restituito
    guida la scelta del tentativo successivo.

    Restituiamo un dizionario invece del solo numero: la chiave "loss" e'
    quella che hyperopt minimizza, "status" gli dice che il tentativo e'
    andato a buon fine, e tutto il resto viene conservato dentro l'oggetto
    Trials. Sono informazioni che poi servono per i grafici, e che
    altrimenti andrebbero perse.
    """
    contatore["n"] += 1
    i = contatore["n"]

    lr = parametri["lr"]
    wd = parametri["weight_decay"]

    print(f"[{i:3d}/{N_TENTATIVI}] lr = {lr:.2e}, weight_decay = {wd:.2e}", flush=True)

    t0 = time.time()

    modello = rete_profonda(n_input=N_INPUT, attivazione="relu")

    logdir = None
    if TENSORBOARD:
        logdir = str(CARTELLA_LOG / f"{FEATURE_SET}_tentativo{i:03d}")

    storia = addestra(
        modello, DATI,
        batch=BATCH,
        lr_iniziale=lr,
        weight_decay=wd,
        max_epoche=MAX_EPOCHE,
        pazienza=PAZIENZA,
        ottimizzatore="adamw",
        seme=SEME,
        logdir=logdir,
        silenzioso=True,
    )

    # --- l'epoca che l'early stopping sceglierebbe -------------------------
    # I pesi finali sono quelli dell'epoca a perdita di validation minima,
    # quindi e' l'AUC di QUELLA epoca a rappresentare il modello che si
    # otterrebbe davvero.
    epoca_scelta = int(np.argmin(storia["perdita_val"]))
    auc = storia["auc_val"][epoca_scelta]

    minuti = (time.time() - t0) / 60

    print(f"          AUC = {auc:.4f}  "
          f"(epoca {epoca_scelta + 1} di {len(storia['auc_val'])}, "
          f"{minuti:.1f} min)", flush=True)

    return {
        "loss": 1.0 - auc,          # hyperopt MINIMIZZA questa quantita'
        "status": STATUS_OK,
        # Da qui in poi e' roba nostra, conservata per i grafici.
        "auc": float(auc),
        "lr": float(lr),
        "weight_decay": float(wd),
        "epoca_scelta": epoca_scelta + 1,
        "n_epoche": len(storia["auc_val"]),
        "minuti": round(minuti, 2),
    }


def main():
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"Ottimizzazione iperparametri - deep {FEATURE_SET}, stack moderno")
    print(f"Tentativi        : {N_TENTATIVI} (di cui {TENTATIVI_CASUALI} casuali)")
    print(f"Eventi train     : {N_TRAIN:,}  (batch {BATCH})")
    print(f"Epoche massime   : {MAX_EPOCHE}")
    print(f"Seme fisso       : {SEME}")
    print("=" * 64)
    print()

    # Trials conserva lo storico completo della ricerca: ogni tentativo con
    # i suoi parametri e il suo risultato. E' da qui che si ricavano i
    # grafici della ricerca, non dal solo valore migliore.
    trials = Trials()

    algoritmo = partial(tpe.suggest, n_startup_jobs=TENTATIVI_CASUALI)

    t0 = time.time()

    migliori = fmin(
        fn=obiettivo,
        space=SPAZIO,
        algo=algoritmo,
        max_evals=N_TENTATIVI,
        trials=trials,
        # Seme del generatore casuale di hyperopt: rende ripetibile anche
        # la sequenza dei tentativi, non solo i singoli addestramenti.
        rstate=np.random.default_rng(0),
        show_progressbar=False,
    )

    minuti_totali = (time.time() - t0) / 60

    # --- raccolta dei risultati -------------------------------------------
    # trials.results e' la lista dei dizionari restituiti da obiettivo(),
    # nell'ordine in cui sono stati provati.
    tentativi = []
    for k, risultato in enumerate(trials.results, start=1):
        tentativi.append({
            "tentativo": k,
            "lr": risultato["lr"],
            "weight_decay": risultato["weight_decay"],
            "auc": risultato["auc"],
            "epoca_scelta": risultato["epoca_scelta"],
            "n_epoche": risultato["n_epoche"],
            "minuti": risultato["minuti"],
        })

    indice_migliore = int(np.argmin([t["loss"] for t in trials.results]))
    migliore = tentativi[indice_migliore]

    uscita = {
        "feature_set": FEATURE_SET,
        "modello": "deep",
        "stack": "moderno",
        "n_tentativi": N_TENTATIVI,
        "n_tentativi_casuali": TENTATIVI_CASUALI,
        "seme": SEME,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "batch": BATCH,
        "max_epoche": MAX_EPOCHE,
        "minuti_totali": round(minuti_totali, 1),
        "spazio": {
            "lr": [1e-4, 1e-2],
            "weight_decay": [1e-6, 1e-2],
        },
        "migliore": migliore,
        "tentativi": tentativi,
    }

    # Il nome contiene la configurazione: cosi' ricerche fatte con dati o
    # batch diversi non si sovrascrivono mai, e restano confrontabili.
    etichetta = f"{N_TRAIN // 1000}k_batch{BATCH}"
    percorso = CARTELLA_RISULTATI / f"ottimizzazione_{FEATURE_SET}_{etichetta}.json"

    with open(percorso, "w") as f:
        json.dump(uscita, f, indent=2)

    # --- riepilogo a schermo ----------------------------------------------
    print()
    print("=" * 64)
    print("MIGLIORE CONFIGURAZIONE")
    print("=" * 64)
    print(f"  learning rate : {migliore['lr']:.3e}")
    print(f"  weight decay  : {migliore['weight_decay']:.3e}")
    print(f"  AUC           : {migliore['auc']:.4f}")
    print(f"  tentativo n.  : {migliore['tentativo']} di {N_TENTATIVI}")
    print()
    print(f"  (default attuali: lr 1e-3, weight_decay 1e-4)")
    print()

    # I cinque migliori: se sono tutti vicini fra loro, il risultato e'
    # robusto; se sono sparsi, la superficie e' piatta e la scelta conta poco.
    ordinati = sorted(tentativi, key=lambda t: -t["auc"])[:5]
    print("  I cinque migliori:")
    print(f"  {'lr':>10}  {'wd':>10}  {'AUC':>7}  {'epoche':>7}")
    for t in ordinati:
        print(f"  {t['lr']:10.2e}  {t['weight_decay']:10.2e}  "
              f"{t['auc']:7.4f}  {t['n_epoche']:7d}")

    print()
    print(f"Tempo totale: {minuti_totali:.1f} min")
    print(f"Salvato in {percorso.name}")
    print("=" * 64)
    print()
    print(f"NOTA: valori trovati su {N_TRAIN:,} eventi; i training finali ne usano")
    print("10M. Il learning rate ottimo tende a scendere al crescere dei dati.")


if __name__ == "__main__":
    main()