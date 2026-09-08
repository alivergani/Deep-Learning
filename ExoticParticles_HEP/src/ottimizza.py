"""
Ottimizzazione degli iperparametri dello stack moderno, con TPE (hyperopt).

DUE MODALITA'
-------------
1) DEFAULT - cerca solo gli iperparametri di AdamW:
       lr             il learning rate
       weight_decay   la regolarizzazione
   L'architettura resta quella del paper (4 strati da 300 unita').
   E' la ricerca che serve al CONFRONTO FRA STACK: il confronto ha senso
   solo a parita' di modello, altrimenti non si saprebbe a quale delle due
   modifiche attribuire la differenza.

2) "arch" - cerca ANCHE l'architettura:
       n_strati       da 2 a 8 strati nascosti
       n_unita        da 100 a 1000 unita' per strato
   E' una domanda diversa dalla precedente: non "quanto cambia la pratica
   di ottimizzazione", ma "qual e' la configurazione migliore per questo
   problema". Va tenuta separata nella discussione dei risultati.

PERCHE' LA RICERCA E' CONGIUNTA
-------------------------------
Nella modalita' "arch" si cercano tutti e quattro i parametri insieme,
invece di fissare lr e weight_decay ai valori gia' trovati.

Il motivo e' che il learning rate ottimo dipende dall'architettura: una
rete a 2 strati e una a 8 non hanno lo stesso valore migliore. Fissarlo
significherebbe confrontare architetture diverse con un learning rate
ottimo per nessuna di esse, e il risultato direbbe piu' sul learning rate
che sull'architettura.

E' anche coerente con quanto visto a lezione: fra gli effetti degli
iperparametri sulla capacita' del modello, il learning rate e' l'unico
descritto come "aumenta la capacita' quando e' tarato in modo ottimale" —
cioe' il suo valore migliore dipende da tutto il resto.

SU QUANTI DATI
--------------
Tre milioni di eventi, un terzo dei dieci milioni dei training finali.

La prima ricerca era stata fatta su un milione, e il valore di weight decay
che ne era uscito (9.8e-3) si e' rivelato eccessivo sui training veri: le
curve di AUC dello stack moderno saturavano circa un punto sotto quelle
dello stack 2014 sui feature set low e complete, mentre su high — dove la
rete deve solo combinare sette masse gia' calcolate — coincidevano.
Il divario compariva cioe' solo dove serve capacita' effettiva, che e'
esattamente cio' che una regolarizzazione troppo forte toglie.

La ragione e' che la regolarizzazione ottima diminuisce al crescere dei
dati: con un milione di eventi il weight decay serve davvero, con dieci
non piu'. Passando da 1 a 3 milioni il rapporto con i training finali
scende da 10 a 3, e i valori trovati diventano trasferibili.

Resta un fattore 3 dichiarato. Se il weight decay scelto risultasse ancora
troppo alto, la direzione della correzione e' nota: verso il basso.

Il batch e' quello del paper (100). E' importante: il learning rate ottimo
dipende fortemente dal batch, e cercarlo con un batch diverso da quello
dei training veri produrrebbe un valore non trasferibile.

SUL BUDGET DI EPOCHE
--------------------
MAX_EPOCHE scende da 60 a 30, ma non e' un taglio: un'epoca su 3 milioni
di eventi contiene 30.000 aggiornamenti invece di 10.000, quindi 30 epoche
su 3M sono 900.000 passi contro i 600.000 delle 60 epoche su 1M. Ogni
tentativo vede piu' aggiornamenti di prima, non meno.

E' la quantita' che conta per il weight decay: la regolarizzazione agisce
lentamente, e con troppi pochi passi la differenza fra un valore alto e uno
basso non farebbe in tempo a manifestarsi. La ricerca sceglierebbe allora
un valore quasi a caso.

COSA MINIMIZZA
--------------
1 - AUC di validation, presa all'epoca che l'early stopping selezionerebbe
davvero (quella a perdita di validation minima). Non il massimo dell'AUC
lungo tutto il training: quello sarebbe un valore che la pipeline vera non
restituirebbe mai, e ottimizzarlo significherebbe tarare gli iperparametri
su un modello diverso da quello che poi si usa.

Il validation set e' usato per intero (500.000 eventi). L'incertezza
statistica sull'AUC scala come 1/sqrt(N): con 200.000 eventi era circa 1.6
volte piu' grande, abbastanza da rendere indistinguibili due tentativi
vicini. Il costo in tempo e' trascurabile, un forward pass contro decine di
migliaia di passi di training.

Il test set non viene mai toccato.

PERCHE' UN FILE A PARTE
-----------------------
esperimenti.py ripete una configurazione fissa su piu' semi; qui invece la
configurazione cambia ad ogni giro, e la successiva dipende da com'e' andata
la precedente. Sono due cicli con logiche opposte, meglio tenerli separati.

USO
---
    pip install hyperopt          (se non e' gia' installato)

    python src/ottimizza.py                 # deep low, solo lr e wd
    python src/ottimizza.py high            # deep high
    python src/ottimizza.py 30              # 30 tentativi
    python src/ottimizza.py arch 20         # cerca anche l'architettura

Produce results_small/ottimizzazione_<feature_set>_<n>k_batch<b>[_arch].json
Il nome contiene la configurazione, cosi' ricerche fatte con dati, batch o
spazi diversi restano tutte su disco e sono confrontabili fra loro: la
ricerca su 1M e quella su 3M convivono, ed e' il confronto fra le due a
mostrare quanto il weight decay ottimo dipenda dalla dimensione dei dati.

Il file viene riscritto dopo OGNI tentativo, con un campo "completo" che dice
se la ricerca e' arrivata in fondo. Una ricerca di 25 tentativi su 3 milioni
di eventi dura ore: se la sessione cade a meta', i tentativi gia' fatti sono
comunque su disco e leggibili.
"""

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from data import prepara_dati
from features import INDICI
from models import MLP, conta_parametri
from train import addestra, PAZIENZA_ADAMW


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

FEATURE_SET = "low"
N_TENTATIVI = 25

# Se True cerca anche n_strati e n_unita. Si attiva da riga di comando
# con l'argomento "arch".
CERCA_ARCHITETTURA = False

# Architettura del paper, usata quando NON si cerca l'architettura.
STRATI_PAPER = 4
UNITA_PAPER = 300

# Un solo seme, uguale per tutti i tentativi: cosi' la differenza fra due
# configurazioni viene dagli iperparametri e non dall'inizializzazione.
SEME = 0

# Parametri della ricerca. Vedi le note "SU QUANTI DATI" e "SUL BUDGET DI
# EPOCHE" in cima al file.
N_TRAIN = 3_000_000
N_VAL = 500_000
N_TEST = 100_000        # non usato: la ricerca non tocca mai il test set
BATCH = 100             # lo stesso dei training finali, deve esserlo
MAX_EPOCHE = 30

# La pazienza dell'early stopping e' presa da train.py invece di essere
# riscritta qui: la ricerca deve usare la stessa regola di arresto dei
# training finali, altrimenti sceglie gli iperparametri di una pipeline che
# poi non e' quella che si usa.
#
# Con 15 e un tetto di 30 epoche l'arresto anticipato quasi non interviene, e
# va bene cosi': una pazienza corta penalizzerebbe le configurazioni che
# migliorano lentamente, cioe' proprio quelle a weight decay basso, che sono
# quelle che ci aspettiamo vincano. Sarebbe un pregiudizio nella direzione
# sbagliata.
PAZIENZA = PAZIENZA_ADAMW

# I primi tentativi TPE li fa a caso, per farsi un'idea dello spazio prima
# di iniziare a sfruttare quello che ha imparato. Con meno di una decina di
# punti il suo modello probabilistico non ha abbastanza informazione.
TENTATIVI_CASUALI = 12

# Se True, ogni tentativo scrive anche i log per TensorBoard.
TENSORBOARD = True

# ---------------------------------------------------------------------------


PROGETTO = Path(__file__).resolve().parent.parent
CARTELLA_DATI = PROGETTO / "data" / "processed"
CARTELLA_RISULTATI = PROGETTO / "results_small"

FEATURE_SET_VALIDI = ["low", "high", "complete"]

for argomento in sys.argv[1:]:
    if argomento in FEATURE_SET_VALIDI:
        FEATURE_SET = argomento
    elif argomento == "arch":
        CERCA_ARCHITETTURA = True
    elif argomento.isdigit():
        N_TENTATIVI = int(argomento)
    else:
        raise SystemExit(
            f"Argomento non riconosciuto: '{argomento}'\n"
            f"Attesi: {FEATURE_SET_VALIDI}, 'arch', oppure un numero."
        )


# --- lo spazio di ricerca --------------------------------------------------
#
# hp.loguniform campiona in scala logaritmica: significa che 1e-4 e 1e-3
# hanno la stessa probabilita' di 1e-3 e 1e-2. E' la scala giusta per lr e
# weight decay, perche' quello che conta e' l'ordine di grandezza, non la
# differenza assoluta: passare da 0.001 a 0.002 cambia molto, passare da
# 0.051 a 0.052 non cambia nulla. Vuole gli estremi gia' come logaritmi,
# da qui i np.log().
#
# hp.quniform invece campiona uniformemente su una griglia: e' quello che
# serve per numeri interi come la profondita' e le unita' per strato.
# Restituisce float, quindi vanno convertiti con int().

LR_MIN, LR_MAX = 1e-4, 1e-2
WD_MIN, WD_MAX = 1e-6, 1e-2
STRATI_MIN, STRATI_MAX = 2, 8
UNITA_MIN, UNITA_MAX = 100, 1000

SPAZIO = {
    "lr": hp.loguniform("lr", np.log(LR_MIN), np.log(LR_MAX)),
    # L'intervallo del weight decay arriva molto in basso di proposito:
    # se la ricerca converge verso 1e-6 la risposta e' "la regolarizzazione
    # non serve", che e' essa stessa un risultato da riportare. Con 3
    # milioni di eventi e' anche l'esito piu' probabile, ed e' il motivo per
    # cui l'estremo inferiore non e' stato alzato.
    "weight_decay": hp.loguniform("weight_decay", np.log(WD_MIN), np.log(WD_MAX)),
}

if CERCA_ARCHITETTURA:
    # Profondita' fino a 8 strati: il paper si ferma a 6 (Supplementary
    # Table 3, dove riporta 0.888 con 6 strati contro 0.880 con 5). Con la
    # tanh andare oltre diventa difficile per la diffusione del gradiente;
    # con ReLU e He il problema non si pone, quindi vale la pena guardare
    # se il guadagno continua dove il paper si era fermato.
    SPAZIO["n_strati"] = hp.quniform("n_strati", STRATI_MIN, STRATI_MAX, 1)
    # Da 100 a 1000 unita', a passi di 100. Il paper esplorava 100-500.
    SPAZIO["n_unita"] = hp.quniform("n_unita", UNITA_MIN, UNITA_MAX, 100)


# --- nomi dei file di uscita ----------------------------------------------
# L'etichetta contiene la configurazione, e finisce sia nel nome del JSON sia
# in quello della cartella dei log. Senza, i log della ricerca su 3 milioni
# finirebbero nelle stesse cartelle di quella su 1 milione (stesso feature
# set, stessi numeri di tentativo) e TensorBoard sovrapporrebbe due ricerche
# diverse sullo stesso grafico.
ETICHETTA = f"{N_TRAIN // 1000}k_batch{BATCH}"
if CERCA_ARCHITETTURA:
    ETICHETTA += "_arch"

PERCORSO_USCITA = (CARTELLA_RISULTATI /
                   f"ottimizzazione_{FEATURE_SET}_{ETICHETTA}.json")

CARTELLA_LOG = PROGETTO / "runs_small" / f"ottimizzazione_{FEATURE_SET}_{ETICHETTA}"


# --- controllo del dispositivo ---------------------------------------------
# Meglio saperlo subito che dopo dieci minuti di caricamento dati: su CPU una
# ricerca come questa non finirebbe in tempi utili, quindi se la GPU non c'e'
# e' quasi sempre un ambiente sbagliato, non una scelta.
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
if DISPOSITIVO == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ATTENZIONE: nessuna GPU trovata, si userebbe la CPU.")
    print("Controlla di essere sulla macchina giusta e nel venv giusto.")
    try:
        risposta = input("Proseguire lo stesso? [s/N] ").strip().lower()
    except EOFError:
        # Lanciato di notte da uno script, senza nessuno a rispondere.
        risposta = "n"
    if risposta != "s":
        raise SystemExit("Interrotto: nessuna GPU disponibile.")
print()


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

# I tentativi gia' conclusi, nell'ordine in cui sono stati provati. Vengono
# riempiti da obiettivo() e riscritti su disco dopo ogni tentativo: se la
# sessione muore a meta' ricerca, quello che era gia' stato fatto resta.
TENTATIVI = []
T_INIZIO = time.time()


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

    # hp.quniform restituisce float anche per valori interi: vanno convertiti.
    if CERCA_ARCHITETTURA:
        n_strati = int(parametri["n_strati"])
        n_unita = int(parametri["n_unita"])
    else:
        n_strati = STRATI_PAPER
        n_unita = UNITA_PAPER

    # Il seme va fissato QUI, non dentro addestra(): i pesi iniziali si
    # estraggono nella riga sotto, cioe' prima che addestra() venga chiamata.
    # Senza questa riga ogni tentativo partirebbe da un'inizializzazione
    # diversa, perche' il generatore casuale globale avanza ad ogni tentativo,
    # e parte della differenza di AUC fra due configurazioni verrebbe dai pesi
    # iniziali invece che dagli iperparametri.
    torch.manual_seed(SEME)

    # Si costruisce MLP direttamente invece di rete_profonda, perche' qui
    # profondita' e larghezza devono poter cambiare.
    modello = MLP(n_input=N_INPUT, n_strati=n_strati,
                  n_unita=n_unita, attivazione="relu")
    n_parametri = conta_parametri(modello)

    descrizione = f"lr = {lr:.2e}, wd = {wd:.2e}"
    if CERCA_ARCHITETTURA:
        descrizione += f", {n_strati} strati x {n_unita} un. ({n_parametri:,} par.)"
    print(f"[{i:3d}/{N_TENTATIVI}] {descrizione}", flush=True)

    t0 = time.time()

    logdir = None
    if TENSORBOARD:
        logdir = str(CARTELLA_LOG / f"{FEATURE_SET}_tentativo{i:03d}")

    # --- il tentativo, protetto -------------------------------------------
    # Con learning rate fino a 1e-2 un tentativo puo' divergere. Se succede,
    # la perdita diventa NaN, l'AUC pure, e hyperopt riceve una "loss" non
    # finita: fmin si ferma con un errore e una ricerca di ore muore a meta'.
    #
    # Qui il tentativo fallito viene invece registrato come AUC 0.5, cioe' il
    # valore di una rete che tira a indovinare. E' la cosa giusta da fare
    # anche nel merito: TPE impara che quella zona dello spazio non funziona
    # e smette di andarci, che e' esattamente il comportamento voluto.
    try:
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

        # --- l'epoca che l'early stopping sceglierebbe ---------------------
        # I pesi finali sono quelli dell'epoca a perdita di validation
        # minima, quindi e' l'AUC di QUELLA epoca a rappresentare il modello
        # che si otterrebbe davvero.
        #
        # (Non coincide al millesimo con l'epoca salvata da addestra, che
        # richiede un miglioramento relativo di MIGLIORAMENTO_MINIMO per
        # aggiornare i pesi migliori. La differenza e' trascurabile, ma vale
        # la pena saperlo.)
        epoca_scelta = int(np.argmin(storia["perdita_val"]))
        auc = storia["auc_val"][epoca_scelta]

        if not np.isfinite(auc):
            raise ValueError("AUC non finita")

    except Exception as errore:
        minuti = (time.time() - t0) / 60
        print(f"          FALLITO ({errore}) - registrato come AUC 0.5, "
              f"{minuti:.1f} min", flush=True)

        fallito = {
            "tentativo": i,
            "lr": float(lr),
            "weight_decay": float(wd),
            "n_strati": n_strati,
            "n_unita": n_unita,
            "n_parametri": n_parametri,
            "auc": 0.5,
            "epoca_scelta": 0,
            "n_epoche": 0,
            "minuti": round(minuti, 2),
            "fallito": True,
        }
        TENTATIVI.append(fallito)
        scrivi_risultati(completo=False)

        return {"loss": 0.5, "status": STATUS_OK, **fallito}

    minuti = (time.time() - t0) / 60

    print(f"          AUC = {auc:.4f}  "
          f"(epoca {epoca_scelta + 1} di {len(storia['auc_val'])}, "
          f"{minuti:.1f} min)", flush=True)

    TENTATIVI.append({
        "tentativo": i,
        "lr": float(lr),
        "weight_decay": float(wd),
        "n_strati": n_strati,
        "n_unita": n_unita,
        "n_parametri": n_parametri,
        "auc": float(auc),
        "epoca_scelta": epoca_scelta + 1,
        "n_epoche": len(storia["auc_val"]),
        "minuti": round(minuti, 2),
    })

    # Su disco subito, non alla fine: una ricerca di 25 tentativi dura ore, e
    # un file scritto solo in fondo significa che qualunque interruzione
    # butta via tutto.
    scrivi_risultati(completo=False)

    # Stima del tempo rimanente, dalla media dei tentativi gia' fatti.
    medio = sum(t["minuti"] for t in TENTATIVI) / len(TENTATIVI)
    rimasti = (N_TENTATIVI - len(TENTATIVI)) * medio
    if rimasti > 0:
        print(f"          mancano ~{rimasti / 60:.1f} h "
              f"({medio:.1f} min per tentativo)", flush=True)

    return {
        "loss": 1.0 - auc,          # hyperopt MINIMIZZA questa quantita'
        "status": STATUS_OK,
        # Da qui in poi e' roba nostra, conservata per i grafici.
        "auc": float(auc),
        "lr": float(lr),
        "weight_decay": float(wd),
        "n_strati": n_strati,
        "n_unita": n_unita,
        "n_parametri": n_parametri,
        "epoca_scelta": epoca_scelta + 1,
        "n_epoche": len(storia["auc_val"]),
        "minuti": round(minuti, 2),
    }


def scrivi_risultati(completo):
    """
    Scrive su disco lo stato attuale della ricerca.

    Viene chiamata dopo ogni tentativo con completo=False e una volta alla
    fine con completo=True. Il file e' sempre lo stesso: se il programma si
    interrompe, quello che resta su disco e' un JSON valido con i tentativi
    fatti fino a quel momento e "completo": false a dirlo.
    """
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    spazio_descritto = {
        "lr": [LR_MIN, LR_MAX],
        "weight_decay": [WD_MIN, WD_MAX],
    }
    if CERCA_ARCHITETTURA:
        spazio_descritto["n_strati"] = [STRATI_MIN, STRATI_MAX]
        spazio_descritto["n_unita"] = [UNITA_MIN, UNITA_MAX]

    migliore = None
    if TENTATIVI:
        migliore = max(TENTATIVI, key=lambda t: t["auc"])

    uscita = {
        "completo": completo,
        "feature_set": FEATURE_SET,
        "modello": "deep",
        "stack": "moderno",
        "cerca_architettura": CERCA_ARCHITETTURA,
        "n_tentativi": N_TENTATIVI,
        "n_tentativi_fatti": len(TENTATIVI),
        "n_tentativi_casuali": TENTATIVI_CASUALI,
        "seme": SEME,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "batch": BATCH,
        "max_epoche": MAX_EPOCHE,
        "pazienza": PAZIENZA,
        "minuti_totali": round((time.time() - T_INIZIO) / 60, 1),
        "spazio": spazio_descritto,
        "migliore": migliore,
        "tentativi": TENTATIVI,
    }

    with open(PERCORSO_USCITA, "w") as f:
        json.dump(uscita, f, indent=2)


def main():
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    if CERCA_ARCHITETTURA:
        print(f"Ottimizzazione ARCHITETTURA + iperparametri - {FEATURE_SET}, "
              f"stack moderno")
        print(f"Spazio           : {2}-{8} strati, {100}-{1000} unita', "
              f"lr e weight decay")
    else:
        print(f"Ottimizzazione iperparametri - deep {FEATURE_SET}, stack moderno")
        print(f"Architettura     : {STRATI_PAPER} strati x {UNITA_PAPER} unita' "
              f"(quella del paper, fissa)")
    print(f"Tentativi        : {N_TENTATIVI} (di cui {TENTATIVI_CASUALI} casuali)")
    print(f"Eventi train     : {N_TRAIN:,}  (batch {BATCH})")
    print(f"Eventi val       : {N_VAL:,}")
    print(f"Epoche massime   : {MAX_EPOCHE}  "
          f"(~{N_TRAIN // BATCH * MAX_EPOCHE:,} aggiornamenti per tentativo)")
    print(f"Pazienza         : {PAZIENZA}  (come i training finali)")
    print(f"Seme fisso       : {SEME}")
    print(f"Dispositivo      : {DISPOSITIVO}")
    print(f"Uscita           : {PERCORSO_USCITA.name}  "
          f"(riscritto dopo ogni tentativo)")
    print("=" * 72)
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
    # I tentativi sono gia' in TENTATIVI, riempita da obiettivo(). Qui si
    # riscrive il file un'ultima volta, stavolta marcato come completo.
    tentativi = TENTATIVI
    migliore = max(tentativi, key=lambda t: t["auc"])
    scrivi_risultati(completo=True)
    percorso = PERCORSO_USCITA

    # --- riepilogo a schermo ----------------------------------------------
    print()
    print("=" * 72)
    print("MIGLIORE CONFIGURAZIONE")
    print("=" * 72)
    print(f"  learning rate : {migliore['lr']:.3e}")
    print(f"  weight decay  : {migliore['weight_decay']:.3e}")
    if CERCA_ARCHITETTURA:
        print(f"  architettura  : {migliore['n_strati']} strati x "
              f"{migliore['n_unita']} unita'")
        print(f"  parametri     : {migliore['n_parametri']:,}")
    print(f"  AUC           : {migliore['auc']:.4f}")
    print(f"  tentativo n.  : {migliore['tentativo']} di {N_TENTATIVI}")
    print()

    # I cinque migliori: se sono tutti vicini fra loro, il risultato e'
    # robusto; se sono sparsi, la superficie e' piatta e la scelta conta poco.
    ordinati = sorted(tentativi, key=lambda t: -t["auc"])[:5]
    print("  I cinque migliori:")
    if CERCA_ARCHITETTURA:
        print(f"  {'lr':>10}  {'wd':>10}  {'strati':>7}  {'unita':>6}  "
              f"{'AUC':>7}  {'epoche':>7}")
        for t in ordinati:
            print(f"  {t['lr']:10.2e}  {t['weight_decay']:10.2e}  "
                  f"{t['n_strati']:7d}  {t['n_unita']:6d}  "
                  f"{t['auc']:7.4f}  {t['n_epoche']:7d}")
    else:
        print(f"  {'lr':>10}  {'wd':>10}  {'AUC':>7}  {'epoche':>7}")
        for t in ordinati:
            print(f"  {t['lr']:10.2e}  {t['weight_decay']:10.2e}  "
                  f"{t['auc']:7.4f}  {t['n_epoche']:7d}")

    print()
    print(f"Tempo totale: {minuti_totali:.1f} min")
    print(f"Salvato in {percorso.name}")
    print("=" * 72)
    print()
    print(f"NOTA: valori trovati su {N_TRAIN:,} eventi; i training finali ne")
    print(f"usano 10.000.000 ({10_000_000 / N_TRAIN:.0f} volte tanto). Learning")
    print("rate e weight decay ottimi tendono a scendere al crescere dei dati:")
    print("i valori trovati sono, se mai, leggermente per eccesso.")
    if CERCA_ARCHITETTURA:
        print()
        print("NOTA: con MAX_EPOCHE fissato, le reti piu' grandi hanno meno")
        print("epoche di quante ne servirebbero. Il risultato va letto come")
        print("'migliore architettura a parita' di budget di epoche'.")


if __name__ == "__main__":
    main()