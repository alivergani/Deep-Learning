"""
Addestramento delle reti.

Il file contiene DUE stack di ottimizzazione:

  1) ottimizzatore="sgd"    -> quello dei Methods del paper (default)
     SGD + rampa di momentum 0.9 -> 0.99 + decadimento del lr ad ogni batch

  2) ottimizzatore="adamw"  -> stack moderno
     AdamW + riduzione del lr quando la validation smette di migliorare

Il default e' sempre il comportamento 2014, quindi i run gia' fatti
restano riproducibili.

TENSORBOARD
-----------
Passando logdir="runs/nome_della_run" l'addestramento scrive anche i log
per TensorBoard. Con logdir=None (default) non succede nulla e il
comportamento e' identico a prima.

Si registrano:
  - a OGNI epoca: perdita di train e validation, AUC, learning rate,
    minuti impiegati, norma dei gradienti strato per strato
  - ogni EPOCHE_DETTAGLI epoche: istogrammi dei pesi e delle attivazioni

Per guardarli:
    tensorboard --logdir=runs
    (e poi il browser su localhost:6006)

Da remoto serve un tunnel:
    ssh -L 6006:localhost:6006 utente@macchina

Uso tipico (da notebook):

    from data import prepara_dati
    from models import rete_profonda
    from train import addestra, valuta

    dati = prepara_dati(feature_set="low")

    # riproduzione
    m = rete_profonda(n_input=21)
    storia = addestra(m, dati)

    # stack moderno, con log per TensorBoard
    m = rete_profonda(n_input=21, attivazione="relu")
    storia = addestra(m, dati, ottimizzatore="adamw",
                      logdir="runs/deep_low_moderno")
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# PARAMETRI dello stack 2014 (valori del paper)
# ---------------------------------------------------------------------------

BATCH = 100                  # eventi per aggiornamento dei pesi

LR_INIZIALE = 0.05           # learning rate di partenza
LR_DIVISORE = 1.0000002      # ad ogni batch: lr = lr / questo numero
LR_MINIMO = 1e-6             # sotto questo valore il lr non scende piu'

MOMENTUM_INIZIALE = 0.9
MOMENTUM_FINALE = 0.99
EPOCHE_RAMPA = 200           # epoche in cui il momentum sale da 0.9 a 0.99

WEIGHT_DECAY = 1e-5          # regolarizzazione L2

# ---------------------------------------------------------------------------
# PARAMETRI dello stack moderno
# ---------------------------------------------------------------------------

# Adam normalizza il passo con una stima della scala dei gradienti, quindi
# lavora su learning rate molto piu' piccoli di SGD. 1e-3 e' il valore
# standard, 0.05 farebbe divergere subito la rete.
LR_ADAMW = 2.1e-3             # trovato con TPE su 1M eventi; era 1e-3

# In AdamW il weight decay e' "disaccoppiato": viene applicato ai pesi in
# modo diretto invece di essere sommato al gradiente. A parita' di numero
# l'effetto e' piu' forte che in SGD, per questo si usa un valore piu' alto
# ma comunque prudente.
WEIGHT_DECAY_ADAMW = 9.76e-3  # trovato con TPE su 1M eventi; era 1e-4

# beta1 e' l'analogo del momentum in Adam: qui resta fisso, non c'e' rampa.
BETA1 = 0.9
BETA2 = 0.999

# Quando la validation non migliora per qualche epoca, il lr viene dimezzato.
PLATEAU_FATTORE = 0.5
PLATEAU_PAZIENZA = 3         # sempre piu' piccola della pazienza di stop

# ---------------------------------------------------------------------------
# PARAMETRI comuni
# ---------------------------------------------------------------------------

MAX_EPOCHE = 1000            # limite di sicurezza
PAZIENZA = 10                # epoche senza miglioramento prima di fermarsi
MIGLIORAMENTO_MINIMO = 1e-5  # miglioramento relativo che conta come progresso

# ---------------------------------------------------------------------------
# PARAMETRI di TensorBoard
# ---------------------------------------------------------------------------

# Ogni quante epoche registrare istogrammi di pesi e attivazioni.
# Sono la parte costosa: un istogramma di 90.000 pesi per strato pesa molto
# piu' di un singolo numero. Ogni 10 epoche si vede benissimo l'evoluzione
# senza rallentare l'addestramento.
EPOCHE_DETTAGLI = 10

# Quanti eventi usare per gli istogrammi delle attivazioni. Non servono
# tutti: la distribuzione si stima benissimo con qualche migliaio.
N_EVENTI_ATTIVAZIONI = 2000

# ---------------------------------------------------------------------------


def scegli_dispositivo():
    """Usa la GPU se disponibile, altrimenti la CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def crea_ottimizzatore(modello, nome, lr, weight_decay):
    """
    Costruisce l'ottimizzatore giusto e, se serve, il suo scheduler.

    Restituisce (ottimizzatore, scheduler). Nello stack 2014 lo scheduler
    e' None, perche' il learning rate viene aggiornato a mano dentro il
    ciclo sui batch.
    """
    if nome == "sgd":
        ottimizzatore = torch.optim.SGD(
            modello.parameters(),
            lr=lr,
            momentum=MOMENTUM_INIZIALE,
            weight_decay=weight_decay,
        )
        return ottimizzatore, None

    if nome == "adamw":
        ottimizzatore = torch.optim.AdamW(
            modello.parameters(),
            lr=lr,
            betas=(BETA1, BETA2), # le betas sono delle costanti di Adam, non vanno confuse con il momentum di SGD
            weight_decay=weight_decay,
        )
        # Scheduler "a plateau": guarda la perdita di validation e dimezza
        # il lr quando smette di scendere. Lo scegliamo perche' non ha
        # bisogno di sapere in anticipo quante epoche durera' il training,
        # e quindi convive bene con l'early stopping. Uno scheduler a
        # coseno, per esempio, richiederebbe di fissare prima il numero
        # totale di epoche.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            ottimizzatore,
            mode="min",
            factor=PLATEAU_FATTORE,
            patience=PLATEAU_PAZIENZA,
        )
        return ottimizzatore, scheduler

    raise ValueError("ottimizzatore deve essere 'sgd' o 'adamw'")


# ---------------------------------------------------------------------------
# FUNZIONI DI SUPPORTO PER TENSORBOARD
# ---------------------------------------------------------------------------

def norme_gradienti(modello):
    """
    Restituisce la norma del gradiente di ogni strato nascosto, piu' quella
    dello strato di uscita.

    A cosa serve: il gradiente viene calcolato all'uscita e propagato
    all'indietro. Se attraversando gli strati si rimpicciolisce troppo, i
    primi strati ricevono un segnale debolissimo e imparano lentamente.
    Confrontare la norma al primo e all'ultimo strato mostra direttamente
    questo fenomeno, che e' esattamente cio' che l'inizializzazione di He
    e' progettata a evitare.

    Va chiamata DOPO backward() e PRIMA di step(): dopo l'aggiornamento i
    gradienti sono ancora li', ma li azzeriamo al giro successivo.
    """
    norme = {}
    for k, strato in enumerate(modello.nascosti, start=1):
        if strato.weight.grad is not None:
            norme[f"strato{k}"] = strato.weight.grad.norm().item()
    if modello.uscita.weight.grad is not None:
        norme["uscita"] = modello.uscita.weight.grad.norm().item()
    return norme


def registra_dettagli(writer, modello, X_campione, epoca):
    """
    Scrive su TensorBoard gli istogrammi dei pesi e delle attivazioni.

    PESI: la distribuzione dei pesi di ogni strato. All'inizio e' quella
    dell'inizializzazione (gaussiana stretta con tanh, piu' larga con He);
    durante l'addestramento si deforma, e il confronto fra i due stack
    mostra quanto i pesi si muovono davvero.

    ATTIVAZIONI: la distribuzione dei valori in uscita da ogni strato
    nascosto, su un campione fisso di eventi. Serve a vedere se il segnale
    si attenua scendendo nella rete. Con la ReLU registriamo anche la
    frazione di unita' esattamente a zero (la "sparsita'"): e' una
    caratteristica della ReLU che la tanh non ha.

    Il modello deve essere gia' in modalita' eval quando si chiama.
    """
    # --- pesi -------------------------------------------------------------
    for k, strato in enumerate(modello.nascosti, start=1):
        writer.add_histogram(f"pesi/strato{k}", strato.weight, epoca)
    writer.add_histogram("pesi/uscita", modello.uscita.weight, epoca)

    # --- attivazioni ------------------------------------------------------
    with torch.no_grad():
        _, attivazioni = modello(X_campione, restituisci_attivazioni=True)

    for k, a in enumerate(attivazioni, start=1):
        writer.add_histogram(f"attivazioni/strato{k}", a, epoca)
        # Ampiezza tipica: un solo numero, comodo da confrontare fra stack.
        writer.add_scalar(f"ampiezza_attivazioni/strato{k}",
                          a.std().item(), epoca)
        # Frazione di unita' spente (significativa solo con ReLU).
        frazione_zeri = (a == 0).float().mean().item()
        writer.add_scalar(f"sparsita/strato{k}", frazione_zeri, epoca)


# ---------------------------------------------------------------------------

def addestra(modello, dati,
             batch=BATCH,
             lr_iniziale=None,
             epoche_rampa=EPOCHE_RAMPA,
             max_epoche=MAX_EPOCHE,
             pazienza=PAZIENZA,
             weight_decay=None,
             ottimizzatore="sgd",
             dispositivo=None,
             seme=None,
             logdir=None,
             epoche_dettagli=EPOCHE_DETTAGLI,
             silenzioso=False):
    """
    Addestra il modello e restituisce lo storico delle metriche.

    Alla fine i pesi del modello sono quelli dell'epoca migliore,
    non quelli dell'ultima epoca.

    lr_iniziale e weight_decay, se lasciati a None, prendono il valore
    adatto all'ottimizzatore scelto.

    logdir: se e' un percorso, scrive i log per TensorBoard in quella
    cartella. Se e' None (default) non scrive nulla.
    """

    # --- valori di default che dipendono dall'ottimizzatore ----------------
    if lr_iniziale is None:
        lr_iniziale = LR_INIZIALE if ottimizzatore == "sgd" else LR_ADAMW
    if weight_decay is None:
        weight_decay = WEIGHT_DECAY if ottimizzatore == "sgd" else WEIGHT_DECAY_ADAMW

    if seme is not None:
        torch.manual_seed(seme)

    if dispositivo is None:
        dispositivo = scegli_dispositivo()

    X_train, y_train = dati["train"]
    X_val, y_val = dati["val"]

    # Portiamo tutto su tensori PyTorch, una volta sola.
    # Se i dati stanno nella memoria della GPU, l'addestramento e' molto
    # piu' veloce perche' non si copia nulla ad ogni batch.
    X_train = torch.from_numpy(np.ascontiguousarray(X_train)).to(dispositivo)
    y_train = torch.from_numpy(np.ascontiguousarray(y_train)).to(dispositivo)

    modello = modello.to(dispositivo)

    # --- TensorBoard: si prepara solo se richiesto -------------------------
    writer = None
    X_campione = None
    if logdir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(logdir)
        # Campione FISSO di eventi per gli istogrammi delle attivazioni:
        # sempre gli stessi, cosi' le differenze fra epoche vengono dalla
        # rete e non dal campione.
        campione = np.ascontiguousarray(X_val[:N_EVENTI_ATTIVAZIONI])
        X_campione = torch.from_numpy(campione).to(dispositivo)

    perdita_fn = nn.BCEWithLogitsLoss()
    opt, scheduler = crea_ottimizzatore(modello, ottimizzatore,
                                        lr_iniziale, weight_decay)

    n = len(X_train)
    lr = lr_iniziale

    # Indice del primo evento dell'ultimo batch dell'epoca: solo su quel
    # batch calcoliamo le norme dei gradienti, per non pagare il conto ad
    # ogni aggiornamento.
    ultimo_batch = ((n - batch) // batch) * batch

    storia = {"perdita_train": [], "perdita_val": [], "auc_val": [],
              "lr": [], "momentum": []}

    migliore_perdita = float("inf")
    migliori_pesi = copy.deepcopy(modello.state_dict())
    epoche_senza_miglioramento = 0

    if not silenzioso:
        print(f"Addestramento su {dispositivo}, {n:,} eventi, batch da {batch}")
        print(f"Ottimizzatore: {ottimizzatore}, lr iniziale {lr_iniziale}")
        print(f"{n // batch:,} aggiornamenti per epoca")
        if writer is not None:
            print(f"Log TensorBoard in {logdir}")
        print()

    t0 = time.time()

    for epoca in range(max_epoche):

        t_epoca = time.time()

        # --- momentum ------------------------------------------------------
        if ottimizzatore == "sgd":
            # Sale linearmente da 0.9 a 0.99, come nel paper.
            frazione = min(epoca / epoche_rampa, 1.0)
            momentum = MOMENTUM_INIZIALE + frazione * (MOMENTUM_FINALE - MOMENTUM_INIZIALE)
            for gruppo in opt.param_groups:
                gruppo["momentum"] = momentum
        else:
            # In AdamW l'analogo del momentum e' beta1, che resta costante.
            # Lo registriamo comunque per non cambiare le chiavi di "storia".
            momentum = BETA1

        # --- un giro su tutti i dati in ordine casuale ---------------------
        modello.train()
        somma_perdita = 0.0
        n_batch = 0
        norme = None

        ordine = torch.randperm(n, device=dispositivo)

        for i in range(0, n - batch + 1, batch):
            indici = ordine[i:i + batch]
            xb = X_train[indici]
            yb = y_train[indici]

            opt.zero_grad()                    # azzera i gradienti precedenti
            logit = modello(xb)                # passaggio in avanti
            perdita = perdita_fn(logit, yb)    # quanto sbaglia
            perdita.backward()                 # calcola i gradienti

            # Le norme dei gradienti vanno lette qui: dopo backward, prima
            # che il giro successivo li azzeri. Solo sull'ultimo batch.
            if writer is not None and i == ultimo_batch:
                norme = norme_gradienti(modello)

            opt.step()                         # aggiorna i pesi

            # --- learning rate: scende a OGNI batch, non a ogni epoca ------
            # Solo nello stack 2014. Nello stack moderno il lr cambia una
            # volta per epoca, deciso dallo scheduler a fine epoca.
            if ottimizzatore == "sgd" and lr > LR_MINIMO:
                lr = max(lr / LR_DIVISORE, LR_MINIMO)
                for gruppo in opt.param_groups:
                    gruppo["lr"] = lr

            somma_perdita += perdita.item()
            n_batch += 1

        perdita_train = somma_perdita / n_batch

        # --- valutazione su validation -------------------------------------
        perdita_val, auc_val = valuta(modello, X_val, y_val, dispositivo=dispositivo)

        # --- scheduler dello stack moderno ---------------------------------
        if scheduler is not None:
            scheduler.step(perdita_val)

        # Leggiamo il lr davvero in uso, qualunque sia lo stack.
        lr = opt.param_groups[0]["lr"]

        minuti_epoca = (time.time() - t_epoca) / 60

        storia["perdita_train"].append(perdita_train)
        storia["perdita_val"].append(perdita_val)
        storia["auc_val"].append(auc_val)
        storia["lr"].append(lr)
        storia["momentum"].append(momentum)

        # --- scrittura su TensorBoard --------------------------------------
        if writer is not None:
            # I nomi con la barra creano dei raggruppamenti nell'interfaccia:
            # tutto cio' che inizia con "perdita/" finisce nello stesso
            # pannello, e le due curve si vedono sovrapposte.
            writer.add_scalar("perdita/train", perdita_train, epoca)
            writer.add_scalar("perdita/validation", perdita_val, epoca)
            writer.add_scalar("auc/validation", auc_val, epoca)
            writer.add_scalar("ottimizzazione/learning_rate", lr, epoca)
            writer.add_scalar("ottimizzazione/momentum", momentum, epoca)
            writer.add_scalar("tempo/minuti_per_epoca", minuti_epoca, epoca)

            if norme is not None:
                for nome_strato, valore in norme.items():
                    writer.add_scalar(f"gradienti/{nome_strato}", valore, epoca)

            # Gli istogrammi solo ogni tanto: sono la parte costosa.
            # modello.eval() e' gia' stato messo da valuta().
            if epoca % epoche_dettagli == 0:
                registra_dettagli(writer, modello, X_campione, epoca)

        if not silenzioso:
            print(f"epoca {epoca + 1:4d} | train {perdita_train:.5f} | "
                  f"val {perdita_val:.5f} | AUC {auc_val:.4f} | "
                  f"lr {lr:.6f} | mom {momentum:.3f} | "
                  f"{(time.time() - t0) / 60:.1f} min")

        # --- early stopping -------------------------------------------------
        # Conta come miglioramento solo una riduzione relativa apprezzabile.
        if perdita_val < migliore_perdita * (1 - MIGLIORAMENTO_MINIMO):
            migliore_perdita = perdita_val
            migliori_pesi = copy.deepcopy(modello.state_dict())
            epoche_senza_miglioramento = 0
        else:
            epoche_senza_miglioramento += 1

        # Nel paper ci si puo' fermare solo dopo che il momentum ha
        # raggiunto il valore massimo: prima di allora il training e'
        # ancora "in rampa" e una stasi non significa convergenza.
        # Nello stack moderno la rampa non esiste, quindi il vincolo non
        # ha senso e va tolto: se lo lasciassimo, la rete non potrebbe
        # fermarsi prima dell'epoca 200 anche avendo gia' convergito,
        # e proprio la velocita' di convergenza e' cio' che vogliamo
        # misurare.
        if ottimizzatore == "sgd":
            puo_fermarsi = epoca >= epoche_rampa
        else:
            puo_fermarsi = True

        if puo_fermarsi and epoche_senza_miglioramento >= pazienza:
            if not silenzioso:
                print(f"\nFermato: nessun miglioramento da {pazienza} epoche.")
            break

    # Ripristiniamo i pesi dell'epoca migliore.
    modello.load_state_dict(migliori_pesi)

    if writer is not None:
        writer.close()          # svuota il buffer: senza, le ultime epoche
                                # potrebbero non finire su disco

    if not silenzioso:
        print(f"\nMigliore perdita di validation: {migliore_perdita:.5f}")
        print(f"Epoche eseguite: {len(storia['perdita_val'])}")
        print(f"Tempo totale: {(time.time() - t0) / 60:.1f} min")

    return storia


def valuta(modello, X, y, batch=10_000, dispositivo=None):
    """
    Calcola perdita e AUC su un insieme di dati.

    Restituisce (perdita, auc).
    """
    if dispositivo is None:
        dispositivo = next(modello.parameters()).device

    modello.eval()                  # spegne dropout e batchnorm, se presenti
    perdita_fn = nn.BCEWithLogitsLoss(reduction="sum")

    somma_perdita = 0.0
    logit_tutti = []

    # Si valuta a pezzi per non riempire la memoria con 500.000 eventi
    # in un colpo solo.
    with torch.no_grad():           # non serve calcolare i gradienti
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dispositivo)
            yb = torch.from_numpy(np.ascontiguousarray(y[i:i + batch])).to(dispositivo)

            logit = modello(xb)
            somma_perdita += perdita_fn(logit, yb).item()
            logit_tutti.append(logit.cpu())

    logit_tutti = torch.cat(logit_tutti).numpy().ravel()

    perdita = somma_perdita / len(X)

    # Per l'AUC non serve applicare la sigmoide: e' una funzione crescente,
    # quindi non cambia l'ordinamento degli eventi.
    auc = roc_auc_score(y.ravel(), logit_tutti)

    return perdita, auc


def predici(modello, X, batch=10_000, dispositivo=None):
    """Restituisce le probabilita' previste, come array numpy (N,)."""
    if dispositivo is None:
        dispositivo = next(modello.parameters()).device

    modello.eval()
    uscite = []

    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dispositivo)
            uscite.append(torch.sigmoid(modello(xb)).cpu())

    return torch.cat(uscite).numpy().ravel()