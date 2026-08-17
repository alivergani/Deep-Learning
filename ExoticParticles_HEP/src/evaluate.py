"""
Valutazione dei modelli: curve ROC, AUC, grafici come nel paper.

Attenzione alla convenzione: la Figura 7 del paper NON e' una ROC standard.
In fisica delle alte energie si mette in ascissa la signal efficiency
(quanti eventi di segnale vengono tenuti) e in ordinata la background
rejection (quanti eventi di fondo vengono scartati).

    signal efficiency   = TPR
    background rejection = 1 - FPR

Una ROC standard avrebbe FPR in ascissa e TPR in ordinata: e' la stessa
informazione, con l'asse verticale ribaltato.

Uso tipico:

    from evaluate import calcola_curva, disegna_curve

    ris = calcola_curva(modello, X_test, y_test, nome="DN low-level")
    disegna_curve([ris])
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from train import predici


PROGETTO = Path(__file__).resolve().parent.parent
CARTELLA_RISULTATI = PROGETTO / "results"


def calcola_curva(modello, X, y, nome="modello"):
    """
    Calcola la curva ROC e l'AUC di un modello su un insieme di dati.

    Restituisce un dizionario con tutto il necessario per i grafici.
    """
    probabilita = predici(modello, X)
    etichette = y.ravel()

    fpr, tpr, soglie = roc_curve(etichette, probabilita)
    auc = roc_auc_score(etichette, probabilita)

    return {
        "nome": nome,
        "auc": float(auc),
        "signal_efficiency": tpr,           # asse x della Figura 7
        "background_rejection": 1 - fpr,    # asse y della Figura 7
        "soglie": soglie,
        "probabilita": probabilita,
    }


def disegna_curve(risultati, titolo=None, salva_in=None):
    """
    Disegna una o piu' curve nello stile della Figura 7 del paper.

    risultati: lista di dizionari prodotti da calcola_curva.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    for ris in risultati:
        ax.plot(
            ris["signal_efficiency"],
            ris["background_rejection"],
            label=f"{ris['nome']} (AUC = {ris['auc']:.3f})",
            linewidth=1.5,
        )

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    if titolo:
        ax.set_title(titolo)

    fig.tight_layout()

    if salva_in is not None:
        fig.savefig(salva_in, dpi=150)
        print("Grafico salvato in", salva_in)

    return fig, ax


def disegna_storia(storia, titolo=None):
    """Disegna l'andamento della perdita e dell'AUC durante l'addestramento."""
    import matplotlib.pyplot as plt

    fig, assi = plt.subplots(1, 2, figsize=(11, 4))

    epoche = range(1, len(storia["perdita_train"]) + 1)

    assi[0].plot(epoche, storia["perdita_train"], label="train")
    assi[0].plot(epoche, storia["perdita_val"], label="validation")
    assi[0].set_xlabel("epoca")
    assi[0].set_ylabel("perdita")
    assi[0].legend()
    assi[0].grid(alpha=0.3)

    assi[1].plot(epoche, storia["auc_val"], color="tab:green")
    assi[1].set_xlabel("epoca")
    assi[1].set_ylabel("AUC su validation")
    assi[1].grid(alpha=0.3)

    if titolo:
        fig.suptitle(titolo)

    fig.tight_layout()
    return fig, assi


def salva_curva(ris, percorso):
    """
    Salva una curva su disco in formato .npz.

    Non salviamo le probabilita' di tutti gli eventi (sarebbero 500.000
    numeri per ogni run): bastano i punti della curva.
    """
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        percorso,
        nome=ris["nome"],
        auc=ris["auc"],
        signal_efficiency=ris["signal_efficiency"].astype(np.float32),
        background_rejection=ris["background_rejection"].astype(np.float32),
    )


def carica_curva(percorso):
    """Ricarica una curva salvata con salva_curva."""
    dati = np.load(percorso, allow_pickle=False)
    return {
        "nome": str(dati["nome"]),
        "auc": float(dati["auc"]),
        "signal_efficiency": dati["signal_efficiency"],
        "background_rejection": dati["background_rejection"],
    }


def tabella_auc(risultati_per_configurazione):
    """
    Costruisce la tabella dei risultati, come la Tabella I del paper.

    risultati_per_configurazione: dizionario del tipo
        {("deep", "low"): [auc_seme0, auc_seme1, ...], ...}

    Restituisce una stringa pronta da stampare.
    """
    righe = []
    righe.append(f"{'modello':10s} {'feature set':12s} {'AUC medio':>10s} {'dev.std':>9s} {'n':>3s}")
    righe.append("-" * 48)

    for (modello, feature_set), valori in sorted(risultati_per_configurazione.items()):
        valori = np.asarray(valori, dtype=float)
        righe.append(
            f"{modello:10s} {feature_set:12s} "
            f"{valori.mean():10.4f} {valori.std():9.4f} {len(valori):3d}"
        )

    return "\n".join(righe)


def raccogli_risultati(cartella=CARTELLA_RISULTATI):
    """
    Legge tutti i file di riepilogo prodotti da esperimenti.py e restituisce
    un dizionario {(modello, feature_set): [auc, auc, ...]}.
    """
    cartella = Path(cartella)
    raccolta = {}

    for file_json in sorted(cartella.glob("*_riepilogo.json")):
        with open(file_json) as f:
            info = json.load(f)
        chiave = (info["modello"], info["feature_set"])
        raccolta.setdefault(chiave, []).extend(info["auc_test"])

    return raccolta
