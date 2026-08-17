"""
Prepara i dati per l'addestramento: divisione, standardizzazione, scelta feature.

"""

from pathlib import Path

import numpy as np

from features import INDICI


PROGETTO = Path(__file__).resolve().parent.parent
CARTELLA_DATI = PROGETTO / "data" / "processed"

# Dimensioni degli insiemi, come nel paper.
# Il test set sono gli ULTIMI 500.000 eventi del file: e' la convenzione
# standard di questo dataset, serve per poter confrontare i risultati
# con quelli della letteratura.
N_TEST = 500_000
N_VAL = 500_000
N_TRAIN = 2_600_000

# ---------------------------------------------------------------------------

def prepara_dati(feature_set="complete",
                 cartella=CARTELLA_DATI,
                 n_train=N_TRAIN,
                 n_val=N_VAL,
                 n_test=N_TEST,
                 silenzioso=False):
    """
    Restituisce un dizionario con le tre coppie (X, y) gia' pronte.

    feature_set puo' essere "low", "high" oppure "complete".
    """

    if feature_set not in INDICI:
        raise ValueError(f"feature_set deve essere uno di {list(INDICI)}")

    # accetta sia una stringa sia un Path
    cartella = Path(cartella)

    # --- 1. carichiamo gli array completi ---------------------------------
    X = np.load(cartella / "X.npy", mmap_mode="r")
    y = np.load(cartella / "y.npy")
    n_totale = len(X)

    if n_train + n_val + n_test > n_totale:
        raise ValueError(
            f"Servono {n_train + n_val + n_test:,} eventi ma il file ne ha "
            f"{n_totale:,}. Riduci n_train, n_val o n_test."
        )

    # --- 2. dividiamo ------------------------------------------------------
    # test: gli ultimi n_test eventi
    # val:  i n_val eventi subito prima
    # train: i primi n_train eventi
    inizio_test = n_totale - n_test
    inizio_val = inizio_test - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[inizio_val:inizio_test], y[inizio_val:inizio_test]
    X_test, y_test = X[inizio_test:], y[inizio_test:]

    # --- 3. teniamo solo le colonne del set richiesto ----------------------
    colonne = INDICI[feature_set]
    X_train = X_train[:, colonne]
    X_val = X_val[:, colonne]
    X_test = X_test[:, colonne]

    # --- 4. standardizziamo ------------------------------------------------
    # Le statistiche si calcolano SOLO sul training set, mai su val o test:
    # altrimenti informazione dal test filtrerebbe nella trasformazione.
    media, dev, positiva = calcola_statistiche(X_train)

    X_train = standardizza(X_train, media, dev, positiva)
    X_val = standardizza(X_val, media, dev, positiva)
    X_test = standardizza(X_test, media, dev, positiva)

    if not silenzioso:
        print(f"feature set : {feature_set}  ({len(colonne)} colonne)")
        print(f"  train : {len(X_train):>9,} eventi")
        print(f"  val   : {len(X_val):>9,} eventi")
        print(f"  test  : {len(X_test):>9,} eventi")
        print(f"  colonne riscalate a media 1 (positive) : {int(positiva.sum())}")
        print(f"  colonne standardizzate a media 0       : {int((~positiva).sum())}")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "statistiche": (media, dev, positiva),
    }


def calcola_statistiche(X_train):
    """
    Calcola media e deviazione standard di ogni colonna del training set,
    e segna quali colonne sono strettamente positive.
    """
    media = X_train.mean(axis=0)
    dev = X_train.std(axis=0)
    positiva = X_train.min(axis=0) > 0

    # Se una colonna e' costante la deviazione standard e' zero: mettiamo 1
    # per non dividere per zero (la colonna diventera' tutta zeri).
    dev[dev == 0] = 1.0

    return media, dev, positiva


def standardizza(X, media, dev, positiva):
    """
    Applica la trasformazione descritta nei Methods del paper.

    Colonne che possono essere negative (eta, phi, ...):
        (x - media) / deviazione standard      ->  media 0, dev 1

    Colonne strettamente positive (pT, masse invarianti, ...):
        x / media                              ->  media 1, sempre positive

    Le seconde non vengono centrate a zero perche' per quelle grandezze
    lo zero e' un estremo fisico del dominio, non un valore centrale.
    """
    X_out = np.empty_like(X)

    normale = ~positiva
    X_out[:, normale] = (X[:, normale] - media[normale]) / dev[normale]
    X_out[:, positiva] = X[:, positiva] / media[positiva]

    return X_out


# Prova rapida: "python src/data.py"
if __name__ == "__main__":
    dati = prepara_dati(feature_set="low")
    X_train, y_train = dati["train"]
    print()
    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("frazione di segnale nel train:", round(float(y_train.mean()), 4))
