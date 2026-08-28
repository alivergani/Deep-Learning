"""
Controllo della classificazione delle colonne in "positive" e "da standardizzare".

La regola nel codice e' empirica: una colonna e' considerata positiva se il
suo minimo sul training set e' maggiore di zero. Qui verifichiamo che il
risultato coincida con quello che ci aspettiamo dalla fisica.

Uso:  python src/controlla_colonne.py
"""

import features
from data import prepara_dati

# Nomi delle 28 colonne del dataset HIGGS, nell'ordine del file UCI.
# Se features.py ha gia' una lista di nomi, usiamo quella.
NOMI_DEFAULT = [
    "lepton pT", "lepton eta", "lepton phi",
    "missing energy magnitude", "missing energy phi",
    "jet1 pt", "jet1 eta", "jet1 phi", "jet1 b-tag",
    "jet2 pt", "jet2 eta", "jet2 phi", "jet2 b-tag",
    "jet3 pt", "jet3 eta", "jet3 phi", "jet3 b-tag",
    "jet4 pt", "jet4 eta", "jet4 phi", "jet4 b-tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]

NOMI = getattr(features, "NOMI", NOMI_DEFAULT)


for feature_set in ["low", "high", "complete"]:

    print("=" * 68)
    print(f"FEATURE SET: {feature_set}")
    print("=" * 68)

    dati = prepara_dati(feature_set=feature_set, silenzioso=True)
    media, dev, positiva = dati["statistiche"]
    colonne = features.INDICI[feature_set]

    print(f"{'col':>4}  {'nome':<28} {'trattamento':<16} {'media':>9}")
    print("-" * 68)

    for i, indice in enumerate(colonne):
        nome = NOMI[indice] if indice < len(NOMI) else f"colonna {indice}"
        trattamento = "positiva (x/m)" if positiva[i] else "standardizzata"
        print(f"{indice:>4}  {nome:<28} {trattamento:<16} {media[i]:>9.4f}")

    n_pos = int(positiva.sum())
    print("-" * 68)
    print(f"positive: {n_pos}   standardizzate: {len(colonne) - n_pos}")
    print()