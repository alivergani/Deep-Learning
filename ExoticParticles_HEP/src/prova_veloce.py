"""
Prova rapida dei due stack di ottimizzazione, su pochi eventi.

Non serve a ottenere risultati: serve solo a vedere che il codice gira
e che il ramo 2014 non e' cambiato.

Uso:  python src/prova_veloce.py
"""

from data import prepara_dati
from models import rete_profonda
from train import addestra

# Quanti eventi usare. Bastano pochi per una prova: deve durare minuti,
# non ore.
N_TRAIN = 50_000
N_VAL = 10_000


def taglia(dati):
    """Tiene solo i primi N eventi di train e validation."""
    X, y = dati["train"]
    dati["train"] = (X[:N_TRAIN], y[:N_TRAIN])
    X, y = dati["val"]
    dati["val"] = (X[:N_VAL], y[:N_VAL])
    return dati


dati = taglia(prepara_dati(feature_set="low"))
n_input = dati["train"][0].shape[1]
print(f"Prova con {N_TRAIN:,} eventi di train, {n_input} variabili in ingresso")


# ---------------------------------------------------------------------------
# 1) stack 2014 - deve dare sempre lo stesso risultato
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STACK 2014  (tanh + SGD)")
print("=" * 60)

modello = rete_profonda(n_input=n_input)
storia_2014 = addestra(modello, dati,
                       seme=0,
                       batch=1000,
                       epoche_rampa=2,
                       max_epoche=5,
                       silenzioso=True)

print("perdite di validation:")
for p in storia_2014["perdita_val"]:
    print(f"  {p:.10f}")


# ---------------------------------------------------------------------------
# 2) stack moderno - deve solo girare senza errori e non divergere
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STACK MODERNO  (relu + He + AdamW)")
print("=" * 60)

modello = rete_profonda(n_input=n_input, attivazione="relu")
storia_moderna = addestra(modello, dati,
                          seme=0,
                          batch=1000,
                          max_epoche=15,
                          ottimizzatore="adamw",
                          silenzioso=False)


# ---------------------------------------------------------------------------
# riepilogo
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RIEPILOGO")
print("=" * 60)
print(f"  2014    : miglior AUC {max(storia_2014['auc_val']):.4f} "
      f"in {len(storia_2014['auc_val'])} epoche")
print(f"  moderno : miglior AUC {max(storia_moderna['auc_val']):.4f} "
      f"in {len(storia_moderna['auc_val'])} epoche")
print()
print("Le due righe non sono confrontabili (numero di epoche diverso):")
print("serve solo a vedere che entrambi gli stack funzionano.")