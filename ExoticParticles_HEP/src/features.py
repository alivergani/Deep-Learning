"""
Nomi delle colonne del dataset HIGGS e definizione dei tre set di feature.

Il file CSV originale ha 29 colonne:
    colonna  0      = label (1 = segnale, 0 = fondo)
    colonne  1-21   = 21 feature low-level
    colonne 22-28   = 7 feature high-level

Dopo aver separato la label, la matrice X ha 28 colonne.
Gli indici scritti qui sotto si riferiscono a X.
"""

# 21 feature low-level: le misure diracte del rivelatore.
# leptone (3) + missing energy (2) + 4 jet x 4 valori (16) = 21
LOW_LEVEL = [
    "lepton_pT",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_b_tag",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_b_tag",
    "jet3_pt",
    "jet3_eta",
    "jet3_phi",
    "jet3_b_tag",
    "jet4_pt",
    "jet4_eta",
    "jet4_phi",
    "jet4_b_tag",
]

# 7 feature high-level: le masse invarianti costruite dalle 21 precedenti.
# Sono anche i target del linear probing.
HIGH_LEVEL = [
    "m_jj",      # W -> jj
    "m_jjj",     # t -> Wb   (solo nel fondo)
    "m_lv",      # W -> l nu (uguale in segnale e fondo)
    "m_jlv",     # t -> Wb   (solo nel fondo)
    "m_bb",      # h0 -> b bbar
    "m_wbb",     # H+- -> W h0  (solo nel segnale)
    "m_wwbb",    # H0 -> W H+-  (solo nel segnale)
]

TUTTE = LOW_LEVEL + HIGH_LEVEL          # 28 in tutto

# Quali colonne di X usare per ciascun esperimento del paper.
INDICI = {
    "low":      list(range(0, 21)),     # colonne 0-20
    "high":     list(range(21, 28)),    # colonne 21-27
    "complete": list(range(0, 28)),     # tutte
}


# Piccolo controllo: eseguendo "python src/features.py" stampa un riepilogo.
if __name__ == "__main__":
    print("numero totale di feature:", len(TUTTE))
    for nome, indici in INDICI.items():
        print(f"  {nome:9s} -> {len(indici):2d} feature")