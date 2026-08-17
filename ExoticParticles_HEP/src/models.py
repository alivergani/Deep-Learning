"""
Le architetture del paper: una rete shallow e una rete profonda.

Sono la stessa cosa, cambia solo il numero di strati nascosti.

Uso tipico:

    from models import rete_profonda, rete_shallow, conta_parametri

    modello = rete_profonda(n_input=21)
    print(conta_parametri(modello))
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PARAMETRI dell'architettura scelta dagli autori
# ---------------------------------------------------------------------------

N_UNITA = 300              # unita' per strato nascosto

# Il paper dice "rete a 5 strati". Il conteggio dei parametri riportato
# nell'articolo (279.901) torna solo contando lo strato di uscita fra i 5,
# cioe' con 4 strati nascosti. Vedi il calcolo nel README.
N_STRATI_PROFONDA = 4
N_STRATI_SHALLOW = 1

# Deviazioni standard per l'inizializzazione dei pesi (sezione Methods).
STD_PRIMO = 0.1
STD_INTERMEDI = 0.05
STD_USCITA = 0.001

# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """
    Rete completamente connessa per la classificazione binaria.

    L'uscita e' un solo numero per evento, SENZA sigmoide: e' un "logit",
    cioe' un valore che puo' andare da meno infinito a piu' infinito.
    La sigmoide viene applicata dentro la funzione di perdita
    (BCEWithLogitsLoss), che e' numericamente piu' stabile.
    """

    def __init__(self, n_input, n_strati=N_STRATI_PROFONDA,
                 n_unita=N_UNITA, attivazione="tanh"):
        super().__init__()

        # Gli strati nascosti, uno dopo l'altro.
        # ModuleList e' una lista che PyTorch sa riconoscere: cosi' i pesi
        # di ogni strato vengono registrati e addestrati.
        self.nascosti = nn.ModuleList()
        n_in = n_input
        for _ in range(n_strati):
            self.nascosti.append(nn.Linear(n_in, n_unita))
            n_in = n_unita

        # Lo strato di uscita: da n_unita a 1 solo numero.
        self.uscita = nn.Linear(n_in, 1)

        # La funzione di attivazione.
        # "tanh" e' quella del paper, "relu" servira' per il confronto
        # con lo stack moderno.
        if attivazione == "tanh":
            self.f = torch.tanh
        elif attivazione == "relu":
            self.f = torch.relu
        else:
            raise ValueError("attivazione deve essere 'tanh' o 'relu'")

        self.attivazione = attivazione
        inizializza_pesi(self)

    def forward(self, x, restituisci_attivazioni=False):
        """
        Passaggio in avanti.

        Con restituisci_attivazioni=True restituisce anche l'uscita di ogni
        strato nascosto: serve per il linear probing, dove si vuole vedere
        cosa la rete ha costruito internamente.
        """
        attivazioni = []

        for strato in self.nascosti:
            x = self.f(strato(x))
            attivazioni.append(x)

        logit = self.uscita(x)

        if restituisci_attivazioni:
            return logit, attivazioni
        return logit


def inizializza_pesi(modello):
    """
    Inizializzazione descritta nei Methods del paper.

    I pesi partono da una gaussiana di media zero, con ampiezza diversa
    a seconda dello strato:
        primo strato    -> 0.1
        strati interni  -> 0.05
        strato di uscita-> 0.001

    Con la tanh l'ampiezza iniziale conta molto: pesi troppo grandi mandano
    le attivazioni nella zona piatta della tanh, dove la derivata e' quasi
    nulla e l'addestramento si blocca; pesi troppo piccoli fanno spegnere
    il segnale strato dopo strato.
    """
    for i, strato in enumerate(modello.nascosti):
        std = STD_PRIMO if i == 0 else STD_INTERMEDI
        nn.init.normal_(strato.weight, mean=0.0, std=std)
        nn.init.zeros_(strato.bias)

    nn.init.normal_(modello.uscita.weight, mean=0.0, std=STD_USCITA)
    nn.init.zeros_(modello.uscita.bias)


def rete_profonda(n_input, attivazione="tanh"):
    """La rete profonda del paper: 4 strati nascosti da 300 unita'."""
    return MLP(n_input, n_strati=N_STRATI_PROFONDA, attivazione=attivazione)


def rete_shallow(n_input, n_unita=N_UNITA, attivazione="tanh"):
    """La rete tradizionale: 1 solo strato nascosto."""
    return MLP(n_input, n_strati=N_STRATI_SHALLOW,
               n_unita=n_unita, attivazione=attivazione)


def conta_parametri(modello):
    """Numero totale di pesi e bias addestrabili."""
    return sum(p.numel() for p in modello.parameters() if p.requires_grad)


# Prova rapida: "python src/models.py"
if __name__ == "__main__":
    print("Verifica dei conteggi riportati nel paper")
    print()

    m = rete_profonda(n_input=28)
    print(f"  rete profonda, 28 input : {conta_parametri(m):>8,} parametri"
          f"   (il paper dice 279.901)")

    m = rete_shallow(n_input=28, n_unita=10_000)
    print(f"  rete shallow, 10k unita': {conta_parametri(m):>8,} parametri"
          f"   (il paper dice 300.001)")

    print()
    print("Prova di funzionamento su dati finti")
    modello = rete_profonda(n_input=21)
    x = torch.randn(5, 21)

    logit = modello(x)
    print("  forma dell'uscita        :", tuple(logit.shape))

    logit, att = modello(x, restituisci_attivazioni=True)
    print("  numero di attivazioni    :", len(att))
    print("  forma di ogni attivazione:", [tuple(a.shape) for a in att])