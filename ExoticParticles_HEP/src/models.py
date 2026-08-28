"""
Le architetture del paper: una rete shallow e una rete profonda.

Sono la stessa cosa, cambia solo il numero di strati nascosti.

Il file contiene DUE configurazioni:

  1) riproduzione 2014  -> attivazione="tanh"  (default)
     tanh + inizializzazione gaussiana con le sigma del paper

  2) stack moderno      -> attivazione="relu"
     ReLU + inizializzazione di He

L'attivazione e' l'unico interruttore: scegliendola si sceglie anche
l'inizializzazione giusta, perche' le due cose vanno insieme.
Il default e' sempre il comportamento 2014, quindi i run gia' fatti
restano riproducibili.

Uso tipico:

    from models import rete_profonda, rete_shallow, conta_parametri

    modello_2014     = rete_profonda(n_input=21)
    modello_moderno  = rete_profonda(n_input=21, attivazione="relu")
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PARAMETRI dell'architettura scelta dagli autori
# ---------------------------------------------------------------------------

N_UNITA = 300              # unita' per strato nascosto

# Il paper dice "rete a 5 strati". Il conteggio dei parametri riportato
# nell'articolo (279.901) torna solo contando lo strato di uscita fra i 5,
# cioe' con 4 strati nascosti.
N_STRATI_PROFONDA = 4
N_STRATI_SHALLOW = 1

# Deviazioni standard per l'inizializzazione dei pesi (sezione Methods).
# Valgono solo per la versione con tanh.
STD_PRIMO = 0.1
STD_INTERMEDI = 0.05

# Sigma dello strato di uscita. La usiamo in ENTRAMBE le versioni:
# pesi di uscita quasi nulli significano logit quasi nulli, cioe' la rete
# parte prevedendo 0.5 per tutti gli eventi. E' una partenza sensata sia
# nel 2014 sia oggi, e cosi' l'unica differenza fra i due stack sta negli
# strati nascosti, che e' proprio quello che vogliamo confrontare.
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
        if attivazione == "tanh":
            self.f = torch.tanh
        elif attivazione == "relu":
            self.f = torch.relu
        else:
            raise ValueError("attivazione deve essere 'tanh' o 'relu'")

        # Va impostata PRIMA di inizializzare i pesi: e' lei a decidere
        # quale inizializzazione usare.
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


# ---------------------------------------------------------------------------
# INIZIALIZZAZIONE
# ---------------------------------------------------------------------------


def inizializza_pesi(modello):
    """
    Sceglie l'inizializzazione in base all'attivazione del modello.

    Le due cose non sono indipendenti: una sigma tarata sulla tanh e'
    sbagliata per la ReLU, e viceversa. Per questo l'utente sceglie solo
    l'attivazione e l'inizializzazione viene di conseguenza.
    """
    if modello.attivazione == "tanh":
        _init_paper(modello)
    else:
        _init_he(modello)

    # Strato di uscita: uguale nei due casi (vedi commento su STD_USCITA).
    nn.init.normal_(modello.uscita.weight, mean=0.0, std=STD_USCITA)
    nn.init.zeros_(modello.uscita.bias)


def _init_paper(modello):
    """
    Inizializzazione descritta nei Methods del paper (versione 2014).

    I pesi partono da una gaussiana di media zero, con ampiezza fissa
    decisa a mano:
        primo strato    -> 0.1
        strati interni  -> 0.05

    Con la tanh l'ampiezza iniziale conta molto: pesi troppo grandi mandano
    le attivazioni nella zona piatta della tanh, dove la derivata e' quasi
    nulla e l'addestramento si blocca; pesi troppo piccoli fanno spegnere
    il segnale strato dopo strato.
    """
    for i, strato in enumerate(modello.nascosti):
        std = STD_PRIMO if i == 0 else STD_INTERMEDI
        nn.init.normal_(strato.weight, mean=0.0, std=std)
        nn.init.zeros_(strato.bias)


def _init_he(modello):
    """
    Inizializzazione di He (2015), quella giusta per la ReLU.

    Invece di scegliere la sigma a mano, la si calcola:

        sigma = sqrt(2 / n_ingressi)

    L'idea: la ReLU azzera meta' dei valori, quindi dimezza la varianza
    del segnale a ogni strato. Il fattore 2 compensa esattamente questa
    perdita, cosi' la varianza resta circa costante mentre si scende nella
    rete e il gradiente non si spegne. Il numero di ingressi compare
    perche' ogni unita' somma n_ingressi contributi indipendenti.

    Nota: qui la sigma dipende dallo strato in modo automatico, mentre nel
    2014 era un numero fissato dagli autori. E' proprio uno dei punti del
    confronto fra i due stack.
    """
    for strato in modello.nascosti:
        nn.init.kaiming_normal_(strato.weight,
                                mode="fan_in",
                                nonlinearity="relu")
        nn.init.zeros_(strato.bias)


# ---------------------------------------------------------------------------
# COSTRUTTORI
# ---------------------------------------------------------------------------


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

    print()
    print("Confronto fra le due inizializzazioni (sigma dei pesi)")
    print("  strato        tanh (paper)   relu (He)")
    m2014 = rete_profonda(n_input=21, attivazione="tanh")
    mnuovo = rete_profonda(n_input=21, attivazione="relu")
    for i in range(N_STRATI_PROFONDA):
        s_a = m2014.nascosti[i].weight.std().item()
        s_b = mnuovo.nascosti[i].weight.std().item()
        print(f"  nascosto {i + 1}        {s_a:.4f}         {s_b:.4f}")

    print()
    print("Ampiezza delle attivazioni all'ultimo strato nascosto")
    x = torch.randn(2000, 21)
    for nome, mod in (("tanh", m2014), ("relu", mnuovo)):
        with torch.no_grad():
            _, att = mod(x, restituisci_attivazioni=True)
        ampiezze = "  ".join(f"{a.std().item():.3f}" for a in att)
        print(f"  {nome:5s}: {ampiezze}")