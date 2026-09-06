#!/usr/bin/env bash
#
# Lancia la ricerca degli iperparametri dello stack moderno.
#
# USO
#   ./esegui_ricerca.sh                 # low, 25 tentativi (caso mio)
#   ./esegui_ricerca.sh high            # high, 25 tentativi
#   ./esegui_ricerca.sh low 10          # low, 10 tentativi (prova veloce)
#   ./esegui_ricerca.sh high arch 20    # cerca anche l'architettura
#
# L'ordine degli argomenti non conta: ottimizza.py li riconosce dal loro
# contenuto (un feature set, la parola "arch", un numero).
#
# Va lanciato dentro tmux, perche' dura ore:
#   tmux new -s ricerca
#   ./esegui_ricerca.sh low
#   Ctrl-b d                            per staccarsi
#   tmux attach -t ricerca              per rientrare

# set -e     ferma tutto al primo comando che fallisce
# set -u     errore se si usa una variabile non definita (tipico refuso)
# set -o pipefail   se python muore dentro una pipe con tee, se ne accorge
set -euo pipefail


# --- da controllare la prima volta -----------------------------------------
# Il progetto e' la cartella dove sta questo script, qualunque essa sia:
# cosi' funziona sia su Dirac che sulla macchina con la GPU senza modifiche.
PROGETTO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Percorso del venv. Se il tuo si chiama diversamente, cambia solo questa riga.
VENV="$PROGETTO/venv"
# ---------------------------------------------------------------------------

# Gli argomenti vengono passati tutti a ottimizza.py cosi' come sono: quel
# file li riconosce per quello che sono e non per la posizione, quindi
# "high arch 20" e "20 arch high" fanno la stessa cosa. Qui servono solo a
# costruire il nome del log.
ETICHETTA="${*:-low}"          # tutti gli argomenti, o "low" se non ce ne sono
ETICHETTA="${ETICHETTA// /_}"  # gli spazi diventano underscore

DATA=$(date +%Y%m%d_%H%M)
CARTELLA_LOG="$PROGETTO/logs"
LOG="$CARTELLA_LOG/ricerca_${ETICHETTA}_${DATA}.txt"

mkdir -p "$CARTELLA_LOG"


# --- controlli prima di partire --------------------------------------------
# Una ricerca dura ore: meglio fallire adesso che fra dieci minuti.

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERRORE: nessun venv in $VENV"
    echo "Crealo con:  python3 -m venv $VENV"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c "import torch, hyperopt" 2>/dev/null; then
    echo "ERRORE: torch o hyperopt non installati nel venv."
    echo "Installali con:  pip install torch hyperopt"
    exit 1
fi

if [ ! -f "$PROGETTO/src/ottimizza.py" ]; then
    echo "ERRORE: non trovo $PROGETTO/src/ottimizza.py"
    echo "Questo script va tenuto nella cartella principale del progetto."
    exit 1
fi

# I dati si caricano dopo qualche secondo di import: senza questo controllo
# l'errore arriverebbe comunque, ma dopo aver gia' occupato la GPU.
if [ ! -d "$PROGETTO/data/processed" ]; then
    echo "ERRORE: manca $PROGETTO/data/processed"
    echo "Lancia prima prepare_data.py."
    exit 1
fi

# Il controllo vero e' dentro ottimizza.py, che sa anche chiedere conferma.
# Qui si stampa solo lo stato della GPU, che finisce nel log: all'orale, se
# ti chiedono su che macchina hai girato, la risposta e' nel file.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used \
               --format=csv,noheader
else
    echo "ATTENZIONE: nvidia-smi non trovato."
fi


# --- lancio -----------------------------------------------------------------
echo "Progetto : $PROGETTO"
echo "Log      : $LOG"
echo

cd "$PROGETTO"

# python -u disattiva il buffering: senza, l'output resterebbe fermo nel
# buffer per minuti e il log sembrerebbe bloccato anche a training in corso.
# tee scrive contemporaneamente a schermo e su file.
python -u src/ottimizza.py "$@" 2>&1 | tee "$LOG"

echo
echo "Finito. Log in $LOG"
echo "Risultati in $PROGETTO/results_small/"