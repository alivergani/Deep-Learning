#!/usr/bin/env bash
#
# Lancia la ricerca degli iperparametri dello stack moderno, di default
# quella completa sull'architettura (4 parametri: lr, weight decay,
# numero di strati, unita' per strato).
#
# USO
#   ./esegui_ricerca.sh                      # low arch 25, parte alle 03:00
#   ./esegui_ricerca.sh --alle 02:30         # stesso, ma alle 02:30
#   ./esegui_ricerca.sh --adesso             # subito, senza aspettare
#   ./esegui_ricerca.sh high arch 40         # altri argomenti, sempre alle 03:00
#   ./esegui_ricerca.sh --adesso low 5       # prova veloce, subito
#
# L'ordine degli argomenti dopo --alle non conta: ottimizza.py li riconosce
# dal loro contenuto (un feature set, la parola "arch", un numero).
#
# I controlli girano SUBITO, prima dell'attesa: se manca il venv o i dati
# lo scopri stasera, non alle 3 davanti a un log vuoto.
#
# Va lanciato dentro tmux, perche' dura ore:
#   tmux new -s ricerca
#   ./esegui_ricerca.sh
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

# Orario di partenza e argomenti di default.
ORARIO="03:00"
PREDEFINITI="low arch 25"
# ---------------------------------------------------------------------------


# --- lettura degli argomenti -----------------------------------------------
# --alle e --adesso vengono tolti di mezzo qui; tutto il resto prosegue
# intatto verso ottimizza.py.
while [ $# -gt 0 ]; do
    case "$1" in
        --alle)
            ORARIO="$2"
            shift 2
            ;;
        --adesso)
            ORARIO="adesso"
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Se non resta nessun argomento, si usano i predefiniti.
if [ $# -eq 0 ]; then
    set -- $PREDEFINITI
fi

# Gli argomenti servono anche a costruire il nome del log.
ETICHETTA="$*"
ETICHETTA="${ETICHETTA// /_}"

DATA=$(date +%Y%m%d_%H%M)
CARTELLA_LOG="$PROGETTO/logs"
LOG="$CARTELLA_LOG/ricerca_${ETICHETTA}_${DATA}.txt"

mkdir -p "$CARTELLA_LOG"


# --- controlli prima di partire --------------------------------------------
# Una ricerca dura ore: meglio fallire adesso che fra dieci minuti.
# E soprattutto meglio fallire stasera che alle tre di notte.

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

# Stato della GPU: finisce nel log, cosi' all'orale, se ti chiedono su che
# macchina hai girato, la risposta e' nel file.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used \
               --format=csv,noheader
else
    echo "ATTENZIONE: nvidia-smi non trovato."
fi

# Qui la GPU occupata e' solo un avviso: un processo che gira adesso
# potrebbe benissimo essere finito per le 3. Il controllo che conta lo
# rifacciamo dopo l'attesa.
if nvidia-smi 2>/dev/null | grep -q "python"; then
    echo
    echo "ATTENZIONE: adesso c'e' un processo python sulla GPU."
    nvidia-smi | grep "python"
    echo "Ricontrollo all'orario di lancio."
fi


# --- riepilogo --------------------------------------------------------------
echo
echo "=== Configurazione ==="
echo "Progetto  : $PROGETTO"
echo "Argomenti : $*"
echo "Log       : $LOG"
echo "commit    : $(git -C "$PROGETTO" rev-parse --short HEAD 2>/dev/null || echo 'non e un repo git')"
git -C "$PROGETTO" status --porcelain 2>/dev/null | head
echo


# --- attesa fino all'orario indicato ---------------------------------------
if [ "$ORARIO" = "adesso" ]; then
    echo "=== Nessuna attesa, parto subito ==="
else
    BERSAGLIO=$(date -d "$ORARIO" +%s)
    ADESSO=$(date +%s)
    # Se l'orario di oggi e' gia' passato, si intende quello di domani.
    # Cosi' funziona uguale che lo lanci alle 23 o alle 00:30.
    if [ "$BERSAGLIO" -le "$ADESSO" ]; then
        BERSAGLIO=$(date -d "tomorrow $ORARIO" +%s)
    fi
    ATTESA=$(( BERSAGLIO - ADESSO ))

    echo "=== Attesa ==="
    echo "Parto alle $(date -d "@$BERSAGLIO" '+%H:%M di %A %d/%m')"
    echo "cioe' fra $(( ATTESA / 3600 ))h $(( (ATTESA % 3600) / 60 ))m."
    echo "Se la riga sopra e' giusta: Ctrl-b, poi d, e buonanotte."
    sleep "$ATTESA"
    echo
    echo "Sveglia: sono le $(date '+%H:%M')."
fi

# Ricontrollo la GPU adesso che siamo all'orario buono. Se c'e' ancora
# qualcosa, mi fermo: sovrapporre una ricerca di ore a un processo ignoto
# peggiorerebbe solo le cose.
if nvidia-smi 2>/dev/null | grep -q "python"; then
    echo "ERRORE: c'e' ancora un processo python sulla GPU."
    nvidia-smi | grep "python"
    echo "Non lancio."
    exit 1
fi


# --- lancio -----------------------------------------------------------------
cd "$PROGETTO"

echo "=== Lancio: ottimizza.py $* ==="
echo

# python -u disattiva il buffering: senza, l'output resterebbe fermo nel
# buffer per minuti e il log sembrerebbe bloccato anche a training in corso.
# tee scrive contemporaneamente a schermo e su file.
# < /dev/null: se ottimizza.py chiede conferma da tastiera, alle 3 di notte
# non c'e' nessuno a rispondere; cosi' legge subito fine-input invece di
# restare piantato per sempre.
python -u src/ottimizza.py "$@" < /dev/null 2>&1 | tee "$LOG"

echo
echo "Finito. Log in $LOG"
echo "Risultati in $PROGETTO/results_small/"