#!/bin/bash
#
# Lancia i tre training dello stack moderno (deep, architettura del paper
# 4 strati x 300 unita') sui tre feature set, in parallelo.
#
# Usa gli iperparametri trovati dalla ricerca TPE su 3 milioni di eventi:
#     lr = 7.251109039499151e-4
#     wd = 1.0725399645937344e-5
# che sono costanti dentro src/train.py, non argomenti da riga di comando.
#
# Uso:
#     ./lancia_moderno.sh
#
# Non serve essere gia' dentro l'ambiente virtuale ne' in tmux: lo script
# fa tutto da solo, quindi si puo' chiudere il terminale dopo.
#
# Per vedere come procede:
#     tail -f logs/moderno_*.log
#
# Per fermare tutto:
#     pkill -f esperimenti.py

set -e   # si ferma al primo errore invece di proseguire a vuoto

PROGETTO=~/Deep-Learning/ExoticParticles_HEP
cd "$PROGETTO"

echo "=== Preparazione ==="
source venv/bin/activate
mkdir -p logs

# --- controlli prima di lanciare -------------------------------------------
# Meglio scoprirlo adesso che domattina davanti a tre log sbagliati.

# Il controllo piu' importante di tutti: che train.py sia la versione con i
# nuovi iperparametri. Se qui c'e' ancora 2.1e-3, la macchina non ha fatto
# git pull e questi tre training rifarebbero esattamente l'errore che la
# ricerca su 3M serviva a correggere.
if ! grep -q "^LR_ADAMW = 7.251109039499151e-4" src/train.py; then
    echo "ERRORE: src/train.py non ha il learning rate nuovo."
    echo "Valore attualmente nel file:"
    grep -E "^(LR_ADAMW|WEIGHT_DECAY_ADAMW) *=" src/train.py
    echo
    echo "Serve un git pull."
    exit 1
fi

if ! grep -q "^WEIGHT_DECAY_ADAMW = 1.0725399645937344e-5" src/train.py; then
    echo "ERRORE: src/train.py non ha il weight decay nuovo."
    grep -E "^(LR_ADAMW|WEIGHT_DECAY_ADAMW) *=" src/train.py
    echo
    echo "Serve un git pull."
    exit 1
fi

if [ ! -d "$PROGETTO/data/processed" ]; then
    echo "ERRORE: manca $PROGETTO/data/processed"
    exit 1
fi

if nvidia-smi | grep -q "python"; then
    echo "ATTENZIONE: c'e' gia' un processo python sulla GPU."
    nvidia-smi | grep "python"
    read -p "Lancio lo stesso? [s/N] " risposta
    if [ "$risposta" != "s" ]; then
        echo "Annullato."
        exit 0
    fi
fi

# --- riepilogo, che finisce a schermo e resta nella cronologia --------------
echo
echo "=== Configurazione ==="
grep -E "^(LR_ADAMW|WEIGHT_DECAY_ADAMW|PAZIENZA_ADAMW|PLATEAU_PAZIENZA|PLATEAU_FATTORE|BATCH|MAX_EPOCHE) *=" src/train.py
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'non e un repo git')"
git status --porcelain 2>/dev/null | head
echo

# --- lancio ----------------------------------------------------------------
# I tre girano in parallelo dentro un'unica sessione tmux, cosi'
# sopravvivono alla chiusura del terminale.
# La sessione si chiama "moderno"; per rientrarci: tmux attach -t moderno
#
# Nota sulla velocita': con batch da 100 il collo di bottiglia sono i lanci
# dei kernel, non il calcolo, quindi tre processi sulla stessa GPU rendono
# circa 2-2.5 volte il seriale, non 3.

echo "=== Lancio dei tre training (sessione tmux: moderno) ==="

tmux new-session -d -s moderno "cd '$PROGETTO' && source venv/bin/activate && \
    python -u src/esperimenti.py deep low moderno 0      > logs/moderno_low.log      2>&1 & \
    python -u src/esperimenti.py deep high moderno 0     > logs/moderno_high.log     2>&1 & \
    python -u src/esperimenti.py deep complete moderno 0 > logs/moderno_complete.log 2>&1 & \
    wait"

# Il caricamento di 10 milioni di eventi richiede un po': aspettiamo prima
# di controllare, altrimenti i log sono ancora vuoti.
echo "Attendo l'avvio (120 secondi)..."
sleep 120

echo
echo "=== Controllo ==="
for f in logs/moderno_low.log logs/moderno_high.log logs/moderno_complete.log; do
    echo "--- $f"
    grep -E "feature set|Addestramento su|Ottimizzatore|lr iniziale|aggiornamenti per epoca" "$f" || \
        echo "  (ancora nessuna riga: potrebbe servire piu' tempo)"
done

echo
echo "=== Fatto ==="
echo "Ogni log deve riportare 'Addestramento su cuda' e 'lr iniziale 0.000725...'."
echo "Per seguire:  tail -f $PROGETTO/logs/moderno_*.log"
echo "Per fermare:  pkill -f esperimenti.py"
echo
echo "Domattina, per i tempi:  grep -H 'Tempo totale' $PROGETTO/logs/moderno_*.log"