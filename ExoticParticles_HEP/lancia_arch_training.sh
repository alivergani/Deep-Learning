#!/bin/bash
#
# Lancia i tre training con l'architettura ottimizzata (7 strati x 700
# unita', lr 2.46e-3, wd 9.85e-3) sui tre feature set, in parallelo.
#
# Uso:
#     ./lancia_arch_training.sh
#
# Non serve essere gia' dentro l'ambiente virtuale ne' in tmux: lo script
# fa tutto da solo, quindi si puo' chiudere il terminale dopo.
#
# Per vedere come procede:
#     tail -f logs/arch_*.log
#
# Per fermare tutto:
#     pkill -f esperimenti.py

set -e   # si ferma al primo errore invece di proseguire a vuoto

PROGETTO=~/Desktop/Code/ali/Deep-Learning/ExoticParticles_HEP
cd "$PROGETTO"

echo "=== Preparazione ==="
source venv/bin/activate
mkdir -p logs

# --- controlli prima di lanciare -------------------------------------------
# Meglio scoprirlo adesso che domattina davanti a tre log sbagliati.

if ! grep -q "ARCH_OTTIMIZZATA" src/esperimenti.py; then
    echo "ERRORE: src/esperimenti.py non ha la modalita' 'arch'."
    echo "Serve un git pull, oppure copiare il file aggiornato da Dirac."
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

# --- lancio ----------------------------------------------------------------
# I tre girano in parallelo dentro un'unica sessione tmux, cosi'
# sopravvivono alla chiusura del terminale.
# La sessione si chiama "archtrain"; per rientrarci: tmux attach -t archtrain

echo "=== Lancio dei tre training (sessione tmux: archtrain) ==="

tmux new-session -d -s archtrain "cd '$PROGETTO' && source venv/bin/activate && \
    python -u src/esperimenti.py deep low moderno arch 0      > logs/arch_low.log      2>&1 & \
    python -u src/esperimenti.py deep high moderno arch 0     > logs/arch_high.log     2>&1 & \
    python -u src/esperimenti.py deep complete moderno arch 0 > logs/arch_complete.log 2>&1 & \
    wait"

# Il caricamento di 10 milioni di eventi richiede un po': aspettiamo prima
# di controllare, altrimenti i log sono ancora vuoti.
echo "Attendo l'avvio (90 secondi)..."
sleep 90

echo
echo "=== Controllo ==="
for f in logs/arch_low.log logs/arch_high.log logs/arch_complete.log; do
    echo "--- $f"
    grep -E "Architettura|lr / wd|Addestramento su|Risultati in" "$f" || \
        echo "  (ancora nessuna riga: potrebbe servire piu' tempo)"
done

echo
echo "=== Fatto ==="
echo "Ogni log deve riportare '7 strati x 700 unita'' e 'Addestramento su cuda'."
echo "Per seguire:  tail -f $PROGETTO/logs/arch_*.log"
echo "Per fermare:  pkill -f esperimenti.py"
