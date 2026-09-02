#!/bin/bash
#
# Lancia l'ottimizzazione dell'architettura sulla macchina con GPU.
#
# Uso:
#     ./lancia_arch.sh
#
# Non serve essere gia' dentro l'ambiente virtuale ne' in tmux: lo script
# fa tutto da solo e si stacca, quindi si puo' chiudere il terminale.
#
# Per vedere come procede:
#     tail -f logs/hyperopt_arch.log
#
# Per fermare tutto:
#     pkill -f ottimizza.py

set -e   # si ferma al primo errore invece di proseguire a vuoto

PROGETTO=~/Desktop/Code/ali/Deep-Learning/ExoticParticles_HEP
cd "$PROGETTO"

echo "=== Preparazione ==="
source venv/bin/activate
mkdir -p logs

# --- controlli prima di lanciare -------------------------------------------
# Meglio scoprire adesso che il file e' quello vecchio, invece che domattina
# davanti a un log senza architettura.

if ! grep -q "CERCA_ARCHITETTURA" src/ottimizza.py; then
    echo "ERRORE: src/ottimizza.py non contiene la ricerca sull'architettura."
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
# tmux serve perche' il processo sopravviva alla chiusura del terminale.
# La sessione si chiama "arch"; per rientrarci: tmux attach -t arch

echo "=== Lancio (sessione tmux: arch) ==="
tmux new-session -d -s arch \
    "cd '$PROGETTO' && source venv/bin/activate && \
     python -u src/ottimizza.py arch 20 > logs/hyperopt_arch.log 2>&1"

# Diamo qualche secondo al caricamento dei dati, poi mostriamo l'inizio del
# log: serve a verificare subito che sia partito con la configurazione giusta.
echo "Attendo l'avvio..."
sleep 45

echo
echo "=== Prime righe del log ==="
head -20 logs/hyperopt_arch.log

echo
echo "=== Fatto ==="
echo "Sessione tmux 'arch' avviata. Il terminale si puo' chiudere."
echo "Per seguire:  tail -f $PROGETTO/logs/hyperopt_arch.log"
echo "Per fermare:  pkill -f ottimizza.py"