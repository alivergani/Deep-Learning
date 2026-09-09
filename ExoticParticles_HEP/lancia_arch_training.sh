#!/bin/bash
#
# Lancia i tre training con l'architettura ottimizzata (7 strati x 400
# unita', lr 8.04e-4, wd 9.58e-3) sui tre feature set, in parallelo.
#
# I valori vengono dalla ricerca TPE congiunta su quattro parametri
# (profondita', larghezza, lr, weight decay), 25 tentativi su 3 milioni
# di eventi. Sono costanti dentro src/esperimenti.py, nel blocco
# "if ARCH_OTTIMIZZATA:", non argomenti da riga di comando.
#
# Uso:
#     ./lancia_arch_training.sh            # aspetta le 03:00 e poi lancia
#     ./lancia_arch_training.sh 02:30      # aspetta le 02:30
#     ./lancia_arch_training.sh adesso     # lancia subito
#
# I controlli girano SUBITO, prima dell'attesa: se i valori nel file sono
# ancora quelli vecchi lo scopri stasera, non domattina.
#
# Va lanciato dentro tmux, altrimenti chiudendo il terminale muore l'attesa:
#     tmux new -s notte_arch
#     ./lancia_arch_training.sh
#     Ctrl+b, poi d
#
# Per vedere come procede:
#     tail -f logs/arch_*.log
#
# Per fermare tutto:
#     pkill -f esperimenti.py

set -e   # si ferma al primo errore invece di proseguire a vuoto

PROGETTO=~/Deep-Learning/ExoticParticles_HEP
cd "$PROGETTO"

ORARIO="${1:-03:00}"

echo "=== Preparazione ==="
source venv/bin/activate
mkdir -p logs

# --- controlli prima di lanciare -------------------------------------------
# Meglio scoprirlo adesso che domattina davanti a tre log sbagliati.

if ! grep -q "ARCH_OTTIMIZZATA" src/esperimenti.py; then
    echo "ERRORE: src/esperimenti.py non ha la modalita' 'arch'."
    echo "Serve un git pull."
    exit 1
fi

# I quattro valori nuovi. Se qui c'e' ancora 700 unita' o lr 2.46e-3, la
# macchina non ha fatto git pull e questi tre training userebbero i valori
# della vecchia ricerca su 1M invece di quelli su 3M.
for atteso in "N_STRATI = 7" "N_UNITA = 400" "LR = 8.04e-4" "WEIGHT_DECAY = 9.58e-3"; do
    if ! grep -q "$atteso" src/esperimenti.py; then
        echo "ERRORE: non trovo '$atteso' in src/esperimenti.py."
        echo "Valori attualmente nel blocco ARCH_OTTIMIZZATA:"
        grep -A6 "^if ARCH_OTTIMIZZATA:" src/esperimenti.py
        echo
        echo "Serve un git pull, o i valori non sono stati aggiornati."
        exit 1
    fi
done

if [ ! -d "$PROGETTO/data/processed" ]; then
    echo "ERRORE: manca $PROGETTO/data/processed"
    exit 1
fi

# I risultati vanno in results/moderno_arch/. Se ci sono gia' dei file per
# il seme 0, esperimenti.py li salta stampando "gia' completato" e domattina
# trovi tre log che non hanno addestrato niente.
if ls results/moderno_arch/deep_*_seme0_modello.pt >/dev/null 2>&1; then
    echo "ERRORE: ci sono gia' risultati in results/moderno_arch/ per il seme 0:"
    ls results/moderno_arch/
    echo
    echo "Spostali via prima di lanciare, altrimenti i training vengono saltati."
    exit 1
fi

# Qui la GPU occupata e' solo un avviso: un processo che gira adesso
# potrebbe benissimo essere finito per le 3. Il controllo che conta lo
# rifacciamo dopo l'attesa.
if nvidia-smi | grep -q "python"; then
    echo
    echo "ATTENZIONE: adesso c'e' un processo python sulla GPU."
    nvidia-smi | grep "python"
    echo "Ricontrollo alle $ORARIO."
fi

# --- riepilogo, che finisce a schermo e resta nella cronologia --------------
echo
echo "=== Configurazione ==="
grep -A6 "^if ARCH_OTTIMIZZATA:" src/esperimenti.py
grep -E "^(PAZIENZA_ADAMW|PLATEAU_PAZIENZA|PLATEAU_FATTORE) *=" src/train.py
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'non e un repo git')"
git status --porcelain 2>/dev/null | head
echo

# --- attesa fino all'orario indicato ---------------------------------------
if [ "$ORARIO" = "adesso" ]; then
    echo "=== Nessuna attesa, lancio subito ==="
else
    BERSAGLIO=$(date -d "$ORARIO" +%s)
    ADESSO=$(date +%s)
    # Se l'orario di oggi e' gia' passato, si intende domani.
    if [ "$BERSAGLIO" -le "$ADESSO" ]; then
        BERSAGLIO=$(date -d "tomorrow $ORARIO" +%s)
    fi
    ATTESA=$(( BERSAGLIO - ADESSO ))

    echo "=== Attesa ==="
    echo "Parto alle $(date -d "@$BERSAGLIO" '+%H:%M di %A %d/%m')"
    echo "cioe' fra $(( ATTESA / 3600 ))h $(( (ATTESA % 3600) / 60 ))m."
    echo "Se la riga sopra e' giusta: Ctrl+b, poi d, e buonanotte."
    sleep "$ATTESA"
    echo
    echo "Sveglia: sono le $(date '+%H:%M')."
fi

# Ricontrollo la GPU adesso che siamo all'orario buono.
if nvidia-smi | grep -q "python"; then
    echo "ERRORE: c'e' ancora un processo python sulla GPU."
    nvidia-smi | grep "python"
    echo "Non lancio, meglio non sovrapporsi."
    exit 1
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
echo "Attendo l'avvio (120 secondi)..."
sleep 120

echo
echo "=== Controllo ==="
for f in logs/arch_low.log logs/arch_high.log logs/arch_complete.log; do
    echo "--- $f"
    grep -E "Architettura|lr / wd|Addestramento su|Risultati in" "$f" || \
        echo "  (ancora nessuna riga: potrebbe servire piu' tempo)"
done

echo
echo "=== Fatto ==="
echo "Ogni log deve riportare '7 strati x 400 unita'' e 'Addestramento su cuda'."
echo "Per seguire:  tail -f $PROGETTO/logs/arch_*.log"
echo "Per fermare:  pkill -f esperimenti.py"
echo
echo "Domattina, per i tempi:  grep -H 'Tempo totale' $PROGETTO/logs/arch_*.log"