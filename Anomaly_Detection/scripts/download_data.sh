#!/usr/bin/env bash
# Scarica il dataset R&D delle LHC Olympics 2020 in data/raw/
#
#   bash scripts/download_data.sh          # solo i file essenziali (~80 MB)
#   bash scripts/download_data.sh --raw    # aggiunge i costituenti (+3 GB)
#
# Zenodo: https://zenodo.org/records/6466204   DOI 10.5281/zenodo.6466204
# Kasieczka, Nachman, Shih - licenza CC-BY-4.0

set -euo pipefail

BASE="https://zenodo.org/records/6466204/files"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$DEST"
cd "$DEST"

fetch () {
    if [ -f "$1" ]; then
        echo "  [ok]   $1 gia' presente"
    else
        echo "  [get]  $1"
        curl -L -# -o "$1" "$BASE/$1?download=1"
    fi
}

echo "Destinazione: $DEST"
echo
echo "File essenziali:"
fetch events_anomalydetection_v2.features.h5           # 74 MB - dataset di lavoro
fetch events_anomalydetection_Z_XY_qqq.features.h5     # 5 MB  - segnale 3-prong
fetch delphes_card_RnD.dat                             # config rivelatore
fetch pythia_RnD_qcd.cmnd                              # generazione fondo
fetch pythia_RnD_Z_XY_qq.cmnd                          # generazione segnale 2-prong
fetch pythia_RnD_Z_XY_qqq.cmnd                         # generazione segnale 3-prong

if [ "${1:-}" = "--raw" ]; then
    echo
    echo "Costituenti grezzi (servono solo per CNN / deep sets):"
    fetch events_anomalydetection_v2.h5                # 2.9 GB
    fetch events_anomalydetection_Z_XY_qqq.h5          # 235 MB
fi

echo
echo "Verifica dei checksum:"
if command -v md5sum >/dev/null 2>&1; then
    md5sum -c --ignore-missing checksums.md5
elif command -v md5 >/dev/null 2>&1; then          # macOS
    while read -r sum name; do
        [ -f "$name" ] || continue
        got=$(md5 -q "$name")
        [ "$got" = "$sum" ] && echo "$name: OK" || echo "$name: FALLITO"
    done < checksums.md5
else
    echo "  (nessun tool md5 disponibile, salto)"
fi

echo
echo "Fatto. Prossimo passo:  python scripts/prepare_data.py"
