#!/usr/bin/env python3
"""Converte i file grezzi di Zenodo in tabelle pronte all'analisi.

Da lanciare **una volta sola** dalla radice del progetto:

    python scripts/prepare_data.py
    python scripts/prepare_data.py --n-events 50000     # prova veloce
    python scripts/prepare_data.py --skip-3prong

Legge:   data/raw/*.features.h5
Scrive:  data/processed/features.parquet
         data/processed/features_3prong.parquet
         results/runs/prepare_data.json   (report dei controlli)

Da qui in poi i notebook caricano il Parquet in un paio di secondi e non
toccano piu' ne' l'HDF5 ne' PyTables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# permette `python scripts/prepare_data.py` dalla radice del progetto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, processed_path, raw_path  # noqa: E402
from src.features import build_features, correlation_with_mjj, sanity_report  # noqa: E402
from src.io import LABEL_COL, load_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="file YAML alternativo (default: config/default.yaml)")
    p.add_argument("--n-events", type=int, default=None,
                   help="processa solo le prime N righe, per test rapidi")
    p.add_argument("--skip-3prong", action="store_true",
                   help="non processare il file di segnale 3-prong")
    p.add_argument("--force", action="store_true",
                   help="rigenera anche se l'output esiste gia'")
    return p.parse_args()


def process_one(src_path: Path, dst_path: Path, has_label: bool,
                n_events: int | None, force: bool) -> dict | None:
    """Legge un file di feature, costruisce le variabili, salva in Parquet."""
    if dst_path.exists() and not force:
        print(f"  [skip] {dst_path.name} esiste gia' (usa --force per rigenerare)")
        return None

    print(f"  Leggo {src_path.name} ...", flush=True)
    t0 = time.time()
    raw = load_features(src_path, has_label=has_label, n_events=n_events)
    print(f"    {len(raw):,} eventi, {raw.shape[1]} colonne "
          f"({time.time() - t0:.1f} s)")

    print("  Costruisco m_JJ e le variabili ausiliarie ...", flush=True)
    df = build_features(raw)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)
    print(f"  Scritto {dst_path} ({dst_path.stat().st_size / 1e6:.1f} MB)")

    report = sanity_report(df)

    # La correlazione va misurata sul solo fondo: e' il fondo che non deve
    # essere scolpito dal taglio.
    bkg = df[df[LABEL_COL] == 0]
    if len(bkg) > 0:
        report["corr_with_mjj_background"] = {
            k: round(float(v), 4) for k, v in correlation_with_mjj(bkg).items()
        }
    return report


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 68)
    print("Preparazione dati - LHC Olympics 2020 R&D")
    print("=" * 68)

    reports: dict = {}

    print("\n[1/2] Dataset principale (1M QCD + 100k segnale 2-prong)")
    try:
        reports["main"] = process_one(
            raw_path(cfg, "features_main"),
            processed_path(cfg, "processed_main"),
            has_label=True,
            n_events=args.n_events,
            force=args.force,
        )
    except FileNotFoundError as exc:
        print(f"  ERRORE: {exc}")
        return 1

    if not args.skip_3prong:
        print("\n[2/2] Segnale alternativo 3-prong (X,Y -> qqq)")
        try:
            reports["signal_3prong"] = process_one(
                raw_path(cfg, "features_signal_3prong"),
                processed_path(cfg, "processed_signal_3prong"),
                has_label=False,
                n_events=args.n_events,
                force=args.force,
            )
        except FileNotFoundError as exc:
            print(f"  [avviso] saltato: {exc}")

    # ---- report dei controlli -------------------------------------------
    main_report = reports.get("main")
    if main_report:
        print("\n" + "-" * 68)
        print("Controlli di sanita'")
        print("-" * 68)
        for key, value in main_report.items():
            if key != "corr_with_mjj_background":
                print(f"  {key:.<38} {value}")

        corr = main_report.get("corr_with_mjj_background", {})
        if corr:
            print("\n  Correlazione con m_JJ (solo fondo):")
            for name, value in corr.items():
                flag = "  <-- alta!" if abs(value) > 0.2 else ""
                print(f"    {name:.<34} {value:+.4f}{flag}")

        if not main_report["mass_ordering_ok"]:
            print("\n  ERRORE: ordinamento in massa violato.")
            return 1
        if main_report["n_nan"] > 0:
            print(f"\n  ERRORE: {main_report['n_nan']} NaN nel dataset.")
            return 1

    runs_dir = cfg["paths"]["results"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_json = runs_dir / "prepare_data.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"config": str(args.config or "default.yaml"),
                   "n_events": args.n_events,
                   "reports": reports}, fh, indent=2)
    print(f"\nReport salvato in {out_json}")
    print("\nFatto. I notebook possono partire da data/processed/*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
