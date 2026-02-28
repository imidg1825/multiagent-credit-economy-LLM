#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from typing import Dict, List

import matplotlib.pyplot as plt


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def to_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def open_in_windows(path: str) -> None:
    """
    Надёжное открытие файла из WSL в Windows:
    explorer.exe <path>
    """
    try:
        subprocess.run(["explorer.exe", path], check=False)
    except Exception:
        # если вдруг explorer.exe недоступен, просто молча не открываем
        pass


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) >= 2 else "run_log.csv"

    os.makedirs("plots", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_png = sys.argv[2] if len(sys.argv) >= 3 else f"plots/economy_{ts}.png"

    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit(f"CSV is empty: {csv_path}")

    # агрегируем по tick: берём последнюю строку каждого тика
    per_tick: Dict[int, Dict[str, str]] = {}
    for r in rows:
        t = to_int(r.get("tick", "0"))
        if t > 0:
            per_tick[t] = r

    ticks = sorted(per_tick.keys())
    gini = [to_float(per_tick[t].get("gini_budget", "0")) for t in ticks]
    deals = [to_int(per_tick[t].get("credit_deals", "0")) for t in ticks]
    open_loans = [to_int(per_tick[t].get("open_loans", "0")) for t in ticks]
    defaults = [to_int(per_tick[t].get("defaults_total", "0")) for t in ticks]
    latency = [to_float(per_tick[t].get("llm_latency_s", "0")) for t in ticks]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(ticks, gini, marker="o", linewidth=2)
    axes[0].set_title("Gini (budget inequality) over time")
    axes[0].set_ylabel("gini_budget")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ticks, deals, label="credit_deals", linewidth=2)
    axes[1].plot(ticks, open_loans, label="open_loans", linewidth=2)
    axes[1].plot(ticks, defaults, label="defaults_total", linewidth=2)
    axes[1].set_title("Credit market")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ticks, latency, marker=".", linewidth=2)
    axes[2].set_title("LLM latency per tick")
    axes[2].set_ylabel("seconds")
    axes[2].set_xlabel("tick")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    print("[OK] saved:", out_png)
    print("[INFO] opening:", out_png)
    open_in_windows(out_png)


if __name__ == "__main__":
    main()
