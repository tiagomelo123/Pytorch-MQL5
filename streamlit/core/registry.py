"""Registro local (JSON) dos modelos treinados, para monitorar e comparar runs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import pandas as pd

from core.config import REGISTRY_PATH, RUNS_DIR


def _load_raw() -> list[dict]:
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw(runs: list[dict]) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)


def new_run_id(symbol: str, arquitetura: str) -> str:
    """Gera um identificador único e legível para um novo run de treino."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{arquitetura}_{ts}_{uuid.uuid4().hex[:6]}"


def run_dir(run_id: str) -> str:
    """Diretório local onde os artefatos de um run são salvos."""
    path = os.path.join(RUNS_DIR, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def add_run(run: dict) -> None:
    """Adiciona (ou substitui, por ``run_id``) um registro de run treinado."""
    runs = _load_raw()
    runs = [r for r in runs if r.get("run_id") != run.get("run_id")]
    runs.append(run)
    _save_raw(runs)


def list_runs() -> pd.DataFrame:
    """Retorna todos os runs registrados como DataFrame, mais recentes primeiro."""
    runs = _load_raw()
    if not runs:
        return pd.DataFrame(
            columns=[
                "run_id", "criado_em", "symbol", "timeframe", "tarefa",
                "arquitetura", "metrica_principal", "valor_metrica", "melhor_epoca",
                "enviado_firebase",
            ]
        )
    df = pd.DataFrame(runs)
    if "criado_em" in df.columns:
        df = df.sort_values("criado_em", ascending=False).reset_index(drop=True)
    return df


def get_run(run_id: str) -> dict | None:
    """Busca um run específico pelo ``run_id``."""
    for r in _load_raw():
        if r.get("run_id") == run_id:
            return r
    return None


def delete_run(run_id: str) -> None:
    """Remove um run do registro (não apaga arquivos locais nem do Firebase)."""
    runs = [r for r in _load_raw() if r.get("run_id") != run_id]
    _save_raw(runs)


def mark_uploaded(run_id: str, firebase_paths: dict) -> None:
    """Marca um run como enviado ao Firebase, guardando os caminhos remotos."""
    runs = _load_raw()
    for r in runs:
        if r.get("run_id") == run_id:
            r["enviado_firebase"] = True
            r["firebase_paths"] = firebase_paths
    _save_raw(runs)
