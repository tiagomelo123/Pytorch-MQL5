"""Integração com o Firebase Storage para armazenar datasets e modelos.

Credenciais e bucket são lidos de ``st.secrets["firebase"]``:

    [firebase]
    credentials_path = "C:/caminho/para/service_account.json"
    storage_bucket = "seu-projeto.appspot.com"

Veja ``.streamlit/secrets.toml.example``.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

try:
    import firebase_admin
    from firebase_admin import credentials, storage
except ImportError:  # pragma: no cover
    firebase_admin = None


class FirebaseError(RuntimeError):
    """Erro de configuração ou comunicação com o Firebase."""


def is_configured() -> bool:
    """Indica se há credenciais do Firebase configuradas em ``st.secrets``.

    Quando não existe nenhum arquivo ``secrets.toml``, o Streamlit levanta
    ``StreamlitSecretNotFoundError`` (ou similar) ao tentar ler ``st.secrets``
    em vez de simplesmente retornar vazio — por isso o acesso é protegido.
    """
    if firebase_admin is None:
        return False
    try:
        return "firebase" in st.secrets
    except Exception:  # noqa: BLE001 - nenhum secrets.toml configurado ainda
        return False


@st.cache_resource(show_spinner=False)
def _get_bucket():
    """Inicializa o app do Firebase Admin (uma vez) e retorna o bucket."""
    if firebase_admin is None:
        raise FirebaseError(
            "Pacote 'firebase-admin' não instalado. Rode: pip install firebase-admin"
        )
    if not is_configured():
        raise FirebaseError(
            "Configuração do Firebase não encontrada. Crie o arquivo "
            ".streamlit/secrets.toml a partir de .streamlit/secrets.toml.example."
        )

    cfg = st.secrets["firebase"]
    if not firebase_admin._apps:
        cred = credentials.Certificate(cfg["credentials_path"])
        firebase_admin.initialize_app(cred, {"storageBucket": cfg["storage_bucket"]})
    return storage.bucket()


def upload_dataframe(df: pd.DataFrame, remote_path: str) -> str:
    """Envia um DataFrame como CSV para o Storage, sem passar por disco.

    Returns:
        Caminho remoto (blob name) do arquivo enviado.
    """
    bucket = _get_bucket()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    blob = bucket.blob(remote_path)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")
    return remote_path


def upload_file(local_path: str, remote_path: str) -> str:
    """Envia um arquivo local para o Storage.

    Returns:
        Caminho remoto (blob name) do arquivo enviado.
    """
    bucket = _get_bucket()
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)
    return remote_path


def upload_json(obj: dict, remote_path: str) -> str:
    """Envia um dicionário serializado como JSON para o Storage."""
    bucket = _get_bucket()
    blob = bucket.blob(remote_path)
    blob.upload_from_string(json.dumps(obj, indent=2), content_type="application/json")
    return remote_path


def download_to_file(remote_path: str, local_path: str) -> None:
    """Baixa um blob do Storage para um arquivo local."""
    bucket = _get_bucket()
    blob = bucket.blob(remote_path)
    blob.download_to_filename(local_path)


def download_dataframe(remote_path: str) -> pd.DataFrame:
    """Baixa um CSV do Storage diretamente para um DataFrame."""
    bucket = _get_bucket()
    blob = bucket.blob(remote_path)
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


def list_blobs(prefix: str) -> list[dict]:
    """Lista os blobs sob um prefixo, com nome, tamanho e data de atualização."""
    bucket = _get_bucket()
    blobs = bucket.list_blobs(prefix=prefix)
    return [
        {
            "nome": b.name,
            "tamanho_kb": round((b.size or 0) / 1024, 1),
            "atualizado": b.updated.strftime("%Y-%m-%d %H:%M") if b.updated else "",
        }
        for b in blobs
        if not b.name.endswith("/")
    ]


def delete_blob(remote_path: str) -> None:
    """Remove um blob do Storage."""
    bucket = _get_bucket()
    bucket.blob(remote_path).delete()
