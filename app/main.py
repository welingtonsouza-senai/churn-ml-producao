from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import ClienteInput, PredicaoOutput
from src.features import preparar_features

modelo_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    artefato = joblib.load("models/churn_pipeline_v1.joblib")
    modelo_cache.update(artefato)
    print(f"Modelo carregado: {artefato['modelo_nome']} (v{artefato['versao']})")
    yield
    modelo_cache.clear()

app = FastAPI(
    title="API de Predição de Churn - Telecom",
    description="Serviço de inferência em tempo real com Pipeline Scikit-Learn e Threshold Otimizado.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "modelo": modelo_cache.get("modelo_nome"),
        "versao": modelo_cache.get("versao"),
        "threshold": modelo_cache.get("threshold")
    }

@app.post("/predict", response_model=PredicaoOutput)
def prever_churn(cliente: ClienteInput):
    if "pipeline" not in modelo_cache:
        raise HTTPException(status_code=503, detail="Modelo não inicializado.")

    df_entrada = pd.DataFrame([cliente.model_dump()])
    df_features = preparar_features(df_entrada)

    pipeline = modelo_cache["pipeline"]
    threshold = modelo_cache["threshold"]

    proba = float(pipeline.predict_proba(df_features)[:, 1][0])
    risco_alto = proba >= threshold

    acao = (
        "ACIONAR RETENÇÃO IMEDIATA: Oferecer bônus de fidelidade (R$ 50) + migração para contrato anual."
        if risco_alto else
        "MANTER FLUXO NORMAL: Cliente dentro da faixa segura de retenção."
    )

    return PredicaoOutput(
        probabilidade_churn=round(proba, 4),
        risco_alto=risco_alto,
        threshold_aplicado=threshold,
        acao_recomendada=acao,
        versao_modelo=modelo_cache["versao"]
    )