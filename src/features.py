import numpy as np
import pandas as pd

COLUNAS_LEAKAGE = ["motivo_ultimo_ticket"]

SERVICOS_ADICIONAIS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

def limpar_valor_monetario(serie: pd.Series) -> pd.Series:
    limpa = (
        serie.astype(str)
        .str.replace("R$", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(limpa, errors="coerce")

def integrar_fontes_brutas(df_crm: pd.DataFrame, df_erp: pd.DataFrame) -> pd.DataFrame:
    erp_dedup = df_erp.drop_duplicates(subset=["customerID"], keep="last").copy()
    return pd.merge(df_crm, erp_dedup, on="customerID", how="inner")

def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    dados = df.copy()

    for col in ["customerID"] + COLUNAS_LEAKAGE:
        if col in dados.columns:
            dados = dados.drop(columns=[col])

    dados["MonthlyCharges"] = limpar_valor_monetario(dados["MonthlyCharges"])
    dados["TotalCharges"] = limpar_valor_monetario(dados["TotalCharges"])
    dados["tenure"] = pd.to_numeric(dados["tenure"], errors="coerce").fillna(0)
    dados["SeniorCitizen"] = dados["SeniorCitizen"].astype(str)

    ticket_historico = dados["TotalCharges"] / (dados["tenure"] + 1.0)
    dados["razao_fatura_atual_vs_historica"] = np.where(
        ticket_historico > 0,
        dados["MonthlyCharges"] / ticket_historico,
        1.0
    )

    dados["qtd_servicos_ativos"] = (dados[SERVICOS_ADICIONAIS] == "Yes").sum(axis=1)

    dados["perfil_volatil"] = (
        (dados["Contract"] == "Month-to-month") &
        (dados["PaymentMethod"] == "Electronic check")
    ).astype(int)

    return dados