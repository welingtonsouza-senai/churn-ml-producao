import os
import numpy as np
import pandas as pd

def construir_cenario_legado():
    os.makedirs("data/raw", exist_ok=True)
    print("Baixando base real Telco Customer Churn...")

    url_github = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        df = pd.read_csv(url_github)
    except Exception:
        from sklearn.datasets import fetch_openml
        dataset = fetch_openml(data_id=42178, as_frame=True, parser="auto")
        df = dataset.frame.copy()
        if "Churn" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "Churn"})
        if "customerID" not in df.columns:
            df["customerID"] = [f"CUST-{10000 + i}" for i in range(len(df))]

    np.random.seed(42)

    cols_crm = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Churn"
    ]
    df_crm = df[cols_crm].copy()

    cols_erp = ["customerID", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
    df_erp = df[cols_erp].copy()

    # Sujeira 1: Formatação de moeda brasileira em ~15% das linhas
    df_erp["MonthlyCharges"] = df_erp["MonthlyCharges"].astype(str)
    mask_moeda = np.random.rand(len(df_erp)) < 0.15
    df_erp.loc[mask_moeda, "MonthlyCharges"] = (
        "R$ " + df_erp.loc[mask_moeda, "MonthlyCharges"].str.replace(".", ",", regex=False)
    )

    # Sujeira 2: Coluna de Data Leakage (gerada APÓS o pedido de cancelamento)
    motivos = np.where(
        (df["Churn"] == "Yes") & (np.random.rand(len(df)) < 0.65),
        "Solicitacao_Cancelamento_Definitivo",
        np.random.choice(["Duvida_Fatura", "Suporte_Tecnico", "Sem_Chamados"], size=len(df))
    )
    df_erp["motivo_ultimo_ticket"] = motivos

    # Sujeira 3: 50 registros duplicados no extrato do ERP
    duplicados = df_erp.sample(n=50, random_state=42)
    df_erp = pd.concat([df_erp, duplicados], ignore_index=True).sample(frac=1, random_state=42)

    df_crm.to_csv("data/raw/crm_clientes.csv", index=False)
    df_erp.to_csv("data/raw/erp_faturamento.csv", index=False)
    print('Tabelas Raw Carregadas')
    
if __name__ == "__main__":
    construir_cenario_legado()