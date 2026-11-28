import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from features import integrar_fontes_brutas, preparar_features

CUSTO_PERDA_CLIENTE = 600.0
CUSTO_ACAO_RETENCAO = 50.0

def calcular_impacto_financeiro(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    prejuizo_sem_modelo = (tp + fn) * CUSTO_PERDA_CLIENTE
    clientes_perdidos = (tp * 0.50) + fn
    custo_campanha = (tp + fp) * CUSTO_ACAO_RETENCAO
    custo_total_com_modelo = (clientes_perdidos * CUSTO_PERDA_CLIENTE) + custo_campanha

    economia_liquida = prejuizo_sem_modelo - custo_total_com_modelo
    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "economia_liquida_brl": round(economia_liquida, 2)
    }

def treinar_e_exportar():
    print("1. Carregando e integrando bases brutas (CRM + ERP)...")
    df_crm = pd.read_csv("data/raw/crm_clientes.csv")
    df_erp = pd.read_csv("data/raw/erp_faturamento.csv")
    df_bruto = integrar_fontes_brutas(df_crm, df_erp)

    y = (df_bruto["Churn"] == "Yes").astype(int)
    X_raw = df_bruto.drop(columns=["Churn"])
    X = preparar_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    colunas_numericas = X.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
    colunas_categoricas = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, colunas_numericas),
        ("cat", cat_transformer, colunas_categoricas)
    ])

    candidatos = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, random_state=42)
    }

    print("\n2. Batalha de Pipelines (Validação no Conjunto de Teste):")
    melhor_nome, melhor_auc, melhor_pipeline = "", 0.0, None

    for nome, clf in candidatos.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        probas = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probas)
        print(f" -> {nome:<22} | ROC-AUC: {auc:.4f}")

        if auc > melhor_auc:
            melhor_auc = auc
            melhor_nome = nome
            melhor_pipeline = pipe

    print(f"\n3. Otimização Financeira de Threshold ({melhor_nome}):")
    probas_teste = melhor_pipeline.predict_proba(X_test)[:, 1]
    melhor_sim = {"economia_liquida_brl": -1e9}

    for th in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        sim = calcular_impacto_financeiro(y_test.values, probas_teste, th)
        print(f" -> Corte {th:.2f} | Abordados: {sim['tp']+sim['fp']:<4} | Economia Líquida: R$ {sim['economia_liquida_brl']:,.2f}")
        if sim["economia_liquida_brl"] > melhor_sim["economia_liquida_brl"]:
            melhor_sim = sim

    threshold_otimo = melhor_sim["threshold"]
    preds_finais = (probas_teste >= threshold_otimo).astype(int)
    print(f"\nRelatório Final com Threshold Ótimo ({threshold_otimo}):")
    print(classification_report(y_test, preds_finais, target_names=["Retido (0)", "Churn (1)"]))

    os.makedirs("models", exist_ok=True)
    artefato = {
        "pipeline": melhor_pipeline,
        "threshold": threshold_otimo,
        "modelo_nome": melhor_nome,
        "roc_auc_teste": round(melhor_auc, 4),
        "versao": "1.0.0"
    }
    joblib.dump(artefato, "models/churn_pipeline_v1.joblib")
    print("4. Artefato salvo com sucesso em: models/churn_pipeline_v1.joblib")

if __name__ == "__main__":
    treinar_e_exportar()