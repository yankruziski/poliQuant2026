# -*- coding: utf-8 -*-
# 03_xgb_prediction.py — Walk-Forward XGBoost (POLIQUANT 2026)
# Treina XGBRegressor por janela mensal, prevendo rank de retorno t+1.
# Input:  master_dataset_ml_enriched.csv
# Output: expected_returns.csv

import os
import warnings
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- Parametros ----------------------------------------------------------------
DATA_INICIO_TESTE = "2015-01-01"       # inicio do out-of-sample
FREQ_REBALANCEAMENTO_MESES = 1         # retreino mensal

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
}

FEATURES_NUMERICAS = [
    "vol_21d", "momentum_6m",
    "dist_sma_20", "dist_sma_50", "dist_sma_200", "rsi_14",
    "vol_21d_z", "momentum_6m_z",
    "dist_sma_20_z", "dist_sma_50_z", "dist_sma_200_z", "rsi_14_z",
]

TARGET = "target_t1"

# -- Carga dos dados -----------------------------------------------------------

caminho = os.path.join(BASE_DIR, "data", "processed", "master_dataset_ml_enriched.csv")
df = pd.read_csv(caminho, index_col=[0, 1], parse_dates=[0])

print(f"Dataset carregado: {df.shape}")
print(f"Colunas: {df.columns.tolist()}")
print(f"Periodo: {df.index.get_level_values('Data').min().date()} "
      f"a {df.index.get_level_values('Data').max().date()}")

# -- One-hot do regime HMM (categorico, nao ordinal) --------------------------
dummies = pd.get_dummies(df["regime_hmm"], prefix="regime_hmm", drop_first=True)
dummies = dummies.astype(int)
df = pd.concat([df, dummies], axis=1)

FEATURES = FEATURES_NUMERICAS + [c for c in dummies.columns]
print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")

# -- Walk-Forward setup --------------------------------------------------------
todas_datas = df.index.get_level_values("Data").unique().sort_values()
datas_teste = todas_datas[todas_datas >= DATA_INICIO_TESTE]
periodos = datas_teste.to_period("M").unique().sort_values()
periodos_rebal = periodos[::FREQ_REBALANCEAMENTO_MESES]

print(f"\n--- Walk-Forward ---")
print(f"Treino desde:    {todas_datas[0].date()}")
print(f"OOS desde:       {DATA_INICIO_TESTE}")
print(f"Periodos rebal:  {len(periodos_rebal)}")

# -- Loop Walk-Forward ---------------------------------------------------------
todas_previsoes = []

for i, periodo in enumerate(periodos_rebal):
    # Janela de teste
    if i + 1 < len(periodos_rebal):
        proximo_periodo = periodos_rebal[i + 1]
        mask_teste = (todas_datas.to_period("M") >= periodo) & \
                     (todas_datas.to_period("M") < proximo_periodo)
    else:
        # Ultimo bloco: pega tudo ate o fim
        mask_teste = todas_datas.to_period("M") >= periodo

    datas_periodo_teste = todas_datas[mask_teste]
    if len(datas_periodo_teste) == 0:
        continue

    data_cutoff = datas_periodo_teste[0]

    # Treino = passado estrito; teste = periodo corrente
    mask_treino = df.index.get_level_values("Data") < data_cutoff
    mask_teste_df = df.index.get_level_values("Data").isin(datas_periodo_teste)

    df_treino = df.loc[mask_treino]
    df_teste = df.loc[mask_teste_df]

    if len(df_treino) < 1000 or len(df_teste) == 0:
        continue

    X_train = df_treino[FEATURES]
    y_train = df_treino[TARGET]

    # Rank cross-sectional: modelo aprende ordenacao relativa, nao nivel absoluto
    y_train = y_train.groupby(level="Data").rank(pct=True)

    # Limpar NaN residuais
    valid_idx = y_train.notna() & X_train.notna().all(axis=1)
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    X_test = df_teste[FEATURES]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    previsoes_periodo = pd.DataFrame(
        {"Retorno_Previsto": y_pred}, index=df_teste.index,
    )
    todas_previsoes.append(previsoes_periodo)

    # Log de progresso
    pct = (i + 1) / len(periodos_rebal) * 100
    barra = '#' * int(pct // 2) + '-' * (50 - int(pct // 2))
    print(f"\r  [{barra}] {pct:5.1f}%  periodo {i+1}/{len(periodos_rebal)}", end="", flush=True)
    if (i + 1) % 12 == 0 or i == 0 or i == len(periodos_rebal) - 1:
        print(f"\n    {periodo} | treino={len(df_treino):,} linhas, teste={len(df_teste):,} linhas")

print()

# -- Consolidacao e export -----------------------------------------------------
df_previsoes = pd.concat(todas_previsoes)
df_previsoes = df_previsoes.reset_index()
df_previsoes = df_previsoes.sort_values(["Data", "Ticker"]).reset_index(drop=True)

print(f"\n--- Resultado Final ---")
print(f"Total de previsoes: {len(df_previsoes):,}")
print(f"Periodo: {df_previsoes['Data'].min().date()} a {df_previsoes['Data'].max().date()}")
print(f"Tickers unicos: {df_previsoes['Ticker'].nunique()}")
print(f"\nEstatisticas do Retorno_Previsto:")
print(df_previsoes["Retorno_Previsto"].describe().round(6))

caminho_output = os.path.join(BASE_DIR, "data", "processed", "expected_returns.csv")
df_previsoes.to_csv(caminho_output, index=False)
print(f"\nSalvo em: {caminho_output}")

# Feature importance do ultimo modelo
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print(f"\nFeature Importance (ultimo modelo):")
for feat, imp in importance.items():
    barra = "#" * int(imp * 50)
    print(f"  {feat:<18s} {imp:.4f} {barra}")

print(f"\n{'='*50}")
print(f"WALK-FORWARD CONCLUIDO")
print(f"{'='*50}")
print(f"Periodos de rebalanceamento: {len(periodos_rebal)}")
print(f"Previsoes geradas: {len(df_previsoes):,}")
print(f"Output: expected_returns.csv")
