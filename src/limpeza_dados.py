# -*- coding: utf-8 -*-
# limpeza_dados.py — Limpeza e feature engineering (POLIQUANT 2026)
# CSV bruto -> retornos log -> winsorizacao -> features -> master_dataset_ml.csv

import os
import warnings
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

warnings.filterwarnings("ignore", message="invalid value encountered in log")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)

# -- Caminhos ------------------------------------------------------------------
caminho_acoes = os.path.join(BASE_DIR, "data", "raw", "adjclose_acoes_b3_2010_2025.csv")
caminho_ibov  = os.path.join(BASE_DIR, "data", "raw", "adjclose_ibov_2010_2025.csv")

dir_processed    = os.path.join(BASE_DIR, "data", "processed")
dir_diagnosticos = os.path.join(BASE_DIR, "outputs", "diagnosticos")
dir_outputs      = os.path.join(BASE_DIR, "outputs")

# -- Parametros ----------------------------------------------------------------
inicio = "2010-01-01"
fim    = "2025-12-31"

min_cobertura = 0.5   # ativo precisa ter dado em >= 50% dos dias
z_limite      = 8     # z-score pra flaggar outlier no diagnostico
k_sigma       = 3.0   # clipping a +-3 sigma na winsorizacao


# -- Carga: acoes --------------------------------------------------------------
acoes = pd.read_csv(caminho_acoes)
acoes["Data"] = pd.to_datetime(acoes["Data"], errors="coerce")
acoes = acoes.set_index("Data").sort_index()
acoes = acoes.dropna(axis=1, how="all")
acoes = acoes.apply(pd.to_numeric, errors="coerce")
acoes = acoes.loc[acoes.index.notna()]

# -- Carga: IBOV ---------------------------------------------------------------
ibov = pd.read_csv(caminho_ibov)
ibov["Data"] = pd.to_datetime(ibov["Data"], errors="coerce")
ibov = ibov.set_index("Data").sort_index()
ibov["IBOV"] = pd.to_numeric(ibov["IBOV"], errors="coerce")

# -- Filtro de periodo ---------------------------------------------------------
acoes = acoes.loc[inicio:fim]
ibov  = ibov.loc[inicio:fim]

if acoes.empty or ibov.empty:
    raise ValueError("Uma das bases ficou vazia apos filtro de periodo.")


# -- Alinhamento: inner join por data ------------------------------------------
dados = acoes.join(ibov[["IBOV"]], how="inner")
dados = dados[~dados.index.duplicated(keep="first")].sort_index()
print("Shape apos join:", dados.shape)


# -- Diagnostico de NaNs -------------------------------------------------------
col_ativos = [c for c in dados.columns if c != "IBOV"]
pct_nan = dados[col_ativos].isna().mean().sort_values(ascending=False)
ativos_nan_extremo = pct_nan[pct_nan > 0.90].index.tolist()
print("Ativos com >90% de NaN:", len(ativos_nan_extremo))
pct_nan.to_csv(os.path.join(dir_diagnosticos, "diagnostico_nan_por_ativo.csv"))


# -- Limpeza de NaNs -----------------------------------------------------------
# Filtra ativos com historico minimo, ffill gaps, remove penny stocks
obs_validas = dados[col_ativos].notna().sum()
ativos_validos = obs_validas[obs_validas >= min_cobertura * len(dados)].index
dados_filtrado = dados[list(ativos_validos) + ["IBOV"]].copy()

dados_ffill = dados_filtrado.sort_index().ffill()
dados_ffill = dados_ffill.bfill(limit=2)

# Penny stocks distorcem a covariancia — remove quem tem preco mediano < R$2
cols_ativos_ffill = [c for c in dados_ffill.columns if c != "IBOV"]
precos_medianos = dados_ffill[cols_ativos_ffill].median()
penny_stocks = precos_medianos[precos_medianos < 2.0].index.tolist()
if penny_stocks:
    dados_ffill = dados_ffill.drop(columns=penny_stocks, errors="ignore")
    print(f"Penny stocks removidos (preco mediano < R$2): {len(penny_stocks)}")

nan_residual = dados_ffill.isna().mean().sort_values(ascending=False)
print(nan_residual.head(10))


# -- Retornos logaritmicos -----------------------------------------------------
log_retornos = np.log(dados_ffill / dados_ffill.shift(1))
log_retornos = log_retornos.dropna(how="all")

ret_ativos = log_retornos.drop(columns=["IBOV"], errors="ignore")
ret_ibov   = log_retornos[["IBOV"]].copy()


# -- Diagnostico de outliers ---------------------------------------------------
q = ret_ativos.quantile([0.001, 0.01, 0.99, 0.999]).T
q.to_csv(os.path.join(dir_diagnosticos, "diagnostico_quantis_retornos.csv"))

z = (ret_ativos - ret_ativos.mean()) / ret_ativos.std(ddof=0)
mask_outlier = z.abs() > z_limite
print("Total de outliers detectados:", mask_outlier.sum().sum())
mask_outlier.to_csv(os.path.join(dir_diagnosticos, "diagnostico_mask_outliers.csv"))


# -- Validacoes ----------------------------------------------------------------
# Remove colunas constantes (var ~0 torna cov singular)
var_por_col = ret_ativos.var()
col_constantes = var_por_col[var_por_col <= 1e-12].index.tolist()
ret_ativos = ret_ativos.drop(columns=col_constantes, errors="ignore")

cov = ret_ativos.cov()
print("Cov shape:", cov.shape)
eigvals = np.linalg.eigvalsh(cov.values)
print("Menor autovalor:", eigvals.min())


# -- Exports intermediarios ----------------------------------------------------
dados_ffill.to_csv(os.path.join(dir_processed, "precos_limpos_ffill.csv"))
log_retornos.to_csv(os.path.join(dir_processed, "retornos_log_brutos.csv"))
ret_ativos.to_csv(os.path.join(dir_processed, "retornos_log_limpos.csv"))

with open(os.path.join(dir_outputs, "relatorio_limpeza.txt"), "w") as f:
    f.write(f"Quantidade inicial de ativos: {len(col_ativos)}\n")
    f.write(f"Quantidade final de ativos: {len(ret_ativos.columns)}\n")
    f.write(f"Ativos removidos por historico curto: {len(col_ativos) - len(ativos_validos)}\n")
    f.write(f"Percentual medio de NaN antes: {pct_nan.mean():.2%}\n")
    f.write(f"Percentual medio de NaN depois: {ret_ativos.isna().mean().mean():.2%}\n")
    f.write(f"Total de outliers detectados: {mask_outlier.sum().sum()}\n")


# -- Winsorizacao k-sigma ------------------------------------------------------
# Clip retornos alem de +-3 sigma pra nao distorcer a covariancia
media  = ret_ativos.mean()
desvio = ret_ativos.std(ddof=0)
lim_inf = media - k_sigma * desvio
lim_sup = media + k_sigma * desvio

qtd_trunc = ((ret_ativos < lim_inf) | (ret_ativos > lim_sup)).sum().sum()
print(f"Total de pontos winsorizados (k={k_sigma}): {int(qtd_trunc)}")

ret_ativos_w = ret_ativos.clip(lower=lim_inf, upper=lim_sup, axis=1)

retornos_limpos = ret_ativos_w.join(ret_ibov, how="left")
retornos_limpos.to_csv(os.path.join(dir_processed, "retornos_limpos_final.csv"))


# -- Feature engineering -------------------------------------------------------
precos_ativos = dados_ffill.drop(columns=["IBOV"], errors="ignore")
precos_ibov   = dados_ffill[["IBOV"]]

# Volatilidade 21d (risco recente)
vol_21d_ativos = ret_ativos_w.rolling(window=21, min_periods=15).std()
vol_21d_ibov   = ret_ibov.rolling(window=21, min_periods=15).std()
vol_21d_ibov.columns = ["vol_21d_IBOV"]

# Momentum 6m (soma log returns = retorno acumulado)
momentum_6m_ativos = ret_ativos_w.rolling(window=126, min_periods=100).sum()
momentum_6m_ibov   = ret_ibov.rolling(window=126, min_periods=100).sum()
momentum_6m_ibov.columns = ["momentum_6m_IBOV"]

# Distancia % para SMAs (curto, medio, longo prazo)
sma_20_ativos = precos_ativos.rolling(window=20, min_periods=15).mean()
dist_sma_20_ativos = (precos_ativos - sma_20_ativos) / sma_20_ativos

sma_50_ativos = precos_ativos.rolling(window=50, min_periods=40).mean()
dist_sma_50_ativos = (precos_ativos - sma_50_ativos) / sma_50_ativos

sma_200_ativos = precos_ativos.rolling(window=200, min_periods=150).mean()
dist_sma_200_ativos = (precos_ativos - sma_200_ativos) / sma_200_ativos

sma_200_ibov = precos_ibov.rolling(window=200, min_periods=150).mean()
dist_sma_200_ibov = (precos_ibov - sma_200_ibov) / sma_200_ibov
dist_sma_200_ibov.columns = ["dist_sma_200_IBOV"]

# RSI 14 dias
def calc_rsi(retornos, janela=14):
    ganhos = retornos.clip(lower=0)
    perdas = (-retornos).clip(lower=0)
    media_ganhos = ganhos.ewm(span=janela, min_periods=janela).mean()
    media_perdas = perdas.ewm(span=janela, min_periods=janela).mean()
    rs = media_ganhos / media_perdas.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

rsi_14_ativos = calc_rsi(ret_ativos_w, janela=14)

# Targets: retorno t+1 (diario) e t+5 (semanal)
target_t1_ativos = ret_ativos_w.shift(-1)
target_t1_ibov   = ret_ibov.shift(-1)
target_t1_ibov.columns = ["target_t1_IBOV"]
target_t5_ativos = ret_ativos_w.rolling(window=5).sum().shift(-5)

# Z-scores cross-section: normaliza cada feature por dia entre todos os ativos
def zscore_cross_section(df_wide):
    mu    = df_wide.mean(axis=1)
    sigma = df_wide.std(axis=1)
    return df_wide.sub(mu, axis=0).div(sigma.replace(0, np.nan), axis=0)

vol_21d_z      = zscore_cross_section(vol_21d_ativos)
momentum_6m_z  = zscore_cross_section(momentum_6m_ativos)
dist_sma_20_z  = zscore_cross_section(dist_sma_20_ativos)
dist_sma_50_z  = zscore_cross_section(dist_sma_50_ativos)
dist_sma_200_z = zscore_cross_section(dist_sma_200_ativos)
rsi_14_z       = zscore_cross_section(rsi_14_ativos)

print("Z-scores cross-section calculados para 6 features.")

# -- Formato Long (tidy) para ML -----------------------------------------------
master_df = pd.concat([
    vol_21d_ativos.stack(),
    momentum_6m_ativos.stack(),
    dist_sma_20_ativos.stack(),
    dist_sma_50_ativos.stack(),
    dist_sma_200_ativos.stack(),
    rsi_14_ativos.stack(),
    vol_21d_z.stack(),
    momentum_6m_z.stack(),
    dist_sma_20_z.stack(),
    dist_sma_50_z.stack(),
    dist_sma_200_z.stack(),
    rsi_14_z.stack(),
    target_t1_ativos.stack(),
    target_t5_ativos.stack(),
], axis=1)

master_df.columns = [
    "vol_21d", "momentum_6m", "dist_sma_20", "dist_sma_50",
    "dist_sma_200", "rsi_14",
    "vol_21d_z", "momentum_6m_z", "dist_sma_20_z", "dist_sma_50_z",
    "dist_sma_200_z", "rsi_14_z",
    "target_t1", "target_t5",
]
master_df.index.names = ["Data", "Ticker"]
master_df = master_df.dropna()

master_df.to_csv(os.path.join(dir_processed, "master_dataset_ml.csv"))

print("Feature engineering concluido.")
print(f"  master_df shape: {master_df.shape}")
print(f"  Colunas: {master_df.columns.tolist()}")
print(f"  Tickers unicos: {master_df.index.get_level_values('Ticker').nunique()}")
print(master_df.head(10))
