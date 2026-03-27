# -*- coding: utf-8 -*-
# hmm_regimes.py — Deteccao de regimes de mercado (POLIQUANT 2026)
# HMM Gaussiano nos retornos IBOV -> classifica cada dia em Bull/Bear/Lateral
# e enriquece o master dataset com regime + target binario.

import os
import warnings
import logging
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*not converging.*")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- Parametros ----------------------------------------------------------------
n_regimes   = 3      # Bull, Bear, Lateral
n_iter      = 1000   # iteracoes EM (converge em ~100, mas folga nao custa)
random_seed = 42
n_splits    = 5      # folds para validacao temporal


# -- Carga dos dados -----------------------------------------------------------
caminho_retornos = os.path.join(BASE_DIR, "data", "processed", "retornos_limpos_final.csv")
df_ret = pd.read_csv(caminho_retornos, index_col="Data", parse_dates=True)

caminho_master = os.path.join(BASE_DIR, "data", "processed", "master_dataset_ml.csv")
master_df = pd.read_csv(caminho_master, index_col=[0, 1], parse_dates=[0])

print(f"Retornos shape: {df_ret.shape}")
print(f"Master shape:   {master_df.shape}")

# -- Retornos IBOV como input do HMM ------------------------------------------
ibov_ret = df_ret[["IBOV"]].dropna()
X = ibov_ret.values.reshape(-1, 1)

print(f"Input HMM: {X.shape}")
print(f"Periodo: {ibov_ret.index[0].date()} a {ibov_ret.index[-1].date()}")


# -- Treinamento do HMM -------------------------------------------------------
model = GaussianHMM(
    n_components=n_regimes,
    covariance_type="full",
    n_iter=n_iter,
    random_state=random_seed,
)
model.fit(X)

# Viterbi: sequencia mais provavel de estados ocultos
regimes = model.predict(X)

print(f"Regimes unicos: {np.unique(regimes)}")
print(f"Contagem por regime: {dict(zip(*np.unique(regimes, return_counts=True)))}")


# -- Identificar qual estado eh Bear/Bull/Lateral -----------------------------
# Bear = maior variancia, Bull = maior media dos restantes, Lateral = sobra
for i in range(n_regimes):
    media_estado = model.means_[i, 0]
    var_estado = model.covars_[i, 0, 0]
    print(f"Estado {i}: media={media_estado:.6f}, desvio={np.sqrt(var_estado):.6f}")

variancias     = [model.covars_[i, 0, 0] for i in range(n_regimes)]
estado_bear    = int(np.argmax(variancias))
restantes      = [i for i in range(n_regimes) if i != estado_bear]
estado_bull    = max(restantes, key=lambda i: model.means_[i, 0])
estado_lateral = [i for i in range(n_regimes) if i not in (estado_bear, estado_bull)][0]

# Mapeamento numerico → nome legivel
mapa_regimes = {
    estado_bull: "bull",
    estado_bear: "bear",
    estado_lateral: "lateral",
}

print(f"\nMapeamento final:")
print(f"  Bull    (estado {estado_bull}): media={model.means_[estado_bull, 0]:.6f}, "
      f"desvio={np.sqrt(model.covars_[estado_bull, 0, 0]):.6f}")
print(f"  Bear    (estado {estado_bear}): media={model.means_[estado_bear, 0]:.6f}, "
      f"desvio={np.sqrt(model.covars_[estado_bear, 0, 0]):.6f}")
print(f"  Lateral (estado {estado_lateral}): media={model.means_[estado_lateral, 0]:.6f}, "
      f"desvio={np.sqrt(model.covars_[estado_lateral, 0, 0]):.6f}")


# -- Serie de regimes ----------------------------------------------------------
df_regimes = pd.DataFrame({"regime_hmm": regimes}, index=ibov_ret.index)
df_regimes["regime_nome"] = df_regimes["regime_hmm"].map(mapa_regimes)

print(f"\nDistribuicao dos regimes:")
print(df_regimes["regime_nome"].value_counts())


# -- Enriquecer master dataset com regime --------------------------------------
master_enriched = master_df.reset_index()
master_enriched = master_enriched.merge(
    df_regimes[["regime_hmm"]],
    left_on="Data", right_index=True, how="left",
)
master_enriched = master_enriched.set_index(["Data", "Ticker"])

nan_regime = master_enriched["regime_hmm"].isna().sum()
print(f"\nNaN em regime_hmm: {nan_regime}")

if nan_regime > 0:
    master_enriched = master_enriched.dropna(subset=["regime_hmm"])
    print(f"Linhas apos dropar NaN: {len(master_enriched)}")

master_enriched["regime_hmm"] = master_enriched["regime_hmm"].astype(int)

# Target binario: 1 = subiu, 0 = caiu/ficou
master_enriched["target_binary"] = (master_enriched["target_t1"] > 0).astype(int)

print(f"\nDistribuicao do target binario:")
print(master_enriched["target_binary"].value_counts(normalize=True).round(4))


# -- Preview dos folds temporais -----------------------------------------------
master_enriched = master_enriched.sort_index(level="Data")
tscv = TimeSeriesSplit(n_splits=n_splits)

datas_unicas = master_enriched.index.get_level_values("Data").unique().sort_values()
print(f"\nTotal de datas unicas: {len(datas_unicas)}")
print(f"Primeira data: {datas_unicas[0].date()}")
print(f"Ultima data:   {datas_unicas[-1].date()}")

print(f"\nPreview dos {n_splits} folds temporais:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(datas_unicas)):
    datas_treino = datas_unicas[train_idx]
    datas_teste = datas_unicas[test_idx]
    print(f"  Fold {fold+1}: treino ate {datas_treino[-1].date()}, "
          f"teste {datas_teste[0].date()} a {datas_teste[-1].date()} "
          f"({len(datas_teste)} dias)")


# -- Exports -------------------------------------------------------------------
caminho_enriched = os.path.join(BASE_DIR, "data", "processed", "master_dataset_ml_enriched.csv")
master_enriched.to_csv(caminho_enriched)
print(f"\nMaster enriched salvo: {master_enriched.shape}")

caminho_modelo = os.path.join(BASE_DIR, "models", "hmm_ibov_model.pkl")
joblib.dump(model, caminho_modelo)
print(f"Modelo HMM salvo em: {caminho_modelo}")

caminho_regimes = os.path.join(BASE_DIR, "data", "processed", "regimes_hmm_ibov.csv")
df_regimes.to_csv(caminho_regimes)
print(f"Regimes salvos em: {caminho_regimes}")

# Matriz de transicao entre regimes
print("\nMatriz de transicao:")
print(pd.DataFrame(
    model.transmat_,
    index=[f"de_{mapa_regimes[i]}" for i in range(n_regimes)],
    columns=[f"para_{mapa_regimes[i]}" for i in range(n_regimes)],
).round(4))

# Resumo final
print(f"\n{'='*50}")
print(f"RESUMO DO HMM (3 REGIMES)")
print(f"{'='*50}")
print(f"Regimes detectados: {n_regimes}")
print(f"Dias em Bull:    {(df_regimes['regime_hmm'] == estado_bull).sum()}")
print(f"Dias em Bear:    {(df_regimes['regime_hmm'] == estado_bear).sum()}")
print(f"Dias em Lateral: {(df_regimes['regime_hmm'] == estado_lateral).sum()}")
print(f"Master enriched shape: {master_enriched.shape}")
print(f"Colunas finais: {master_enriched.columns.tolist()}")
