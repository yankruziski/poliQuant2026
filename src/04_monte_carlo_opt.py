# -*- coding: utf-8 -*-
# 04_monte_carlo_opt.py — MC CVaR + Markowitz + Backtest (POLIQUANT 2026)
# Walk-Forward mensal: seleciona top-N via XGBoost → cov por regime → MC CVaR
# → pesos Markowitz (max Sharpe) → retornos realizados.

import os
import warnings
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- Parametros ----------------------------------------------------------------
RISK_FREE_RATE_ANUAL = 0.1075
RISK_FREE_DIARIO = (1 + RISK_FREE_RATE_ANUAL) ** (1/252) - 1

MAX_PESO_ATIVO = 0.10
MIN_PESO_ATIVO = 0.0
MAX_PESO_SETOR = 0.30       # reservado (sem dados setoriais no dataset)

N_SIMULACOES = 5000
HORIZONTE_DIAS = 21
CVAR_NIVEL = 0.05

DATA_INICIO_BACKTEST = "2015-01-01"
FREQ_REBALANCEAMENTO_MESES = 1
TOP_N_ATIVOS = 30
MIN_DIAS_COV = 60


# -- Carga dos dados -----------------------------------------------------------
caminho_ret = os.path.join(BASE_DIR, "data", "processed", "retornos_limpos_final.csv")
df_ret = pd.read_csv(caminho_ret, index_col="Data", parse_dates=True)

caminho_er = os.path.join(BASE_DIR, "data", "processed", "expected_returns.csv")
df_er = pd.read_csv(caminho_er, parse_dates=["Data"])

caminho_reg = os.path.join(BASE_DIR, "data", "processed", "regimes_hmm_ibov.csv")
df_regimes = pd.read_csv(caminho_reg, index_col="Data", parse_dates=True)

ret_ibov = df_ret[["IBOV"]].copy()
ret_acoes = df_ret.drop(columns=["IBOV"], errors="ignore")

caminho_precos = os.path.join(BASE_DIR, "data", "processed", "precos_limpos_ffill.csv")
df_precos = pd.read_csv(caminho_precos, index_col="Data", parse_dates=True)
df_precos = df_precos.drop(columns=["IBOV"], errors="ignore")

print(f"Retornos acoes: {ret_acoes.shape}")
print(f"Expected returns: {df_er.shape}")
print(f"Regimes: {df_regimes.shape}")


# -- Funcoes auxiliares --------------------------------------------------------

def covariancia_regime(ret_acoes, df_regimes, data_ref, regime_atual, tickers,
                       min_dias=MIN_DIAS_COV):
    """
    Covariancia condicional ao regime HMM. Fallback: cov global com shrinkage.
    """
    # Filtrar dias passados no mesmo regime
    mask_regime = (df_regimes.index < data_ref) & (df_regimes["regime_hmm"] == regime_atual)
    datas_regime = df_regimes.index[mask_regime]

    ret_filtrado = ret_acoes.loc[ret_acoes.index.isin(datas_regime), tickers].dropna(axis=1, how="all")

    if len(ret_filtrado) < min_dias:
        ret_filtrado = ret_acoes.loc[ret_acoes.index < data_ref, tickers].dropna(axis=1, how="all")

    ret_filtrado = ret_filtrado.dropna()

    cov = ret_filtrado.cov()

    # Shrinkage simples (mistura com diagonal)
    alpha = 0.1
    diag = np.diag(np.diag(cov.values))
    cov_shrunk = (1 - alpha) * cov.values + alpha * diag
    cov_shrunk = pd.DataFrame(cov_shrunk, index=cov.index, columns=cov.columns)

    return cov_shrunk


def monte_carlo_cvar(mu_hat, cov_matrix, n_sim=N_SIMULACOES,
                     horizonte=HORIZONTE_DIAS, nivel=CVAR_NIVEL):
    """
    Simulacao Monte Carlo: N cenarios N(mu, Sigma) → CVaR e travas de peso.
    """
    n_ativos = len(mu_hat)
    if n_ativos == 0:
        return 0, 0, np.array([]), np.array([])

    mu = mu_hat.values if hasattr(mu_hat, "values") else np.array(mu_hat)

    cov = cov_matrix.values if hasattr(cov_matrix, "values") else np.array(cov_matrix)
    cov = cov + np.eye(n_ativos) * 1e-8

    try:
        retornos_sim = np.random.multivariate_normal(mu, cov, size=(n_sim, horizonte))
    except np.linalg.LinAlgError:
        cov = np.diag(np.diag(cov))
        retornos_sim = np.random.multivariate_normal(mu, cov, size=(n_sim, horizonte))

    retornos_acum = retornos_sim.sum(axis=1)
    ret_portfolio = retornos_acum.mean(axis=1)

    ret_sorted = np.sort(ret_portfolio)
    idx_var = int(n_sim * nivel)
    var = ret_sorted[idx_var]
    cvar = ret_sorted[:idx_var].mean() if idx_var > 0 else var

    # Travas de peso: ativos que mais contribuem p/ perda nos piores cenarios
    piores_cenarios = retornos_acum[ret_portfolio <= var]
    if len(piores_cenarios) > 0:
        contrib_perda = piores_cenarios.mean(axis=0)
        perda_abs = np.abs(contrib_perda)
        perda_rel = perda_abs / (perda_abs.sum() + 1e-12)
        # Peso maximo inversamente proporcional a contribuicao de perda
        limite_pesos = np.clip(MAX_PESO_ATIVO / (1 + 3 * perda_rel), 0.01, MAX_PESO_ATIVO)
    else:
        limite_pesos = np.full(n_ativos, MAX_PESO_ATIVO)

    return cvar, var, limite_pesos, retornos_sim


def otimizar_markowitz(mu_hat, cov_matrix, limite_pesos, rf=RISK_FREE_DIARIO):
    """
    Max Sharpe via SLSQP. Long only, soma = 1, limites individuais via CVaR.
    """
    n = len(mu_hat)
    mu = mu_hat.values if hasattr(mu_hat, "values") else np.array(mu_hat)
    cov = cov_matrix.values if hasattr(cov_matrix, "values") else np.array(cov_matrix)

    def neg_sharpe(w):
        ret_port = w @ mu
        vol_port = np.sqrt(w @ cov @ w)
        if vol_port < 1e-12:
            return 1e6
        return -(ret_port - rf) / vol_port

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(MIN_PESO_ATIVO, lim) for lim in limite_pesos]
    w0 = np.full(n, 1.0 / n)

    result = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    pesos = result.x if result.success else w0
    pesos = pesos / pesos.sum()
    return pesos


# -- Walk-Forward Backtest -----------------------------------------------------

todas_datas = ret_acoes.index.sort_values()
datas_backtest = todas_datas[todas_datas >= DATA_INICIO_BACKTEST]

periodos = datas_backtest.to_period("M").unique().sort_values()
periodos_rebal = periodos[::FREQ_REBALANCEAMENTO_MESES]

print(f"\n--- Backtest ---")
print(f"Periodo: {DATA_INICIO_BACKTEST} a {todas_datas[-1].date()}")
print(f"Rebalanceamentos: {len(periodos_rebal)} | Top-{TOP_N_ATIVOS} ativos")
print(f"MC: {N_SIMULACOES} sims, {HORIZONTE_DIAS}d, CVaR {CVAR_NIVEL*100:.0f}%")

np.random.seed(42)

resultados_diarios = []    # (Data, ret_portfolio, ret_ibov)
historico_pesos = []       # (Data_rebal, Ticker, Peso)

for i, periodo in enumerate(periodos_rebal):
    # Janela de teste
    if i + 1 < len(periodos_rebal):
        proximo = periodos_rebal[i + 1]
        mask = (todas_datas.to_period("M") >= periodo) & \
               (todas_datas.to_period("M") < proximo)
    else:
        mask = todas_datas.to_period("M") >= periodo

    datas_periodo = todas_datas[mask]
    if len(datas_periodo) == 0:
        continue

    data_ref = datas_periodo[0]

    # mu_hat do XGBoost: media das previsoes do mes anterior (ranking)
    mes_anterior = periodo - 1
    mask_er = df_er["Data"].dt.to_period("M") == mes_anterior
    er_periodo = df_er.loc[mask_er].groupby("Ticker")["Retorno_Previsto"].mean()

    if len(er_periodo) == 0:
        # Se nao ha previsoes no mes anterior, tentar o proprio periodo
        mask_er = df_er["Data"].dt.to_period("M") == periodo
        er_periodo = df_er.loc[mask_er].groupby("Ticker")["Retorno_Previsto"].mean()

    if len(er_periodo) < 5:
        continue

    # Top-N por ranking XGBoost
    tickers_disponiveis = [t for t in er_periodo.index if t in ret_acoes.columns]
    er_filtrado = er_periodo[tickers_disponiveis]

    # Filtro penny stocks (preco < R$2)
    precos_antes = df_precos.loc[df_precos.index < data_ref]
    if len(precos_antes) > 0:
        ultimo_preco = precos_antes.iloc[-1]
        tickers_caros = [t for t in er_filtrado.index
                         if t in ultimo_preco.index and ultimo_preco[t] >= 2.0]
        er_filtrado = er_filtrado[tickers_caros]

    top_tickers = er_filtrado.nlargest(TOP_N_ATIVOS).index.tolist()

    if len(top_tickers) < 3:
        continue

    # mu_hat = media historica trailing 252d (retornos reais, nao ranks)
    janela_mu = ret_acoes.loc[ret_acoes.index < data_ref, top_tickers].tail(252)
    mu_hat = janela_mu.mean()
    mu_hat = mu_hat.dropna()
    top_tickers = mu_hat.index.tolist()

    if len(top_tickers) < 3:
        continue

    # Covariancia condicional ao regime vigente
    regimes_passados = df_regimes.loc[df_regimes.index < data_ref]
    if len(regimes_passados) == 0:
        continue

    regime_atual = regimes_passados["regime_hmm"].iloc[-1]

    cov_matrix = covariancia_regime(
        ret_acoes, df_regimes, data_ref, regime_atual, top_tickers
    )

    tickers_validos = [t for t in top_tickers if t in cov_matrix.columns]
    if len(tickers_validos) < 3:
        continue

    mu_hat = mu_hat[tickers_validos]
    cov_matrix = cov_matrix.loc[tickers_validos, tickers_validos]

    # MC CVaR + Markowitz
    cvar, var, limite_pesos, retornos_sim_periodo = monte_carlo_cvar(mu_hat, cov_matrix)
    pesos_otimos = otimizar_markowitz(mu_hat, cov_matrix, limite_pesos)

    # Guardar pesos
    for ticker, peso in zip(tickers_validos, pesos_otimos):
        historico_pesos.append({
            "Data_Rebal": data_ref,
            "Ticker": ticker,
            "Peso": peso,
            "Regime": regime_atual,
        })

    # Retornos realizados no periodo
    ret_periodo = ret_acoes.loc[datas_periodo, tickers_validos].fillna(0)

    ret_port_diario = ret_periodo.values @ pesos_otimos
    ret_ibov_periodo = ret_ibov.loc[datas_periodo, "IBOV"].fillna(0)

    for j, data in enumerate(datas_periodo):
        resultados_diarios.append({
            "Data": data,
            "ret_portfolio": ret_port_diario[j],
            "ret_ibov": ret_ibov_periodo.iloc[j],
        })

    # Log de progresso
    pct = (i + 1) / len(periodos_rebal) * 100
    barra = '#' * int(pct // 2) + '-' * (50 - int(pct // 2))
    print(f"\r  [{barra}] {pct:5.1f}%  periodo {i+1}/{len(periodos_rebal)}", end="", flush=True)
    if (i + 1) % 12 == 0 or i == 0 or i == len(periodos_rebal) - 1:
        n_ativos = (pesos_otimos > 0.001).sum()
        print(f"\n    {periodo}: regime={'bull' if regime_atual==1 else 'bear' if regime_atual==2 else 'lateral'}, "
              f"{n_ativos} ativos, CVaR={cvar:.4f}, "
              f"max_peso={pesos_otimos.max():.2%}")

print()

# -- Consolidacao --------------------------------------------------------------

df_backtest = pd.DataFrame(resultados_diarios)
df_backtest["Data"] = pd.to_datetime(df_backtest["Data"])
df_backtest = df_backtest.sort_values("Data").reset_index(drop=True)

# Retornos acumulados
df_backtest["acum_portfolio"] = (1 + df_backtest["ret_portfolio"]).cumprod()
df_backtest["acum_ibov"] = (1 + df_backtest["ret_ibov"]).cumprod()

df_pesos = pd.DataFrame(historico_pesos)

print(f"\n--- Resultados ---")
print(f"Periodo: {df_backtest['Data'].min().date()} a {df_backtest['Data'].max().date()}")
print(f"Dias: {len(df_backtest)} | Rebalanceamentos: {df_pesos['Data_Rebal'].nunique()}")

# -- Metricas ------------------------------------------------------------------

def calcular_metricas(retornos, nome, rf_diario=RISK_FREE_DIARIO):
    """Metricas padrao: Sharpe, Max DD, Calmar, Win Rate."""
    # Retorno total
    ret_total = (1 + retornos).prod() - 1

    # Retorno anualizado
    n_anos = len(retornos) / 252
    ret_anual = (1 + ret_total) ** (1 / max(n_anos, 0.01)) - 1

    # Volatilidade anualizada
    vol_anual = retornos.std() * np.sqrt(252)

    # Sharpe Ratio anualizado
    excess = retornos.mean() - rf_diario
    sharpe = (excess / retornos.std()) * np.sqrt(252) if retornos.std() > 0 else 0

    # Max Drawdown
    acum = (1 + retornos).cumprod()
    pico = acum.cummax()
    drawdown = (acum - pico) / pico
    max_dd = drawdown.min()

    # Calmar Ratio (ret_anual / |max_dd|)
    calmar = ret_anual / abs(max_dd) if max_dd != 0 else 0

    # Win rate (% de dias positivos)
    win_rate = (retornos > 0).mean()

    return {
        "Nome": nome,
        "Ret Total": f"{ret_total:.2%}",
        "Ret Anual": f"{ret_anual:.2%}",
        "Vol Anual": f"{vol_anual:.2%}",
        "Sharpe": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Calmar": f"{calmar:.3f}",
        "Win Rate": f"{win_rate:.2%}",
    }


metricas_port = calcular_metricas(df_backtest["ret_portfolio"], "Portfolio")
metricas_ibov = calcular_metricas(df_backtest["ret_ibov"], "IBOV")

print(f"\n{'='*60}")
print(f"{'METRICAS DE PERFORMANCE':^60}")
print(f"{'='*60}")
metricas_df = pd.DataFrame([metricas_port, metricas_ibov]).set_index("Nome")
print(metricas_df.to_string())
print(f"{'='*60}")

# -- Exports -------------------------------------------------------------------

caminho_bt = os.path.join(BASE_DIR, "data", "processed", "portfolio_backtest.csv")
df_backtest.to_csv(caminho_bt, index=False)

caminho_pesos = os.path.join(BASE_DIR, "data", "processed", "portfolio_pesos.csv")
df_pesos.to_csv(caminho_pesos, index=False)

# 3) Relatorio
caminho_rel = os.path.join(BASE_DIR, "outputs", "relatorio_portfolio.txt")
with open(caminho_rel, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("RELATORIO DE PORTFOLIO - POLIQUANT 2026\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Periodo: {df_backtest['Data'].min().date()} a {df_backtest['Data'].max().date()}\n")
    f.write(f"Dias de operacao: {len(df_backtest)}\n")
    f.write(f"Rebalanceamentos: {df_pesos['Data_Rebal'].nunique()}\n")
    f.write(f"Top N ativos: {TOP_N_ATIVOS}\n")
    f.write(f"Max peso por ativo: {MAX_PESO_ATIVO*100:.0f}%\n")
    f.write(f"Monte Carlo: {N_SIMULACOES} sims, {HORIZONTE_DIAS} dias, CVaR {CVAR_NIVEL*100:.0f}%\n\n")

    f.write("METRICAS\n")
    f.write("-" * 60 + "\n")
    f.write(metricas_df.to_string() + "\n\n")

    # Retorno acumulado final
    f.write(f"Retorno acumulado Portfolio: {df_backtest['acum_portfolio'].iloc[-1]:.4f}x\n")
    f.write(f"Retorno acumulado IBOV:      {df_backtest['acum_ibov'].iloc[-1]:.4f}x\n\n")

    # Ultimo rebalanceamento
    ultimo_rebal = df_pesos.loc[df_pesos["Data_Rebal"] == df_pesos["Data_Rebal"].max()]
    ultimo_rebal = ultimo_rebal.sort_values("Peso", ascending=False)
    f.write("ULTIMO REBALANCEAMENTO\n")
    f.write("-" * 60 + "\n")
    for _, row in ultimo_rebal.head(15).iterrows():
        f.write(f"  {row['Ticker']:<10s} {row['Peso']:.2%}\n")

print(f"Relatorio salvo em: {caminho_rel}")

# -- Graficos ------------------------------------------------------------------

dir_plots = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(dir_plots, exist_ok=True)

# --- 9.1) Retorno Acumulado: Portfolio vs IBOV ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_backtest["Data"], df_backtest["acum_portfolio"],
        label="Portfolio", linewidth=1.5, color="#1f77b4")
ax.plot(df_backtest["Data"], df_backtest["acum_ibov"],
        label="IBOV", linewidth=1.5, color="#ff7f0e", alpha=0.8)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Retorno Acumulado: Portfolio vs IBOV", fontsize=14)
ax.set_xlabel("Data")
ax.set_ylabel("Retorno Acumulado (1 = inicio)")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(dir_plots, "retorno_acumulado.png"), dpi=150)
plt.close(fig)
print(f"Grafico retorno acumulado salvo.")

# 9.2) Underwater (Drawdown)
acum = df_backtest["acum_portfolio"]
pico = acum.cummax()
drawdown = (acum - pico) / pico  # valores negativos

fig_dd, ax_dd = plt.subplots(figsize=(14, 5))
ax_dd.fill_between(df_backtest["Data"], drawdown, 0,
                   color="#e74c3c", alpha=0.25)
ax_dd.plot(df_backtest["Data"], drawdown, color="#c0392b", linewidth=1.2)
ax_dd.set_title("Underwater: Drawdown do Portfolio", fontsize=14, fontweight="bold")
ax_dd.set_xlabel("Data")
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_dd.grid(True, alpha=0.3)
# Anotar max drawdown
idx_max_dd = drawdown.idxmin()
data_max_dd = df_backtest.loc[idx_max_dd, "Data"]
val_max_dd = drawdown[idx_max_dd]
ax_dd.annotate(f"Max DD: {val_max_dd:.1%}",
               xy=(data_max_dd, val_max_dd),
               xytext=(data_max_dd + pd.Timedelta(days=90), val_max_dd + 0.03),
               fontsize=10, fontweight="bold", color="#c0392b",
               arrowprops=dict(arrowstyle="->", color="#c0392b"))
fig_dd.tight_layout()
fig_dd.savefig(os.path.join(dir_plots, "drawdown_underwater.png"), dpi=150)
plt.close(fig_dd)
print(f"Grafico drawdown salvo.")

# 9.3) Sharpe Rolling (252d)
JANELA_SHARPE = 252
ret_port = df_backtest["ret_portfolio"]
rolling_mean = ret_port.rolling(JANELA_SHARPE).mean()
rolling_std = ret_port.rolling(JANELA_SHARPE).std()
rolling_sharpe = ((rolling_mean - RISK_FREE_DIARIO) / rolling_std) * np.sqrt(252)

fig_sh, ax_sh = plt.subplots(figsize=(14, 5))
ax_sh.plot(df_backtest["Data"], rolling_sharpe,
           color="black", linewidth=1.5, label="Sharpe Rolling 252d")
ax_sh.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax_sh.fill_between(df_backtest["Data"], rolling_sharpe, 0,
                   where=rolling_sharpe >= 0, color="#a8d8ea", alpha=0.4)
ax_sh.fill_between(df_backtest["Data"], rolling_sharpe, 0,
                   where=rolling_sharpe < 0, color="#e74c3c", alpha=0.3)
ax_sh.set_title("Sharpe Ratio Rolling (janela 252 dias)", fontsize=14, fontweight="bold")
ax_sh.set_xlabel("Data")
ax_sh.set_ylabel("Sharpe Ratio (anualizado)")
ax_sh.legend(fontsize=11)
ax_sh.grid(True, alpha=0.3)
fig_sh.tight_layout()
fig_sh.savefig(os.path.join(dir_plots, "sharpe_rolling.png"), dpi=150)
plt.close(fig_sh)
print(f"Grafico Sharpe rolling salvo.")

# 9.4) Monte Carlo: projecao spaghetti 6 meses
MC_HORIZONTE_LONGO = 126  # 6 meses em dias uteis

# Recuperar mu_hat e cov do ultimo rebalanceamento (ja existem no escopo)
try:
    n_ativos_mc = len(mu_hat)
    mu_mc = mu_hat.values if hasattr(mu_hat, "values") else np.array(mu_hat)
    cov_mc = cov_matrix.values if hasattr(cov_matrix, "values") else np.array(cov_matrix)
    cov_mc = cov_mc + np.eye(n_ativos_mc) * 1e-8

    # Simular retornos diarios multivariados
    np.random.seed(123)
    ret_sim_longo = np.random.multivariate_normal(
        mu_mc, cov_mc, size=(N_SIMULACOES, MC_HORIZONTE_LONGO)
    )  # (n_sim, 126, n_ativos)

    # Retorno do portfolio ponderado pelos pesos otimos (ultimo rebal)
    ret_port_sim = (ret_sim_longo * pesos_otimos).sum(axis=2)  # (n_sim, 126)

    # Acumular: retorno acumulado por cenario
    caminhos = np.cumprod(1 + ret_port_sim, axis=1)  # (n_sim, 126)

    # --- Grafico estilo "spaghetti" com caminhos coloridos ---
    fig2, ax2 = plt.subplots(figsize=(16, 7))
    dias_x = np.arange(1, MC_HORIZONTE_LONGO + 1)

    # Plotar 100 caminhos com cores distintas (ciclicas)
    n_plot = min(100, caminhos.shape[0])
    cmap = plt.cm.get_cmap("tab20", 20)
    for j in range(n_plot):
        ax2.plot(dias_x, caminhos[j], color=cmap(j % 20),
                 alpha=0.35, linewidth=0.7)

    # Percentis
    p05 = np.percentile(caminhos, 5, axis=0)
    p25 = np.percentile(caminhos, 25, axis=0)
    p50 = np.percentile(caminhos, 50, axis=0)
    p75 = np.percentile(caminhos, 75, axis=0)
    p95 = np.percentile(caminhos, 95, axis=0)

    # Faixas de confianca
    ax2.fill_between(dias_x, p05, p95, color="#b0d4f1", alpha=0.25,
                     label="Intervalo 5%-95%")
    ax2.fill_between(dias_x, p25, p75, color="#6baed6", alpha=0.25,
                     label="Intervalo 25%-75%")

    # Mediana em destaque
    ax2.plot(dias_x, p50, color="black", linewidth=3.0, label="Mediana",
             zorder=10)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    # Calcular datas de inicio e fim da simulacao
    mc_data_inicio = df_backtest["Data"].max()
    mc_data_fim = mc_data_inicio + pd.offsets.BDay(MC_HORIZONTE_LONGO)
    ax2.set_title(
        f"Monte Carlo: {N_SIMULACOES} cenarios "
        f"({mc_data_inicio.strftime('%d/%m/%Y')} ~ {mc_data_fim.strftime('%d/%m/%Y')})",
        fontsize=15, fontweight="bold")
    ax2.set_xlabel("Dia", fontsize=12)
    ax2.set_ylabel("Retorno acumulado do portfolio", fontsize=12)
    ax2.legend(fontsize=11, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Anotar valor final da mediana
    ax2.annotate(f"Mediana: {p50[-1]:.3f}x",
                 xy=(dias_x[-1], p50[-1]),
                 xytext=(dias_x[-1] - 20, p50[-1] + 0.02),
                 fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="black"))

    fig2.tight_layout()
    fig2.savefig(os.path.join(dir_plots, "monte_carlo_paths.png"), dpi=150)
    plt.close(fig2)
    print(f"Grafico Monte Carlo salvo em: {dir_plots}/monte_carlo_paths.png")

except Exception as e:
    print(f"Aviso: nao foi possivel gerar grafico MC longo: {e}")


# 9.5) Monte Carlo 3D — exporta dados para chart-data.js (HTML modular)
try:
    import json
    from scipy.stats import gaussian_kde

    ret_pct = (caminhos - 1) * 100

    dias_kde = list(range(0, MC_HORIZONTE_LONGO, 7))
    if (MC_HORIZONTE_LONGO - 1) not in dias_kde:
        dias_kde.append(MC_HORIZONTE_LONGO - 1)

    y_min, y_max = np.percentile(ret_pct, 1), np.percentile(ret_pct, 99)
    n_y = 60
    y_grid = np.linspace(y_min, y_max, n_y)

    Z_surface = np.zeros((len(dias_kde), n_y))
    for idx, d in enumerate(dias_kde):
        dados_dia = ret_pct[:, d]
        dados_dia = dados_dia[np.isfinite(dados_dia)]
        if len(dados_dia) > 10 and dados_dia.std() > 1e-8:
            Z_surface[idx, :] = gaussian_kde(dados_dia, bw_method=0.3)(y_grid)

    X_dias = [int(d + 1) for d in dias_kde]
    mc_data_inicio = df_backtest["Data"].max()
    mc_data_fim = mc_data_inicio + pd.offsets.BDay(MC_HORIZONTE_LONGO)

    # Mediana sobre a superficie
    mediana_ret = ((np.median(caminhos, axis=0) - 1) * 100).tolist()
    mediana_dias = list(range(1, MC_HORIZONTE_LONGO + 1))
    dias_kde_arr = np.array(dias_kde)
    mediana_z = np.empty(MC_HORIZONTE_LONGO)
    for d in range(MC_HORIZONTE_LONGO):
        kde_idx = np.argmin(np.abs(dias_kde_arr - d))
        mediana_z[d] = np.interp(mediana_ret[d], y_grid, Z_surface[kde_idx, :]) + 0.002

    chart_data = {
        "x_dias": X_dias,
        "y_grid": np.round(y_grid, 4).tolist(),
        "z_surface": np.round(Z_surface.T, 6).tolist(),
        "mediana_dias": mediana_dias,
        "mediana_ret": [round(v, 4) for v in mediana_ret],
        "mediana_z": np.round(mediana_z, 6).tolist(),
        "titulo": "Monte Carlo 3D: Distribuicao de Retornos",
        "subtitulo": (f"{N_SIMULACOES} cenarios | "
                      f"{mc_data_inicio.strftime('%d/%m/%Y')} ~ "
                      f"{mc_data_fim.strftime('%d/%m/%Y')}"),
    }

    caminho_data_js = os.path.join(dir_plots, "js", "chart-data.js")
    os.makedirs(os.path.dirname(caminho_data_js), exist_ok=True)
    with open(caminho_data_js, "w", encoding="utf-8") as f:
        f.write("// chart-data.js — Gerado por 04_monte_carlo_opt.py\n")
        f.write("var CHART_DATA = ")
        json.dump(chart_data, f, separators=(",", ":"))
        f.write(";\n")

    print(f"Dados 3D exportados para: {caminho_data_js}")
    print(f"Abra outputs/plots/monte_carlo_3d.html no navegador.")

except Exception as e:
    print(f"Aviso: nao foi possivel gerar dados 3D: {e}")


# Resumo final
print(f"\n{'='*50}")
print(f"PIPELINE COMPLETO")
print(f"{'='*50}")
print(f"Portfolio acumulado:  {df_backtest['acum_portfolio'].iloc[-1]:.4f}x")
print(f"IBOV acumulado:       {df_backtest['acum_ibov'].iloc[-1]:.4f}x")
print(f"Alpha (excesso):      {df_backtest['acum_portfolio'].iloc[-1] - df_backtest['acum_ibov'].iloc[-1]:.4f}x")
