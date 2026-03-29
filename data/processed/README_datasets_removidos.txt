Descrição dos arquivos removidos por tamanho:

1. master_dataset_ml.csv (204.569 KB)
- Este é o dataset principal em formato "long" para machine learning, contendo as features calculadas para cada ativo em cada dia (ex: volatilidade, momentum, distâncias para médias móveis, RSI, etc.) e os targets de retorno futuro. Ele serve como base para treinar e testar modelos preditivos.
- É gerado pelo script limpeza_dados.py e utilizado como insumo para o enriquecimento com regimes de mercado no script hmm_regimes.py.

2. master_dataset_ml_enriched.csv (429.870 KB)
- É uma versão enriquecida do master_dataset_ml.csv, incluindo colunas adicionais como o regime de mercado (detectado pelo HMM) e targets binários (ex: se o ativo subiu ou caiu no dia seguinte). Esse arquivo é usado para treinar modelos que consideram também o contexto de regime de mercado.
- É gerado pelo script hmm_regimes.py e utilizado como entrada principal para o script 03_xgb_prediction.py, que treina o modelo XGBoost.

Motivo da remoção:

3. precos_limpos_ffill.csv
- Preços ajustados dos ativos e IBOV, após limpeza, preenchimento de gaps e remoção de penny stocks. Gerado por limpeza_dados.py. Usado como base para cálculo de retornos e features.

4. retornos_log_brutos.csv
- Retornos logarítmicos brutos dos ativos e IBOV, calculados a partir de precos_limpos_ffill.csv. Gerado por limpeza_dados.py. Usado para análise de outliers e validações.

5. retornos_log_limpos.csv
- Retornos logarítmicos dos ativos, após limpeza e remoção de colunas problemáticas. Gerado por limpeza_dados.py. Usado para cálculo de features e validações.

6. retornos_limpos_final.csv
- Retornos logarítmicos dos ativos e IBOV, após winsorização (remoção de outliers extremos). Gerado por limpeza_dados.py. Usado como insumo para o cálculo de métricas, backtest e detecção de regimes.

7. expected_returns.csv
- Previsões de retorno dos ativos, geradas pelo modelo XGBoost em 03_xgb_prediction.py. Usado como entrada para o backtest e otimização de portfólio em 04_monte_carlo_opt.py.

Passo a passo para regenerar os arquivos:
1. Execute limpeza_dados.py para gerar precos_limpos_ffill.csv, retornos_log_brutos.csv, retornos_log_limpos.csv e retornos_limpos_final.csv, além de master_dataset_ml.csv.
2. Execute hmm_regimes.py para gerar master_dataset_ml_enriched.csv e regimes_hmm_ibov.csv.
3. Execute 03_xgb_prediction.py para gerar expected_returns.csv.
4. Execute 04_monte_carlo_opt.py para gerar os resultados finais e outputs.

Motivo da remoção:
- Os arquivos acima foram removidos do repositório devido ao espaço ocupado, para facilitar o envio e compartilhamento do projeto. Todos podem ser recriados automaticamente ao rodar o pipeline completo, desde que os dados brutos estejam presentes em data/raw/.

Orientação:
- Para reproduzir o pipeline, basta executar os scripts na ordem indicada acima. Os arquivos serão gerados novamente em data/processed/ conforme o fluxo de processamento.

Orientação:
- Para reproduzir o pipeline, basta executar os scripts na ordem indicada no README. Os arquivos serão gerados novamente em data/processed/ conforme o fluxo de processamento.