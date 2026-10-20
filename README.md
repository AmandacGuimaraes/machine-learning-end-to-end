# Machine Learning End-to-End

Template profissional para classificacao binaria tabular, com EDA, validacao, baseline e testes.

## Instalacao (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Dados
Coloque seu CSV em `data/raw/dataset.csv`. Configure em `configs/config.yaml` o nome da coluna-alvo `target` e a `positive_class` (valor real da classe positiva, por exemplo `1` ou `Alto desempenho`). Verifique a definicao da classe antes de executar.

## Execucao
```powershell
python -m pytest -q
python -m scripts.run_pipeline --config configs/config.yaml
```

Resultados: `reports/figures/target_distribution.png`, `reports/metrics/baseline.json` e `models/baseline.joblib`.

## Importante
- O baseline usa divisao estratificada aleatoria; para dados temporais ou grupos repetidos, substitua por validacao adequada.
- Nunca use informacoes posteriores ao momento da previsao.
- O EDA inicial e descritivo; testes estatisticos exigem hipoteses e verificacao de premissas.
- `src/api/main.py` e `src/monitoring/drift.py` sao pontos de extensao; nao representam deploy e monitoramento completos.
- AutoGluon e opcional: `pip install -e ".[automl]"` em ambiente compativel. Nao faz parte do pipeline inicial.
- O Dockerfile e um esqueleto para evolucao; dados sao montados ou fornecidos em tempo de execucao.

## Publicacao no GitHub
Crie um repositorio vazio no GitHub e execute:
```powershell
git init
git add .
git commit -m "feat: estrutura inicial do pipeline de machine learning"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/machine-learning-end-to-end.git
git push -u origin main
```
