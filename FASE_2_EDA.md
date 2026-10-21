# Fase 2 — EDA e testes estatisticos

## Preparacao (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Coloque o CSV em `data/raw/dataset.csv` e edite `configs/config.yaml` para apontar para o caminho e o nome real do target.

## Execucao

```powershell
python -m scripts.run_eda
python -m pytest -v
```

Saidas em `reports/metrics/` e `reports/figures/`.

## Cuidados metodologicos

- Realize testes de associacao exploratorios preferencialmente no conjunto de treino, sem consultar o teste final.
- Valores-p ajustados usam Benjamini-Hochberg; interpretar tamanho de efeito, relevancia pratica e pressupostos.
- Variaveis identificadoras e variaveis indisponiveis no instante de inferencia precisam ser revisadas manualmente.
- Mann-Whitney e Kruskal-Wallis avaliam diferencas nas distribuicoes sob suas respectivas hipoteses, nao demonstram causalidade.
- Para alvos continuos, adapte a bateria de testes: a rotina atual foi desenhada para classificacao.
