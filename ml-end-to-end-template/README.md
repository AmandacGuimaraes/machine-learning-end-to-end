# Machine Learning End-to-End — Template

Template Python 3.12 para classificação e regressão tabular. Não exige dataset real no repositório.

## Instalação (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Demonstração com dados sintéticos
```powershell
python -m scripts.generate_sample_data --task classification
python -m scripts.run_eda --config configs/classification.yaml
python -m scripts.run_training --config configs/classification.yaml
python -m pytest -v
uvicorn src.api.main:app --reload
```
API: GET /health, GET /ready, POST /predict (JSON `{"records":[{"feature_0":1.0,...}]}`).

## Novo projeto
1. Adicione CSV em `data/raw/` (não versionado).
2. Copie `configs/classification.yaml` ou `configs/regression.yaml` e ajuste `data.path` e `data.target`.
3. Escolha divisão `random`, `stratified` ou `temporal`; para temporal configure `data.date_column`.
4. Execute EDA e treinamento com `--config`.
5. Revise as métricas e os riscos de vazamento antes de considerar deploy.

## Extras opcionais
- `pip install -e ".[automl]"` e `model.automl: true` para AutoGluon
- `pip install -e ".[explain]"` para futuras análises SHAP
- `pip install -e ".[mlops]"` para integração MLflow/Evidently

## Limitações
Este é um ponto de partida executável, não uma plataforma pronta para produção.
- Otimização usa validação cruzada simples: para séries temporais e grupos, adapte o CV.
- A inferência usa modelo joblib local e não implementa autenticação, limites de taxa nem registro central de modelos.
- O monitoramento implementa teste KS numérico; variáveis categóricas, atrasos de rótulos e regras de alerta exigem implementação específica.
- SHAP e MLflow estão previstos como integrações opcionais, não habilitados automaticamente.
- Não carregue arquivos joblib de origem desconhecida.
- Não use o teste independente repetidamente para selecionar features ou hiperparâmetros.
