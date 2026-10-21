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

# Ordem de desenvolvimento recomendada

Agora que temos a base, sugiro desenvolver cada módulo nesta sequência:

## Fase 1 — Estrutura, ingestão e baseline inicial

## Fase 2 — EDA aprofundada e testes estatísticos

Principais perguntas que queremos responder

1. Os dados possuem qualidade suficiente para modelagem?
2. Existem variáveis com valores ausentes ou distribuições problemáticas?
3. As classes estão balanceadas?
4. As variáveis apresentam associação estatística com o target?
5. Existem evidências de sinal preditivo?
6. Há indícios de vazamento de informação?
7. Quais hipóteses merecem ser investigadas na próxima fase?
Importante: associação estatística não garante capacidade preditiva. Precisaremos validar as descobertas em dados não utilizados para selecionar variáveis ou ajustar modelos.

## Fase 3 — Feature Engineering e validação

Objetivo: descobrir quais variáveis realmente contribuem para a previsão, construir atributos informativos e garantir que o processo de preparação não introduza vazamento de dados.
3.1 Qualidade e transformação das variáveis

1. Quais variáveis são relevantes para o problema?
2. Existem variáveis redundantes, constantes ou com baixa variabilidade?
3. Quais variáveis possuem alta correlação entre si?
4. Existem relações não lineares entre as features e o target?
5. Quais variáveis precisam de normalização ou padronização?
6. Como devemos tratar variáveis categóricas?
7. Existem valores ausentes que precisam de imputação?
8. Os outliers representam erros ou informações relevantes?
9. Existem variáveis com alta cardinalidade?
10. As transformações podem ser reproduzidas durante a inferência?
3.2 Criação e seleção de features
11. Podemos criar novas variáveis a partir das existentes?
12. Existem interações entre variáveis que aumentam a capacidade preditiva?
13. Variáveis temporais, agregações ou lags podem melhorar a previsão?
14. Quais features apresentam maior associação com o target?
15. Quais features possuem maior informação mútua (Mutual Information)?
16. A remoção de variáveis redundantes melhora a generalização?
17. Qual é o impacto da seleção de features no desempenho do modelo?
18. A importância das variáveis permanece estável entre diferentes amostras?
19. As features estarão disponíveis no momento real da previsão?
20. Existe alguma variável com potencial de data leakage?
3.3 Estratégia de validação
21. Qual estratégia de divisão é mais adequada: aleatória, temporal ou por grupos?
22. O conjunto de treino representa a população em que o modelo será utilizado?
23. Existe desbalanceamento entre as classes?
24. Precisamos utilizar StratifiedKFold, GroupKFold ou TimeSeriesSplit?
25. Como evitar vazamento de dados durante imputação, encoding e seleção de features?
26. Quantos folds são adequados para a validação cruzada?
27. Existe diferença significativa entre os conjuntos de treino e teste?
28. As transformações devem ser aplicadas dentro de um Pipeline do scikit-learn?
29. Como garantir a reprodutibilidade dos experimentos?
30. Como preservar um conjunto de teste independente para avaliação final?
Entregáveis da Fase 3: pipeline de pré-processamento, engenharia de atributos, estratégia de validação documentada, testes de integridade e relatório de seleção de features.

## Fase 4 — Modelagem, AutoGluon e otimização

Objetivo: identificar a abordagem de modelagem que apresenta o melhor equilíbrio entre desempenho, generalização, interpretabilidade e custo computacional.
4.1 Baseline e escolha dos algoritmos

1. Qual é o modelo mais simples capaz de resolver o problema?
2. Qual desempenho obtemos com um DummyClassifier ou DummyRegressor?
3. O problema realmente apresenta sinal preditivo?
4. Quais algoritmos são mais adequados ao tipo de dado?
5. Modelos lineares conseguem capturar as relações presentes?
6. Modelos baseados em árvores apresentam desempenho superior?
7. Existe vantagem em utilizar Random Forest, XGBoost, LightGBM ou CatBoost?
8. Os algoritmos estão sendo comparados sob as mesmas condições de validação?
9. O modelo apresenta sinais de overfitting ou underfitting?
10. O ganho de desempenho justifica o aumento de complexidade?
4.2 AutoGluon e AutoML
11. Qual é o desempenho do AutoGluon em comparação aos modelos tradicionais?
12. Quais modelos o AutoGluon selecionou?
13. O WeightedEnsemble melhora os resultados?
14. Qual é o impacto do ensemble no tempo de inferência?
15. O AutoGluon consegue trabalhar adequadamente com os dados categóricos e ausentes?
16. O AutoML está utilizando a métrica de avaliação apropriada?
17. Existe risco de overfitting durante a busca automatizada?
18. O modelo escolhido pelo AutoML é estável entre diferentes execuções?
19. Qual é o custo computacional do treinamento automatizado?
20. O AutoGluon oferece uma vantagem relevante para o problema?
4.3 Otimização de hiperparâmetros
21. Quais hiperparâmetros possuem maior influência no desempenho?
22. Devemos utilizar GridSearchCV, RandomizedSearchCV ou Optuna?
23. Qual é o orçamento computacional disponível?
24. Quantas tentativas de otimização são suficientes?
25. O tuning realmente melhora o baseline?
26. Os ganhos permanecem consistentes na validação cruzada?
27. Como evitar otimização excessiva sobre o conjunto de validação?
28. Como registrar e comparar os experimentos com MLflow?
29. Qual modelo apresenta o melhor equilíbrio entre qualidade e custo?
30. Qual modelo deve ser promovido como candidato à produção?
Entregáveis da Fase 4: leaderboard dos modelos, experimentos versionados, resultados de otimização, artefatos treinados e justificativa técnica da escolha do modelo.

## Fase 5 — Explicabilidade e avaliação robusta

Objetivo: demonstrar que o modelo é confiável, compreender suas decisões, identificar limitações e verificar se seu desempenho se mantém em diferentes condições.
5.1 Avaliação de desempenho

1. Qual é o desempenho do modelo no conjunto de teste independente?
2. Quais métricas são mais relevantes para o problema de negócio?
3. Accuracy é suficiente ou precisamos priorizar Precision, Recall, F1 ou ROC-AUC?
4. Em problemas desbalanceados, a PR-AUC é mais informativa?
5. Quais erros são mais custosos: falsos positivos ou falsos negativos?
6. Qual é o melhor limiar de decisão para o objetivo do negócio?
7. As probabilidades previstas estão bem calibradas?
8. Qual é a variabilidade das métricas entre folds?
9. Quais são os intervalos de confiança das principais métricas?
10. O desempenho observado é superior ao baseline de forma relevante?
5.2 Explicabilidade
11. Quais variáveis mais influenciam as previsões?
12. O que o SHAP revela sobre o comportamento global do modelo?
13. Por que o modelo produziu determinada previsão individual?
14. As importâncias identificadas pelo SHAP e pela Permutation Importance são consistentes?
15. Existem relações não lineares ou interações importantes?
16. O comportamento do modelo está alinhado ao conhecimento do domínio?
17. O modelo depende excessivamente de uma única variável?
18. Existem variáveis que funcionam como proxies de atributos sensíveis?
19. A explicação é estável em diferentes amostras?
20. Conseguimos comunicar os resultados para pessoas não técnicas?
5.3 Robustez, generalização e análise de erros
21. Em quais segmentos o modelo apresenta mais erros?
22. Existem diferenças relevantes de desempenho entre grupos?
23. O modelo mantém desempenho em dados de períodos diferentes?
24. Como reage a valores ausentes ou distribuições diferentes das observadas no treino?
25. Pequenas alterações nas features provocam grandes mudanças nas previsões?
26. O modelo apresenta dependência de padrões espúrios?
27. Há sinais de data leakage que passaram despercebidos?
28. O desempenho é consistente em diferentes sementes e amostras?
29. Quais limitações impedem ou restringem o uso em produção?
30. O modelo atende aos critérios técnicos e de negócio para aprovação?
Entregáveis da Fase 5: relatório de avaliação final, matriz de confusão, curvas de desempenho, gráficos SHAP, análise de erros, testes de robustez e documentação das limitações.

Um cuidado importante: a escolha de hiperparâmetros e limiares deve ocorrer antes da avaliação final no teste independente.

## Fase 6 — API, deploy, monitoramento e MLOps

Objetivo: transformar o modelo treinado em uma solução utilizável, escalável, observável e passível de manutenção.
6.1 API e inferência

1. Como disponibilizar o modelo para outros sistemas?
2. Devemos utilizar inferência batch, online ou streaming?
3. Quais endpoints serão necessários?
4. Como validar os dados recebidos pela API?
5. Como garantir que o pré-processamento da inferência seja idêntico ao do treinamento?
6. Como lidar com valores ausentes, tipos inválidos e categorias desconhecidas?
7. Qual deve ser o formato das respostas?
8. Como tratar erros e exceções?
9. Qual é a latência máxima aceitável?
10. Como proteger a API com autenticação e autorização?
Ferramentas: FastAPI, Pydantic, pytest e Docker.
6.2 Deploy e infraestrutura
11. O modelo será implantado em Azure, GCP, AWS ou infraestrutura local?
12. Devemos utilizar containers Docker?
13. Como configurar variáveis de ambiente e segredos?
14. Como versionar os artefatos de modelos?
15. Como garantir a reprodutibilidade do ambiente?
16. Quais recursos de CPU e memória são necessários?
17. Como escalar a solução conforme o volume de requisições?
18. Como controlar custos de infraestrutura?
19. Como implementar health checks?
20. Como realizar rollback se uma nova versão apresentar problemas?
6.3 CI/CD e governança
21. Quais testes devem ser executados automaticamente?
22. Como configurar GitHub Actions?
23. Como validar código, dependências e imagens Docker?
24. Como automatizar a publicação de novas versões?
25. Como registrar experimentos, parâmetros e métricas?
26. Como garantir rastreabilidade entre dados, código e modelo?
27. Quais critérios devem autorizar a promoção de um modelo?
28. Como documentar as versões em produção?
29. Como gerenciar permissões e acesso aos dados?
30. Como garantir conformidade com requisitos de privacidade e governança?
6.4 Monitoramento e evolução
31. Como identificar mudanças na distribuição dos dados (data drift)?
32. Como detectar mudanças na relação entre features e target (concept drift)?
33. Como acompanhar a performance quando os rótulos reais estiverem disponíveis?
34. Quais métricas operacionais devemos monitorar?
35. Como detectar aumento de latência ou falhas na API?
36. Como identificar dados inválidos durante a inferência?
37. Quais condições devem gerar alertas?
38. Quando devemos retreinar o modelo?
39. Como comparar um novo modelo com a versão em produção?
40. Como verificar se o retreinamento realmente trouxe melhorias?
Ferramentas: MLflow, Evidently, GitHub Actions, Docker e soluções de observabilidade.

Entregáveis da Fase 6: API funcional, Dockerfile, testes de integração, pipeline CI/CD, versionamento de modelos, relatórios de monitoramento e estratégia de retreinamento.

# Estrutura recomendada do código-fonte

machine-learning-end-to-end/
│
├── .github/workflows/
│   └── ci.yml
│
├── configs/
│   ├── default.yaml
│   └── examples/
│       ├── classification.yaml
│       └── regression.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
│
├── src/
│   ├── config.py
│   │
│   ├── data/                   # FASE 1
│   │   ├── ingestion.py
│   │   └── validation.py
│   │
│   ├── analysis/               # FASE 2
│   │   ├── eda.py
│   │   ├── statistical_tests.py
│   │   └── diagnostics.py
│   │
│   ├── features/               # FASE 3
│   │   ├── engineering.py
│   │   ├── selection.py
│   │   └── splitting.py
│   │
│   ├── modeling/               # FASE 4
│   │   ├── baseline.py
│   │   ├── train.py
│   │   ├── automl.py
│   │   └── optimization.py
│   │
│   ├── evaluation/             # FASE 5
│   │   ├── metrics.py
│   │   ├── explainability.py
│   │   └── robustness.py
│   │
│   ├── monitoring/             # FASE 6
│   │   └── drift.py
│   │
│   └── api/
│       ├── main.py
│       └── schemas.py
│
├── scripts/
│   ├── generate_sample_data.py
│   ├── run_eda.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_monitoring.py
│
├── tests/
├── models/
├── reports/
├── Dockerfile
├── pyproject.toml
├── .gitignore
└── README.md
flowchart TD
    A["Configuração YAML + Dataset"] --> B["Fase 1: Ingestão e validação"]
    B --> C["Fase 2: EDA e testes estatísticos"]
    C --> D["Fase 3: Feature Engineering"]
    D --> E["Fase 4: Baseline + ML + AutoGluon"]
    E --> F["Fase 5: Avaliação + SHAP + Robustez"]
    F --> G{"Aprovado para produção?"}
    G -->|Sim| H["Fase 6: API + Docker + CI/CD"]
    G -->|Não| D
    H --> I["Monitoramento: Drift + Performance"]
    I --> J{"Retreinamento necessário?"}
    J -->|Sim| D
    J -->|Não| I
