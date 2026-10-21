from pathlib import Path
import yaml

def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Configuração YAML inválida")
    return config

def validate_config(config):
    task = config["project"]["task"]
    if task not in {"classification", "regression"}:
        raise ValueError("task deve ser classification ou regression")
    if not config["data"].get("path") or not config["data"].get("target"):
        raise ValueError("Configure data.path e data.target")
    return config
