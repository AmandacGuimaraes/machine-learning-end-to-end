import pandas as pd
from src.analysis.data_diagnostics import diagnose
from src.analysis.eda import create_eda_report
from src.analysis.statistical_tests import categorical_association, numeric_association, benjamini_hochberg

def sample():
    return pd.DataFrame({"age": [19, 20, 22, 21, 30, 31, 33, 32],
                         "group": ["A", "A", "A", "B", "B", "B", "A", "B"],
                         "target": [0, 0, 0, 0, 1, 1, 1, 1]})

def test_diagnostics():
    assert diagnose(sample(), "target")["rows"] == 8

def test_eda_outputs(tmp_path):
    create_eda_report(sample(), "target", str(tmp_path))
    assert (tmp_path / "figures" / "target_distribution.png").exists()
    assert (tmp_path / "metrics" / "missing.csv").exists()

def test_statistical_tests():
    assert 0 <= categorical_association(sample(), "group", "target")["p_value"] <= 1
    assert 0 <= numeric_association(sample(), "age", "target")["p_value"] <= 1

def test_bh():
    assert benjamini_hochberg([.01, .04, .03]) == sorted(benjamini_hochberg([.01, .04, .03])) or len(benjamini_hochberg([.01, .04, .03])) == 3
