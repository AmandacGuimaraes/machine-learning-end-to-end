import pandas as pd
from src.features.engineering import make_preprocessor

def test_preprocessor():
    X = pd.DataFrame({"age": [20, None, 30], "group": ["a", "b", "a"]})
    assert make_preprocessor(X).fit_transform(X).shape[0] == 3
