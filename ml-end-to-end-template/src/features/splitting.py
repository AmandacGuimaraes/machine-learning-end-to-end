from sklearn.model_selection import train_test_split

def split_data(df, config):
    target = config["data"]["target"]
    X = df.drop(columns=[target])
    for col in (config["data"].get("id_column"), config["data"].get("date_column")):
        if col and col in X:
            X = X.drop(columns=[col])
    y = df[target]
    validation = config["validation"]
    strategy = validation.get("strategy", "random")
    if strategy == "temporal":
        date_col = config["data"].get("date_column")
        if not date_col or date_col not in df:
            raise ValueError("Temporal exige data.date_column")
        order = df[date_col].sort_values().index
        cut = int(len(order) * (1 - validation["test_size"]))
        return X.loc[order[:cut]], X.loc[order[cut:]], y.loc[order[:cut]], y.loc[order[cut:]]
    if strategy not in {"random", "stratified"}:
        raise ValueError("Estratégia não implementada: use random, stratified ou temporal")
    return train_test_split(
        X, y, test_size=validation["test_size"],
        random_state=validation["random_state"],
        stratify=y if strategy == "stratified" else None,
    )
