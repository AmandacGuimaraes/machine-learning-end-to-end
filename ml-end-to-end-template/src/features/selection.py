from sklearn.feature_selection import SelectPercentile, mutual_info_classif, f_regression

def feature_selector(task, percentile=100):
    score = mutual_info_classif if task == "classification" else f_regression
    return SelectPercentile(score_func=score, percentile=percentile)
