import numpy as np

def mae(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def wape(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float(np.nan)
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)

def smape(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)

def compute_metrics(y_true, y_pred):
    return {"MAE": mae(y_true,y_pred), "WAPE": wape(y_true,y_pred), "sMAPE": smape(y_true,y_pred)}
