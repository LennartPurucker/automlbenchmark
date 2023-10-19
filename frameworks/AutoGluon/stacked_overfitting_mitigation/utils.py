import pandas as pd

def get_label_train_data(fit_para, predictor_para):
    fit_para = fit_para.copy()
    train_data = pd.read_parquet(fit_para.get("train_data"))
    label = predictor_para["label"]
    fit_para.pop("train_data")

    return train_data, label, fit_para
