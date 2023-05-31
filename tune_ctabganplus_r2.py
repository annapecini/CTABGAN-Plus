from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
import os
import json
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast
import glob

num_exp = 1
dataset = "Metro_Interstate_Traffic_Volume_short"
real_path = "Real_Datasets/Metro_Interstate_Traffic_Volume_short_processed.csv"
fake_file_root = "Fake_Datasets"
ds_name = "metro"
prefix = "ctabganplus"
parent_path = Path(f'exp/{ds_name}/')

def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')

def objective(trial):

    data = pd.read_csv("Real_Datasets/Metro_Interstate_Traffic_Volume_short_processed.csv")
    # Hyperparameters
    nr_layers = trial.suggest_categorical("nr_layers", [2, 3, 4, 5, 6])
    dims = trial.suggest_categorical("dims", [64, 128, 256, 512])
    random_dim = trial.suggest_categorical("random_dim", [100, 150, 200])
    num_channels = trial.suggest_categorical("num_channels", [16, 32, 64, 128])
    l2scale = trial.suggest_float("l2scale", 0.00001, 0.00005, step=0.00001)
    batch_size = trial.suggest_categorical("batch_size", [100, 200, 500])
    epochs = trial.suggest_categorical("epochs", [1, 3, 5, 10, 2])

    trial.set_user_attr("class_dim", [dims] * nr_layers)
    trial.set_user_attr("random_dim", random_dim)
    trial.set_user_attr("num_channels", num_channels)
    trial.set_user_attr("l2scale", l2scale)
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("epochs", epochs)

    synthesizer = CTABGAN(raw_csv_path=real_path,
                          test_ratio=0.20,
                          categorical_columns=["cal_holiday", "weather_main", "weather_description", "hour"],
                          log_columns=[],
                          mixed_columns={},
                          general_columns=["date", "weather_temp", "weather_rain_1h", "weather_snow_1h", "weather_clouds_all", "traffic_volume"],
                          non_categorical_columns=[],
                          integer_columns=[],
                          problem_type={"Regression": "traffic_volume"},
                          class_dim=trial.user_attrs["class_dim"],
                          random_dim=trial.user_attrs["random_dim"],
                          num_channels=trial.user_attrs["num_channels"],
                          l2scale=trial.user_attrs["l2scale"],
                          batch_size=trial.user_attrs["batch_size"],
                          epochs=trial.user_attrs["epochs"]
                          )

    synthesizer.fit()
    syn = synthesizer.generate_samples()
    fake_path = fake_file_root + "/" + dataset + "/" + dataset + f"_fake_{prefix}_ctabplus.csv"
    syn.to_csv(fake_path, index=False)

    # fake_paths = glob.glob(fake_file_root + "/" + dataset + "/" + "*")

    # model_dict = {"Regression": ["l_reg", "ridge", "lasso", "B_ridge"]}
    # result_mat = get_utility_metrics(real_path, fake_paths, "MinMax", model_dict, test_ratio=0.20)
    print(stat_sim(real_path, fake_path, cat_cols=["cal_holiday", "weather_main", "weather_description", "hour"]))

    return 0

    
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2, show_progress_bar=True)
best_trial = study.best_trial

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')
dump_json(best_trial.user_attrs, parent_path / f'{prefix}_best/parameters.json')

# print(best_trial)
print(best_trial.user_attrs)
