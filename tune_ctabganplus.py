from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics
from sdmetrics.single_table import LinearRegression, MLPRegressor
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sdv.evaluation import evaluate
import os
import json
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast

num_exp = 1
dataset = "nike"
real_path = "Real_Datasets/nike_short_processed.csv"
fake_file_root = "Fake_Datasets"
ds_name = "nike"
prefix = "ctabganplus"
parent_path = Path(f'exp/{ds_name}/')

def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def objective(trial):

    data = pd.read_csv(real_path)

    # Hyperparameters
    nr_layers = trial.suggest_categorical("nr_layers", [2, 3, 4, 5, 6])
    dims = trial.suggest_categorical("dims", [64, 128, 256, 512])
    random_dim = trial.suggest_categorical("random_dim", [100, 150, 200])
    num_channels = trial.suggest_categorical("num_channels", [16, 32, 64, 128])
    l2scale = trial.suggest_float("l2scale", 0.00001, 0.00005, step=0.00001)
    batch_size = trial.suggest_categorical("batch_size", [100, 200, 500])
    epochs = trial.suggest_categorical("epochs", [300, 500, 1000, 2000])

    trial.set_user_attr("class_dim", [dims]* nr_layers)
    trial.set_user_attr("random_dim", random_dim)
    trial.set_user_attr("num_channels", num_channels)
    trial.set_user_attr("l2scale", l2scale)
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("epochs", epochs)

    synthesizer =  CTABGAN(raw_csv_path = real_path,
                     test_ratio = 0.20,
                     categorical_columns = [], #["cal_holiday", "weather_snow_1h", "weather_main", "weather_description", "hour"],
                     log_columns = [],
                     mixed_columns= {},
                     general_columns= [],
                     non_categorical_columns= ["date", "value", "GPD_per_capita", "eu_prod_index"], #["date", "weather_temp", "weather_rain_1h",  "traffic_volume"],
                     integer_columns = ["date"], # ["date", "weather_clouds_all"],
                     problem_type= {"Regression": "value"}, #{"Regression": "traffic_volume"},
                   class_dim=trial.user_attrs["class_dim"],
                   random_dim=trial.user_attrs["random_dim"],
                   num_channels=trial.user_attrs["num_channels"],
                   l2scale=trial.user_attrs["l2scale"],
                   batch_size=trial.user_attrs["batch_size"],
                   epochs=trial.user_attrs["epochs"]
                           )

    synthesizer.fit()

    metadata = {
            "tables":{
                "nike": {
                    "fields": {
                        "date": {
                            "type": "numerical",
                            "subtype": "integer"
                        },
                        "value": {
                            "type": "numerical",
                            "subtype": "float"
                        },
                        "GPD_per_capita": {
                            "type": "numerical",
                            "subtype": "float"
                        },
                        "eu_prod_index": {
                            "type": "numerical",
                            "subtype": "float"
                        }
                    }
                }

            }
            
    }

    #metadata = {
    #        "tables": {
    #            "metro": {
    #                "fields": {
    #                    "date": {
    #                        "type": "numerical",
    #                        "subtype": "integer"
    #                    },
    #                    "cal_holiday": {
    #                        "type": "categorical"
    #                    },
    #                    "weather_temp": {
    #                        "type": "numerical",
    #                        "subtype": "float"
    #                    },
    #                    "weather_rain_1h": {
    #                        "type": "numerical",
    #                        "subtype": "float"
    #                    },
    #                    "weather_snow_1h": {
    #                        "type": "categorical"
    #                    },
    #                    "weather_clouds_all": {
    #                        "type": "numerical",
    #                        "subtype": "integer"
    #                    },
    #                    "weather_main": {
    #                        "type": "categorical"
    #                    },
    #                    "weather_description": {
    #                        "type": "categorical"
    #                    },
    #                    "traffic_volume": {
    #                        "type": "numerical",
    #                        "subtype": "float"
    #                    },
    #                    "hour": {
    #                        "type": "categorical"
    #                    }
    #                }
    #            }  
    #        }
    #    }
    new_data = synthesizer.generate_samples()
    
    print(new_data.head())
    print(len(new_data))
    # Convert number of days to date
    # new_data["date"] = pd.to_datetime(new_data["date"], unit="d", errors='coerce')

    # Create datetime column

    # notna_msk = new_data['date'].notna()
    # new_data.loc[notna_msk, 'date'] = new_data.loc[notna_msk, 'date'].dt.strftime('%Y-%m-%d') + ' ' + new_data.loc[notna_msk, 'hour'].astype(str) + ':00:00'

    # Convert datetime column to datetime format
    # new_data['date'] = pd.to_datetime(new_data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # df.drop_duplicates(subset=["date"], inplace=True)
    # new_data.drop('hour', axis=1, inplace=True)
    # Sort by date
    # new_data.dropna(subset=['date'], inplace=True)
    # new_data.sort_values(by=['date'], inplace=True)
    
    # new_data.dropna(inplace=True)
    #data.dropna(subset=['traffic_volume'], inplace =True)

    # score = MLPRegressor.compute(
    #        test_data=data,
    #        train_data=new_data,
    #    target='traffic_volume'
    #)
    score = evaluate(synthetic_data={'nike': new_data}, real_data={'nike': data}, metadata=metadata)
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)
best_trial = study.best_trial

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')
dump_json(best_trial.user_attrs, parent_path / f'{prefix}_best/parameters.json')

# print(best_trial)
print(best_trial.user_attrs)
