import os
import pandas as pd
from pathlib import Path
import json
from model.ctabgan import CTABGAN
try:
    from typing import get_args, get_origin
except ImportError:
    from typing_extensions import get_args, get_origin
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast

def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)

ds_name = "nike"
prefix = "ctabganplus_best"

parent_path = Path(f'exp/{ds_name}/{prefix}/')
real_data_path = "Real_Datasets/nike_short_processed.csv"

data = pd.read_csv(real_data_path, parse_dates=['date'])
data = data.dropna(subset=['value'])

parameters = load_json(os.path.join(parent_path, 'parameters.json'))

def train_sample_ctabganp():
    synthesizer = CTABGAN(raw_csv_path = real_data_path,
                         test_ratio = 0.20,
                         categorical_columns = [], #["cal_holiday", "weather_snow_1h", "weather_main", "weather_description", "hour"],
                         log_columns = [],
                         mixed_columns= {},
                         general_columns= [],
                         non_categorical_columns= ["date", "value", "GPD_per_capita", "eu_prod_index"], #["date", "weather_temp", "weather_rain_1h",  "traffic_volume"],
                         integer_columns = ["date"], # ["date", "weather_clouds_all"],
                         problem_type= {"Regression": "value"}, #{"Regression": "traffic_volume"},
                         **parameters
                        )
    synthesizer.fit()
    new_data = synthesizer.generate_samples(num_samples=data.shape[0])
    # Sort by date
    new_data.sort_values(by=['date'], inplace=True)

    # Save dataframe as csv
    new_data.to_csv(os.path.join(parent_path, 'synthetic_ctabganp.csv'), index=False)

if __name__ == "__main__":
    train_sample_ctabganp()
