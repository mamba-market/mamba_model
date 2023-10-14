"""This module generates dummy data for model construction"""
from typing import Union, List
import hydra
from omegaconf import DictConfig
import numpy
import random
import pandas
import logging
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="configs", config_name="dummy_data")
def main(cfg: DictConfig):
    n_rows = cfg.n_rows
    n_players = cfg.n_players
    score_ids = cfg.score_ids
    categorical_features = cfg.categorical_features
    numerical_features = cfg.numerical_features
    response = cfg.response

    n_rows = 1200
    n_players = 10
    score_ids = ['A', 'B', 'C', 'D', 'E']
    categorical_features = {'home_guest': ["Home", "Guest"],
                            'venue_type': ["Hard", "Soft"],
                            'weather': ["Rainy", "Cloudy", "Sunny", "Windy"]}
    numerical_features = {
    'num_features_1': [0.0, 1.0],
    'num_features_2': [0.0, 1.0],
    'num_features_3': [0.0, 1.0] }
    response = 'score_val'

    print(categorical_features)
    dfs = []
    dfs.append(pandas.DataFrame({'player': random.choices(list(range(n_players)), k=n_rows)}))
    dfs.append(pandas.DataFrame({'score_id': random.choices(list(range(len(score_ids))), k=n_rows)}))
    for k, v in categorical_features.items():
        dfs.append(pandas.DataFrame({k: random.choices(list(range(len(v))), k=n_rows)}))
    for k, v in numerical_features.items():
        dfs.append(create_dummy_data_with_discrete_inputs(v, n_rows, prefix=k, categorical_var=False))
    dfs.append(pandas.DataFrame({response: numpy.random.random(n_rows)}))

    dataframe = pandas.concat(dfs, axis=1)
    dataframe.to_pickle(f"{cfg.ouput_fp}")



def create_dummy_data_with_discrete_inputs(variable: Union[List, int], n_rows: int,
                                           prefix: str=None, categorical_var: bool=True) -> pandas.DataFrame:
    if categorical_var:
        if type(variable) == int:
            df = pandas.Series(list(range(variable)))
        else:
            df = pandas.Series(variable)
        if prefix is not None:
            df = pandas.get_dummies(df, prefix=prefix)
        else:
            df = pandas.get_dummies(df)
    else:
        assert type(variable) == list
        assert prefix is not None
        df = pandas.DataFrame({prefix: list(numpy.random.uniform(low=variable[0], high=variable[1], size=n_rows))})
    df = shuffle(pandas.concat([df] * int(n_rows / len(df))))
    df.reset_index(inplace=True, drop=True)
    return df


if __name__ == "__main__":
    main()


