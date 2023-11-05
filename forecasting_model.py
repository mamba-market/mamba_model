"""Forecasting model, facebook"""
import os
import numpy
import pandas
import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig
from prophet import Prophet



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    inference_input_fp = cfg.inference_input_fp
    data_type = str(cfg.training_input_fp).strip("_cricket_training_data.csv").strip("_cricket_training_data").strip(
        "data/")
    data_type = 'ODI'
    training_input_fp = f"data/{data_type}_sampled_cricket_training_data_tz_{cfg.target_lower_limit}_{cfg.target_upper_limit}.csv"

    inference_output_fp = f"inference_results/game_preds_forecast_{datetime.strftime(datetime.utcnow(), '%Y-%m-%d')}.csv"
    inference_input = pandas.read_csv(inference_input_fp)
    training_df = pandas.read_csv(training_input_fp)
    training_df['start_time'] = training_df['start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
    player_ids_of_interest = inference_input['player_id'].unique()
    training_df = training_df.query("player_id in @player_ids_of_interest")

    training_df = training_df.groupby(list(cfg.stratefied_sampling_categories), as_index=False).agg({cfg.response: list,
                                                                                               'start_time': list})
    training_df.sort_values("start_time", inplace=True)
    training_df.reset_index(inplace=True, drop=True)
    training_df[f'forecasted_{cfg.response}'] = None
    print(set(training_df['player_id']))

    forecasts = []
    for i, row in training_df.iterrows():
        m = Prophet()
        # import ipdb; ipdb.set_trace()
        m.fit(pandas.DataFrame({"y": training_df.loc[i, cfg.response], "ds": training_df.loc[i, 'start_time']}))
        future = m.make_future_dataframe(periods=1, include_history=False)
        future['cap'] = 100
        future['floor'] = 0
        forecast = m.predict(future)
        forecast['player_id'] = training_df.loc[i, 'player_id']
        forecast['target_id'] = training_df.loc[i, 'target_id']
        forecasts.append(forecast)

    forcasts = pandas.concat(forecasts, axis=0)
    forcasts.to_csv(inference_output_fp)


if __name__ == '__main__':
    main()