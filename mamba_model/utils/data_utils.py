from mamba_model.utils.db import model_data_db, model_data_engine
import pandas as pd

def get_data_from_db(player_data_threshold: int = 10):
    query = f"""
    select ar.* from athlete_records ar
    join athlete_record_count arc on arc.player_id = ar.player_id
    where arc.record_count >= {player_data_threshold}; 
    """
    df = pd.read_sql(query, model_data_engine)
    return df