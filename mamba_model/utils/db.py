from devtools.db import get_engine, MambaDB

model_data_engine = get_engine("model_data")
model_data_db = MambaDB("model_data")
