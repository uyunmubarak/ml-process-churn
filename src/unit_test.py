from data_collection import split_data
import utils
from utils import load_config, pickle_dump, pickle_load
import pandas as pd
import numpy as np

def test_split_data():
    config = utils.load_config()

    X_columns = config["predictors"]
    y_columns = config["label"]

    # arrange
    ## make data with predictors and target column and 10 rows
    mock_X = {k:[i for i in range(10)] for k in X_columns}
    mock_y = [i for i in np.random.randint(0,2,10)]
    mock_X = pd.DataFrame(mock_X)
    mock_y = pd.DataFrame(mock_y, columns=y_columns)
    mock_df = pd.DataFrame(pd.concat([mock_X, mock_y], axis=1))

    # act
    x_train, _, _, _, _, _ = split_data(config, mock_df)

    # assert
    assert x_train.shape[0] == 6