import pandas as pd
import copy
from sklearn.model_selection import train_test_split
import utils
from utils import load_config, pickle_dump

config = load_config()

def read_raw_df(raw_data_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_data_path, index_col=0)

    df.drop(config["drop_columns"], axis=1, inplace=True)

    return df

def split_data(dataframe: pd.DataFrame)->pd.DataFrame:
    df = copy.deepcopy(dataframe)

    x = df[config["predictors"]]
    y = df[config["label"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 42,
        stratify = y)

    x_valid, x_test, y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = config["valid_size"],
        random_state = 42,
        stratify = y_test
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def main():
    df = read_raw_df(config["raw_dataset_path"])

    pickle_dump(df, config["raw_df_path"])


    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(df)

    # reset index
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    x_valid = x_valid.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)

    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    pickle_dump(x_train, config["train_set_path"][0])
    pickle_dump(y_train, config["train_set_path"][1])

    pickle_dump(x_valid, config["valid_set_path"][0])
    pickle_dump(y_valid, config["valid_set_path"][1])

    pickle_dump(x_test, config["test_set_path"][0])
    pickle_dump(y_test, config["test_set_path"][1])


if __name__ == "__main__":

    main()