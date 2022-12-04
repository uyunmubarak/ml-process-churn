import pandas as pd
import numpy as np
import utils
import copy
from utils import load_config, pickle_dump, pickle_load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def load_dataset(config: dict):
    x_train = pickle_load(config["train_set_path"][0])
    y_train = pickle_load(config["train_set_path"][1])

    x_valid = pickle_load(config["valid_set_path"][0])
    y_valid = pickle_load(config["valid_set_path"][1])

    x_test = pickle_load(config["test_set_path"][0])
    y_test = pickle_load(config["test_set_path"][1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def numericalImputation(data, numerical_column):
    #Filter numeric data
    numerical_data = data[numerical_column]
    
    #Buat imputer
    imputer_numerical = SimpleImputer(missing_values=np.nan,
                                     strategy="median")
    imputer_numerical.fit(numerical_data)
    
    #Transform
    imputed_data = imputer_numerical.transform(numerical_data)
    numerical_data_imputed = pd.DataFrame(imputed_data)
    
    numerical_data_imputed.columns = numerical_column
    numerical_data_imputed.index = numerical_data.index
    
    return numerical_data_imputed, imputer_numerical

def categoricalImputation(data, categorical_column):
    """
    Fungsi untuk melakukan imputasi data kategorik
    :param data: <pandas dataframe> sample data input
    :param categorical_column: <list> list kolom kategorikal data
    :return categorical_data: <pandas datafarame> data kategorikal
    """
    #Seleksi data
    categorical_data = data[categorical_column]
    
    #Lakukan imputasi
    #categorical_data = categorical_data.dropna(subset=["gender"])
    categorical_data = categorical_data.fillna(value="KOSONG")
    
    return categorical_data


def extractCategorical(data, categorical_column):
    """
    Fungsi untuk ekstrak data kategorikal dengan One Hot Encoding
    :param data: <pandas dataframe> data sample
    :param categorical_column: <list> list kolom kategorik
    :return categorical_ohe: <pandas dataframe> data sample dengan ohe
    """
    data_categorical = categoricalImputation(data=data,
                                            categorical_column=categorical_column)
    categorical_ohe = pd.get_dummies(data_categorical)
    return categorical_ohe


def std_scaler_fit(x_train: pd.DataFrame):
    std_scaler = StandardScaler()
    std_scaler.fit(x_train)
    return std_scaler

def std_scaler_transform(features: pd.DataFrame, scaler: object) -> pd.DataFrame:

    '''
    this function transform features using standar scaler machine
    '''
    
    col_names = scaler.feature_names_in_

    #feat = copy.deepcopy(features)

    scaled = scaler.transform(features)

    scaled_df = pd.DataFrame(scaled, columns=col_names)

    return scaled_df

def ohe_fit(data_tobe_fitted: dict, ohe_path: str) -> OneHotEncoder:
    # Create ohe object
    ohe = OneHotEncoder(categories='auto')

    # Save ohe object
    pickle_dump(
        ohe,
        ohe_path
    )

    # Return trained ohe
    return ohe

def ohe_transform(set_data: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Transform variable categorical of set data, resulting array
    categorical_features = ohe.fit_transform(set_data).toarray()
    
    categorical_column = ohe.categories_
    categorical_column = np.array(ohe.get_feature_names_out()).ravel()

    # Convert to dataframe
    categorical_features = pd.DataFrame(
        categorical_features,
        columns = categorical_column)

    # Return new feature engineered set data
    return categorical_features

if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Load dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset(config)

    # 3. Numerical set
    x_train_numerical = x_train[config["numerical_column"]]
    x_valid_numerical = x_valid[config["numerical_column"]]
    x_test_numerical = x_test[config["numerical_column"]]

    # 4. Categorical Set
    x_train_categorical = x_train[config["categorical_column"]]
    x_valid_categorical = x_valid[config["categorical_column"]]
    x_test_categorical = x_test[config["categorical_column"]]

    # 5. Handling variable days_since_last_login in train, valid and test set
    x_train_numerical["days_since_last_login"] = x_train_numerical["days_since_last_login"].replace(-999, np.nan)
    x_valid_numerical["days_since_last_login"] = x_valid_numerical["days_since_last_login"].replace(-999, np.nan)
    x_test_numerical["days_since_last_login"] = x_test_numerical["days_since_last_login"].replace(-999, np.nan)

    # 6. Handling variable points_in_wallet in train, valid and test set
    x_train_numerical["points_in_wallet"] = x_train_numerical["points_in_wallet"].where(lambda x: x >=0, np.nan)
    x_valid_numerical["points_in_wallet"] = x_valid_numerical["points_in_wallet"].where(lambda x: x >=0, np.nan)
    x_test_numerical["points_in_wallet"] = x_test_numerical["points_in_wallet"].where(lambda x: x >=0, np.nan)

    # 7. Handling variable gender in train, valid and test set
    x_train_categorical["gender"] = x_train_categorical["gender"].replace('Unknown', np.nan)
    x_valid_categorical["gender"] = x_valid_categorical["gender"].replace('Unknown', np.nan)
    x_test_categorical["gender"] = x_test_categorical["gender"].replace('Unknown', np.nan)

    # 8. Handling variable joined_through_referral in train, valid and test set
    x_train_categorical["joined_through_referral"] = x_train_categorical["joined_through_referral"].replace('?', np.nan)
    x_valid_categorical["joined_through_referral"] = x_valid_categorical["joined_through_referral"].replace('?', np.nan)
    x_test_categorical["joined_through_referral"] = x_test_categorical["joined_through_referral"].replace('?', np.nan)

    # 9. Handling variable medium_of_operation in train, valid and test set
    x_train_categorical["medium_of_operation"] = x_train_categorical["medium_of_operation"].replace('?', np.nan)
    x_valid_categorical["medium_of_operation"] = x_valid_categorical["medium_of_operation"].replace('?', np.nan)
    x_test_categorical["medium_of_operation"] = x_test_categorical["medium_of_operation"].replace('?', np.nan)

    # 10. Numerical Imputation in train, valid and test
    x_train_numerical, imputer_numerical = numericalImputation(data = x_train_numerical, numerical_column = config["numerical_column"])
    x_valid_numerical, imputer_valid_numerical = numericalImputation(data = x_valid_numerical, numerical_column = config["numerical_column"])
    x_test_numerical, imputer_test_numerical = numericalImputation(data = x_test_numerical, numerical_column = config["numerical_column"])
    
    # 11. Categorical Imputation in train, valid and test
    x_train_categorical = categoricalImputation(data = x_train_categorical, categorical_column=config["categorical_column"])
    x_valid_categorical = categoricalImputation(data = x_valid_categorical, categorical_column=config["categorical_column"])
    x_test_categorical = categoricalImputation(data = x_test_categorical, categorical_column=config["categorical_column"])

    # reset index numerical
    x_train_numerical = x_train_numerical.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_valid_numerical = x_valid_numerical.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    x_test_numerical = x_test_numerical.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # reset index categorical
    x_train_categorical = x_train_categorical.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_valid_categorical = x_valid_categorical.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    x_test_categorical = x_test_categorical.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # 12. OHE Categorical
    ohe_categorical = ohe_fit(config["categorical_column"], config["ohe_categorical_path"])
    x_train_categorical_ohe = ohe_transform(x_train_categorical, ohe_categorical)
    x_valid_categorical_ohe = ohe_transform(x_valid_categorical, ohe_categorical)
    x_test_categorical_ohe = ohe_transform(x_test_categorical, ohe_categorical)
    
    # 13. Join Numeric and Categorical
    x_train_concat = pd.concat([x_train_numerical, x_train_categorical_ohe], axis=1)
    x_train_concat = x_train_concat.dropna()
    x_valid_concat = pd.concat([x_valid_numerical, x_valid_categorical_ohe], axis=1)
    x_valid_concat = x_valid_concat.dropna()
    x_test_concat = pd.concat([x_test_numerical, x_test_categorical_ohe], axis=1)
    x_test_concat = x_test_concat.dropna()

    # 14. Standardizing Variables
    scaler = std_scaler_fit(x_train_concat)
    pickle_dump(scaler, config["scaler_path"])
    
    x_train_clean = std_scaler_transform(x_train_concat, scaler)
    x_valid_clean = std_scaler_transform(x_valid_concat, scaler)
    x_test_clean = std_scaler_transform(x_test_concat, scaler)

    #print(x_train_clean.index)
    #print(y_train.index)
    #y_train = y_train[x_train_clean.index]
    #y_valid = y_valid[x_valid_clean.index]
    #y_test = y_test[x_test_clean.index]

    pickle_dump(x_train_clean, config["train_clean_set_path"][0])
    pickle_dump(y_train, config["train_clean_set_path"][1])

    pickle_dump(x_valid_clean, config["valid_clean_set_path"][0])
    pickle_dump(y_valid, config["valid_clean_set_path"][1])

    pickle_dump(x_test_clean, config["test_clean_set_path"][0])
    pickle_dump(y_test, config["test_clean_set_path"][1])

 