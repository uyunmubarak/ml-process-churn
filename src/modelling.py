import utils
from utils import load_config, pickle_dump, pickle_load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_dataset(config: dict):
    x_train = pickle_load(config["train_clean_set_path"][0])
    y_train = pickle_load(config["train_clean_set_path"][1])

    x_valid = pickle_load(config["valid_clean_set_path"][0])
    y_valid = pickle_load(config["valid_clean_set_path"][1])

    x_test = pickle_load(config["test_clean_set_path"][0])
    y_test = pickle_load(config["test_clean_set_path"][1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Load dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset(config)

    # 3. model
    random_forest = RandomForestClassifier(random_state=123)
    params_randomforest = {
    "n_estimators":[100,200,300],
    "max_depth" : [2,3,4,5],
    "criterion" : ["gini", "entrophy", "log_loss"],
    "max_features" : ["sqrt", "log2"]
    }

    rf_cv = GridSearchCV(estimator = random_forest,
                     param_grid = params_randomforest,
                     cv = 5,
                     n_jobs=-1,
                     verbose=2)

    rf_cv.fit(x_train, y_train)       
    rf_cv.best_params_  
    # Refit RF
    random_forest = RandomForestClassifier(n_estimators = rf_cv.best_params_["n_estimators"],
                max_depth = rf_cv.best_params_["max_depth"],
                criterion = rf_cv.best_params_["criterion"],
                max_features = rf_cv.best_params_["max_features"],
                random_state = 123)

    # Fit model
    random_forest.fit(x_train, y_train)    
    y_pred_train_rf = random_forest.predict(x_valid)
    print(classification_report(y_true = y_valid, y_pred = y_pred_train_rf))

    y_pred_test_rf = random_forest.predict(x_test)
    print(classification_report(y_true = y_test, y_pred = y_pred_test_rf))

    pickle_dump(random_forest, config["production_model_path"])
