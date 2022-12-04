import yaml
import joblib

config_path = "../config/config.yaml"

def load_config() -> dict: 
    # Try to load yaml file
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

    except FileNotFoundError as fe:
        raise RuntimeError("Parameters file not found in path.")

    # Return params in dict format
    return config

def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)

def pickle_dump(file, file_path: str) -> None:
    # Dump data into file
    return joblib.dump(file, file_path)

if __name__ == "__main__":

    config_path = "../config/config.yaml"

    config = load_config(config_path)