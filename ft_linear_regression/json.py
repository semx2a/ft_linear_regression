import json
import os

import numpy as np


def generate_json(name: str, content: dict) -> None:
    """Generate a JSON file with the given content.

    Parameters:
    name (str): name of the JSON file.
    content (dict): content to write in the JSON file.
    """
    try:
        if not name.endswith(".json"):
            name += ".json"
        for key, value in content.items():
            if isinstance(value, np.ndarray):
                content[key] = value.tolist()
        with open(name, "w") as f:
            json.dump(content, f)
            f.close()

    except Exception as e:
        print(f"Error: {e}")


def load_json(path: str) -> dict | None:
    """Load a JSON file and return its content.

    Parameters:
    path (str): path to the JSON file.
    """
    try:
        if not path.endswith(".json"):
            path += ".json"
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                return data
        else:
            return None

    except FileNotFoundError as fnf:
        print(f"Load JSON Error: {fnf}")
    except Exception as e:
        print(f"Load JSON Error: {e}")
