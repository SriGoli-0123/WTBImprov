import json
import os
import string
import random


def load_file(file_path, sort_by_id=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))

    if sort_by_id:
        result.sort(key=sort_key)
    return result


def sort_key(entry):
    parts = entry["id"].rsplit("_", 1)
    index = parts[1]
    return int(index)


def generate_random_string(length):
    # Define the set of possible characters
    characters = string.ascii_letters + string.digits
    # Use random.choices() to randomly select a specified number of characters
    random_string = "".join(random.choices(characters, k=length))
    return random_string


def write_list_of_dicts_to_file(filename, data, subdir=None):
    if subdir:
        # Ensure the subdirectory exists
        os.makedirs(subdir, exist_ok=True)

        # Construct the full path to the file
        filename = os.path.join(subdir, filename)

    # Write the list of dictionaries to the file in JSON format
    with open(filename, "w") as f:
        for i, entry in enumerate(data):
            # Go through each key-value pair in the dictionary to make sure the values are JSON serializable
            entry = make_json_serializable(entry)
            json_str = json.dumps(entry)
            f.write(json_str)
            if i < len(data) - 1:
                f.write("\n")


def make_json_serializable(value):
    if isinstance(value, dict):
        # If the value is a dictionary, we need to go through each key-value pair recursively
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        # If the value is a list, we need to process each element recursively
        return [make_json_serializable(item) for item in value]
    else:
        # Try to serialize the value directly, and if it fails, convert it to a string
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)


def write_dicts_to_file(filename, data, subdir=None):
    if subdir:
        # Ensure the subdirectory exists
        os.makedirs(subdir, exist_ok=True)

        # Construct the full path to the file
        filename = os.path.join(subdir, filename)

    # Write the list of dictionaries to the file in JSON format
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
