import pandas as pd
import os


def generate_dataframe(json_filenames=None):
    if json_filenames is None:
        json_filenames = [
            pos_json for pos_json in os.listdir(".") if pos_json.endswith(".json")
        ]
    print(f"Loading data from: {json_filenames}\n")

    dfs = []
    print(f"{'filename':<20} {'samples':<7}")
    for filename in json_filenames:
        json_df = pd.read_json(path_or_buf=filename, lines=True)
        print(f"{filename.split('.')[0]:<20} {len(json_df):<7}")
        dfs.append(json_df)

    df = pd.concat(dfs)
    print(f"\nData loaded, {len(df)} total samples.")
    return df
