import fire
from fire import Fire

import json
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def prepare_dataset(data_file: str) -> None:
    # Gets data from sklearn library and split dataset
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Creates `data` structure to save
    data = {'x_train': x_train.tolist(),
            'y_train': y_train.tolist(),
            'x_test': x_test.tolist(),
            'y_test': y_test.tolist()}

    # Creates a json object based on `data`
    data_json = json.dumps(data)

    with open(data_file, 'w') as output_file:
        json.dump(data_json, output_file)


def main(data: str) -> None:
    Path(data).parent.mkdir(parents=True, exist_ok=True)

    prepare_dataset(data)


if __name__ == '__main__':
    Fire(main)
