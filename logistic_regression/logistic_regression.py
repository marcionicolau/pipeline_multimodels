from pathlib import Path

from fire import Fire
from sklearn.linear_model import LogisticRegression

from classify import ModelClassify


def logistic(data_input: str, accuracy_output: str) -> None:
    lr = ModelClassify(data_input, classifier=LogisticRegression())
    lr.run(accuracy_output)


def main(data: str, accuracy: str) -> None:
    Path(accuracy).parent.mkdir(parents=True, exist_ok=True)
    logistic(data_input=data, accuracy_output=accuracy)


if __name__ == '__main__':
    Fire(main)
