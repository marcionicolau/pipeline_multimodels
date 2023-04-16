from pathlib import Path

from fire import Fire
from sklearn.naive_bayes import GaussianNB

from classify import ModelClassify


def naive_bayes(data_input: str, accuracy_output: str) -> None:
    nb = ModelClassify(data_input, classifier=GaussianNB())
    nb.run(accuracy_output)


def main(data: str, accuracy: str) -> None:
    Path(accuracy).parent.mkdir(parents=True, exist_ok=True)
    naive_bayes(data_input=data, accuracy_output=accuracy)


if __name__ == '__main__':
    Fire(main)
