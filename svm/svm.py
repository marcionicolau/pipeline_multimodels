from pathlib import Path

from fire import Fire
from sklearn.svm import SVC

from classify import ModelClassify


def svm(data_input: str, accuracy_output: str) -> None:
    svc = ModelClassify(data_input, classifier=SVC(kernel='linear'))
    svc.run(accuracy_output)


def main(data: str, accuracy: str) -> None:
    Path(accuracy).parent.mkdir(parents=True, exist_ok=True)

    svm(data_input=data, accuracy_output=accuracy)


if __name__ == '__main__':
    Fire(main)
