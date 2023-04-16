from pathlib import Path

from fire import Fire
from sklearn.tree import DecisionTreeClassifier

from classify import ModelClassify


def decision_tree(data_input: str, accuracy_output: str) -> None:
    dt = ModelClassify(input_data=data_input, classifier=DecisionTreeClassifier(max_depth=4))
    dt.run(accuracy_output)


def main(data: str, accuracy: str) -> None:
    Path(accuracy).parent.mkdir(parents=True, exist_ok=True)
    decision_tree(data_input=data, accuracy_output=accuracy)


if __name__ == '__main__':
    Fire(main)
