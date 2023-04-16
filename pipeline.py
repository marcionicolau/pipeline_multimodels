from fire import Fire

import kfp
from kfp import dsl
from kfp.components import func_to_container_op


@func_to_container_op
def show_results(decision_tree: float, logistic_regression: float, svm: float, naive_bayes: float) -> None:
    print(f"Decision tree (accuracy): {decision_tree}")
    print(f"Logistic regression (accuracy): {logistic_regression}")
    print(f"SVM (SVC) (accuracy): {svm}")
    print(f"Naive Bayes (Gaussian) (accuracy): {naive_bayes}")


@dsl.pipeline(name='ML Models Pipeline',
              description='Run Decision Tree, Logistic Regression, SVM and Naive Bayes for classification problem.')
def models_pipeline() -> None:
    prepare_data = kfp.components.load_component_from_file('prepare_data/prepare_data.yaml')
    decision_tree = kfp.components.load_component_from_file('decision_tree/decision_tree.yaml')
    logistic_regression = kfp.components.load_component_from_file('logistic_regression/logistic_regression.yaml')
    svm = kfp.components.load_component_from_file('svm/svm.yaml')
    naive_bayes = kfp.components.load_component_from_file('naive_bayes/naive_bayes.yaml')

    # Run prepare_data
    prepare_task = prepare_data()

    # Run ML models tasks with input data
    decision_tree_task = decision_tree(prepare_task.output)
    logistic_regression_task = logistic_regression(prepare_task.output)
    svm_task = svm(prepare_task.output)
    naive_bayes_task = naive_bayes(prepare_task.output)

    show_results(decision_tree=decision_tree_task.output,
                 logistic_regression=logistic_regression_task.output,
                 svm=svm_task.output,
                 naive_bayes=naive_bayes_task.output)


def main(pipeline: str = 'Classify-Models-Pipeline.yaml') -> None:
    kfp.compiler.Compiler().compile(models_pipeline, pipeline)


if __name__ == '__main__':
    Fire(main)
