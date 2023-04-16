import json

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score


class ModelClassify:

    def __init__(self, input_data: str, classifier: ClassifierMixin):
        with open(input_data) as data_file:
            data = json.load(data_file)

        # Data type is 'dict', however since the file was loaded as a json object, it is first loaded as a string
        # thus we need to load again from such string in order to get the dict-type object.
        data = json.loads(data)

        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_test = data['x_test']
        self.y_test = data['y_test']
        self.classifier = classifier

    def run(self, accuracy_output: str):
        # Initialize and train the model
        model = self.classifier
        model.fit(self.x_train, self.y_train)

        # Get predictions
        y_pred = model.predict(self.x_test)

        # Get accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        # Save output into file
        with open(accuracy_output, 'w') as accuracy_file:
            accuracy_file.write(str(accuracy))
