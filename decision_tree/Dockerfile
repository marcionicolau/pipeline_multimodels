FROM python:3.9-slim
WORKDIR /pipeline
COPY decision_tree/requeriments.txt /pipeline
RUN pip install -r requeriments.txt
COPY classify.py decision_tree/decision_tree.py /pipeline/