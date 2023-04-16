FROM python:3.9-slim
WORKDIR /pipeline
COPY logistic_regression/requeriments.txt /pipeline
RUN pip install -r requeriments.txt
COPY classify.py logistic_regression/logistic_regression.py /pipeline/
ENTRYPOINT ["python", "logistic_regression.py"]