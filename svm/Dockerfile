FROM python:3.9-slim
WORKDIR /pipeline
COPY svm/requeriments.txt /pipeline
RUN pip install -r requeriments.txt
COPY classify.py svm/svm.py /pipeline/