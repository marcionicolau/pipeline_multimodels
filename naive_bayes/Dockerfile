FROM python:3.9-slim
WORKDIR /pipeline
COPY naive_bayes/requeriments.txt /pipeline
RUN pip install -r requeriments.txt
COPY classify.py naive_bayes/naive_bayes.py /pipeline/
ENTRYPOINT ["python", "naive_bayes.py"]