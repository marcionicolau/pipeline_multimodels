FROM python:3.9-slim
WORKDIR /pipeline
COPY requeriments.txt /pipeline
RUN pip install -r requeriments.txt
COPY prepare_data.py /pipeline/