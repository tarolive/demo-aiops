FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /app
COPY . /app

RUN pip install --upgrade Flask prometheus-api-client requests urllib3

ENV FLASK_APP app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
