FROM python:3.9-slim-buster

RUN apt-get update -y

RUN pip install pynetdicom sqlalchemy

RUN mkdir /app
WORKDIR /app
ADD . /app/

CMD ["python", "-m", "pynetdicom", "qrscp", "-c", "/app/qrscp.cfg", "-v"]