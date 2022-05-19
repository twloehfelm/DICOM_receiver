FROM python:3.10.4-slim-buster

RUN apt-get update -y &&  apt-get install -y apt-transport-https
# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-dev wget -y

RUN pip install pydicom==2.2.2 pynetdicom==2.0.1 numpy

# Development packages
RUN pip install flask flask-cors requests

RUN mkdir /app
WORKDIR /app
ADD . /app/

CMD ["python", "/app/scp_scu.py"]
