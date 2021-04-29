FROM python:3.9-slim-buster

RUN apt-get update -y &&  apt-get install -y apt-transport-https
# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-dev wget -y

# Detectron2 prerequisites
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install pydicom highdicom pynetdicom

# Detectron2
# RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Development packages
RUN pip install flask flask-cors requests opencv-python

RUN mkdir /app
WORKDIR /app
ADD . /app/

# Path updated 2021-04-18 with updated DICOM SEG trainer
# Total Loss: 0.3968
# Classes: left adrenal, left kidney, liver, pancreas, right adrenal, right kidney, spleen
RUN wget https://www.dropbox.com/s/550j3bmqo9b7sae/model_final.pth

CMD ["python", "/app/scp_scu.py"]
