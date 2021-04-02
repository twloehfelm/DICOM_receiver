FROM python:3.9-slim-buster

RUN apt-get update -y

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

# Path updated 2021-03-31 with updated model after round 1
RUN wget https://www.dropbox.com/s/hrwa7dz81a7cf87/model_final.pth

CMD ["python", "/app/scp_scu.py"]
