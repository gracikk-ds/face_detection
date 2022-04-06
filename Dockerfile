FROM python:3.8-slim-buster

RUN pip install -U pip

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install google-cloud google-cloud-aiplatform
RUN pip install opencv-contrib-python
RUN pip install opencv-python streamlit pandas Pillow pymilvus==2.0.0rc8
RUN pip install plotly
RUN pip install albumentations==0.5.2
RUN pip install facenet_pytorch
RUN pip install dynaconf
RUN pip install sklearn
RUN pip install mediapipe

RUN mkdir /app

ADD ./branded /app/branded
ADD ./src /app/src
ADD ./examples /app/examples

WORKDIR /app/src

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port", "8080", "–server.address", "0.0.0.0"]
