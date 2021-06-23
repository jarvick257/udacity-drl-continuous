FROM rocm/pytorch:rocm4.2_ubuntu18.04_py3.6_pytorch

WORKDIR /data

RUN apt update -y
RUN apt install -y git python3 python3-pip

RUN git clone https://github.com/udacity/deep-reinforcement-learning.git udacity
RUN cd /data/udacity/python && sed -i '/torch/d' requirements.txt && pip3 install .

