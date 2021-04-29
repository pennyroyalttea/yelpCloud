FROM anibali/pytorch:cuda-10.0

WORKDIR /yelpcloud

COPY requirements.txt /yelpcloud
RUN sudo apt-get update
RUN sudo apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN pip install -r ./requirements.txt

COPY . /yelpcloud
RUN sudo chmod -R a+rwx /yelpcloud/
CMD ["python3","server2.py"]~
