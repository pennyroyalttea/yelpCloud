FROM anibali/pytorch:cuda-10.0

WORKDIR /yelpcloud

COPY requirements.txt /yelpcloud
RUN sudo apt-get update
RUN sudo apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN python3.7 -m pip install -r ./requirements.txt
ARG FLASK_ENV="production"
ENV FLASK_ENV="${FLASK_ENV}" \
    PYTHONUNBUFFERED="true"

CMD [ "flask", "run" ]
COPY . /yelpcloud
RUN sudo chmod -R a+rwx /yelpcloud/
ADD . /yelpcloud/flask_app/
ADD . . 
CMD ["python3.7","/yelpcloud/flask_app/server2.py"]~
