FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

COPY ./application /opt/application
COPY ./models /opt/models
COPY ./utils /opt/utils
COPY ./requirements.txt /opt/requirements.txt
RUN mkdir -p /opt/resources/

WORKDIR /opt

RUN apt-get update

RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip install -r /opt/requirements.txt

ENV PYTHONPATH /opt

EXPOSE 8081

CMD [ "python", "/opt/application/main.py"]