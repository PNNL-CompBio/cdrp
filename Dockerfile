FROM python:3.8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --allow-insecure-repositories
RUN apt-get install -y --allow-unauthenticated build-essential --fix-missing libpq-dev python3.10 python3-pip python3-setuptools python3-dev python3-venv libcurl4-openssl-dev libxml2-dev 

RUN python3 -m venv /opt/venv
RUN /opt/venv/bin/pip3 install --upgrade pip

ENV PYTHONPATH "${PYTHONPATH}:/app"
WORKDIR /app

ADD requirements.txt .
ADD MPNST_input/* .
# ADD configs/* .
ADD *.py .
ADD *.sh .

# installing python libraries
RUN /opt/venv/bin/pip3 install -r requirements.txt

VOLUME ["/tmp"]