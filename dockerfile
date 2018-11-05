FROM ubuntu:16.04

RUN apt-get update && apt-get install -y wget ca-certificates \
    git curl vim python3-dev python3-pip \
    libfreetype6-dev libpng12-dev libhdf5-dev

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas==0.23.4 scikit-learn==0.20.0 matplotlib seaborn jupyter stop-words
RUN pip3 install keras
RUN pip3 install theano
RUN pip3 install flask-restful

ADD . /

EXPOSE 80

CMD [ "python3", "/main.py" ]