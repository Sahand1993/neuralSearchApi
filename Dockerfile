FROM maven:3.6.3-jdk-11

RUN apt-get clean \
    && apt-get -y update
RUN apt-get -y install python3.7

RUN apt-get -y install nginx \
    && apt-get -y install python3-dev \
    && apt-get -y install build-essential
RUN apt-get -y install python3-setuptools

RUN apt -y install python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install wheel
RUN apt-get -y install libpcre3 libpcre3-dev
RUN pip3 install uwsgi

ARG confluence_username
ARG confluence_password

ENV confluence_username $confluence_username
ENV confluence_password $confluence_password

RUN mkdir -p /srv/flask_app

RUN cd /srv && git clone https://github.com/Sahand1993/DataPreprocessor && cd -
RUN mkdir /srv/raw_datasets
RUN touch /srv/raw_datasets/empty.txt
RUN mkdir /srv/preprocessed_datasets
RUN cd /srv/DataPreprocessor && git checkout azure && mvn spring-boot:run && cd -

COPY datasetiterators /srv/flask_app/datasetiterators
COPY dssm /srv/flask_app/dssm
COPY helpers /srv/flask_app/helpers
COPY uwsgi.ini /srv/flask_app
COPY requirements.txt /srv/flask_app
COPY start.sh /srv/flask_app
COPY wsgi.py /srv/flask_app
COPY app.py /srv/flask_app/app.py
WORKDIR /srv/flask_app

RUN pip install -r requirements.txt --src /usr/local/src

#RUN useradd --no-create-home www-data
#RUN groupadd www-data
RUN usermod -a -G www-data www-data

RUN rm /etc/nginx/sites-enabled/default
RUN rm -r /root/.cache

COPY nginx.conf /etc/nginx/
RUN chmod +x ./start.sh

ENV FLASK_APP app.py
ENV NEURALSEARCH_TRIGRAMS_PATH /srv/preprocessed_datasets/trigrams.txt
ENV CONFLUENCE_INDICES_FILE /srv/preprocessed_datasets/confluence/data.csv
ENV CONFLUENCE_TEXT_FILE /srv/preprocessed_datasets/confluence/mid.json

EXPOSE 5000

# Running directly from flask because otherwise it doesn't work.
ENTRYPOINT ["flask", "run", "--host", "0.0.0.0"]
#ENTRYPOINT ["./start.sh"]
#ENTRYPOINT ["uwsgi", "--master", "--http-socket", "0.0.0.0:5000", "--protocol=http", "-w", "wsgi:app", "-uid", "nginx"]