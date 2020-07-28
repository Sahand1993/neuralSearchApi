FROM maven:3.6.3-jdk-11

RUN apt-get update
RUN apt-get install python3.7
RUN apt-get install python3-setuptools
RUN pip3 install --upgrade pip
RUN pip3 install wheel
RUN git clone https://github.com/Sahand1993/DataPreprocessor
RUN mkdir raw_datasets
RUN touch raw_datasets/empty.txt
RUN mkdir preprocessed_datasets

ARG confluence_username
ARG confluence_password

ENV confluence_username $confluence_username
ENV confluence_password $confluence_password

RUN cd DataPreprocessor && git checkout azure && mvn spring-boot:run

RUN pip3 install -r requirements.txt

ENV FLASK_APP app.py
ENV PYTHONUNBUFFERED 1
ENV NEURALSEARCH_TRIGRAMS_PATH preprocessed_datasets/trigrams.txt
ENV CONFLUENCE_INDICES_FILE preprocessed_datasets/confluence/data.csv

ENTRYPOINT ["flask", "run"]