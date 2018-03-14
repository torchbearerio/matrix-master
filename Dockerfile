FROM debian:8.5

MAINTAINER Kamil Kwiek <kamil.kwiek@continuum.io>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN apt-get install -y libmysqlclient15-dev

RUN apt-get install -y build-essential

RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev \
    libavformat-dev libswscale-dev

RUN apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

RUN apt-get install -y libatlas-base-dev gfortran

ADD environment.yml environment.yml
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/matrix/bin:$PATH

ADD https://api.github.com/repos/torchbearerio/python-core/git/refs/heads/master version.json
RUN pip install git+https://github.com/torchbearerio/python-core.git --upgrade

ADD matrixmaster app

EXPOSE 8080
ENV PORT 8080

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "python", "-m", "app" ]
