FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ENV PATH /opt/conda/bin:$PATH

# Install Dependencies of Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# conda create python 3.7.13
RUN conda create -n inference python=3.9.13

# add default environment in bashrc
RUN echo "conda activate inference" >> ~/.bashrc

WORKDIR /app

COPY . .
# install requirements
RUN /bin/bash -c "source activate inference && \
                 pip install -r requirements.txt"

# issue -opencv error
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libglib2.0-0

# download necessary files
RUN /bin/bash -c "source activate inference && \
                  python download_cartoon_data.py && \
                  bash download_deca_model.sh && \
                  mkdir cartoon_image avatar"

# run terminal
CMD ["/bin/bash"]