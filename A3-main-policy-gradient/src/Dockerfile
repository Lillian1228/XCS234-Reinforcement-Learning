###################################################################
# This Dockerfile creates an Ubuntu 22.04 image with MuJoCo 2.3.5 and its
# dependencies installed/configured in order to run code from
# assignment 3 of XCS234.
#
# Notes: the image->container this file creates executes assignment
# code on the system CPU. Instructions for CUDA setup are included
# for students who wish to explore running the container with CUDA
# access. The labeling of steps below are distinct from the steps
# outlined by Docker when creating the image.
###################################################################

# 1. Pull base ubuntu image
FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

ARG CUDA

# 2. Update apt repositories and install wget
RUN apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y wget libgl1-mesa-glx libosmesa6

# 3. Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/miniconda_install.sh \
&& chmod +x /home/miniconda_install.sh \
&& bash /home/miniconda_install.sh -b -p /opt/miniconda \
&& rm /home/miniconda_install.sh

# Set working directory
WORKDIR /home/scpdxcs/A3

# 4. Copy and install conda environment
ADD ./environment.yml /home
ADD ./environment_cuda.yml /home

RUN if [[ -z "$CUDA" ]] ; then \
        source /opt/miniconda/bin/activate && cd /home && conda env create -f environment.yml ; \
    else \
        source /opt/miniconda/bin/activate && cd /home && conda env create -f environment_cuda.yml ; \
    fi

# 5. Ensure conda environment runs activated at startup
RUN if [[ -z "$CUDA" ]] ; then \
        echo "source /opt/miniconda/bin/activate && conda activate XCS234_A3" > ~/.bashrc ; \
    else \
        echo "source /opt/miniconda/bin/activate && conda activate XCS234_A3_CUDA" > ~/.bashrc ; \
    fi

# 6. Create virtual screen for recording episodes
ADD ./startup_script.sh /usr/local/bin/startup_script.sh
RUN chmod 777 /usr/local/bin/startup_script.sh
ENV DISPLAY=:1
ENV MUJOCO_GL=osmesa
ENTRYPOINT ["/usr/local/bin/startup_script.sh"]
