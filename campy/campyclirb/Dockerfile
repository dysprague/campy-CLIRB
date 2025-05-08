FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ENV NV_CUDNN_VERSION=8.6.0
ENV NV_CUDNN_PACKAGE=libcudnn8=8.6.0-1+cuda11.8
ENV NV_CUDNN_PACKAGE_DEV=libcudnn8-dev=8.6.0-1+cuda11.8
LABEL com.nvidia.cudnn.version=8.6.0

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Eastern 

# Setup miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Deps for OpenCV (ref: https://stackoverflow.com/a/63377623)
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get install -y git

RUN git config --global core.autocrlf input

# Create an environment (named "campyclirb")
COPY environment.yml /tmp
RUN conda env create -f /tmp/environment.yml

# Make conda run commands use the env:
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "campyclirb", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y dos2unix

# Fix some bash/docker stuff
# https://gist.github.com/xkortex/84226629c2a1286120bf139b93bca9bf
ENV CONDA_DEFAULT_ENV=campyclirb
RUN conda init bash && echo 'conda activate "${CONDA_TARGET_ENV:-campyclirb}"' >>  ~/.bashrc
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/miniconda3/envs/campyclirb/lib/python3.8/site-packages/tensorrt:/usr/local/cuda-11.8/targets/x86_64-linux/lib"
ENV PATH="/usr/local/cuda-11.8/bin:/usr/local/cuda-11.8/targets/x86_64-linux/lib:$PATH"
COPY entrypoint.sh /usr/local/bin/
RUN dos2unix /usr/local/bin/entrypoint.sh 
RUN chmod u+x /usr/local/bin/entrypoint.sh
SHELL ["/usr/local/bin/entrypoint.sh", "/bin/bash", "-c"]

# Setup our jupyter lab shortcut
# https://stackoverflow.com/questions/49024624/how-to-dockerize-jupyter-lab
COPY jlab.sh /usr/local/bin/
RUN dos2unix /usr/local/bin/jlab.sh
RUN chmod u+x /usr/local/bin/jlab.sh
RUN ln -s /usr/local/bin/jlab.sh /usr/local/bin/jlab
EXPOSE 8888


#FROM tf AS jlab
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tf", "jupyter", "lab","--ip=0.0.0.0","--allow-root"]

# Build with:
# docker build .

# Run with:
# docker run --gpus all -it -w /mnt -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tf:latest
# docker run --gpus all -it -w /mnt -v $PWD:/mnt tf:latest
# docker run --gpus all -it -w /mnt -v $PWD:/mnt jlab:latest

# docker run --gpus all -it -w /mnt -v $PWD:/mnt tf:latest bash
# docker run --gpus all -it -w /mnt -v $PWD:/mnt tf:latest jlab