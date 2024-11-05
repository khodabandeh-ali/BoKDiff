FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel
WORKDIR /mnt

RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && apt-get install -y git \
    && apt-get install -y python3-rdkit \
    && apt-get install -y libopenbabel-dev \
    && apt-get install -y libffi-dev \
    && apt-get install -y python3-dev \
    && apt-get install -y apt-utils
    && rm -rf /var/lib/apt/lists/*

# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH
# RUN conda create -n decompdiff python=3.8
RUN pip3 install torch_geometric
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
# SHELL ["conda", "run", "-n", "decompdiff", "/bin/bash", "-c"]
# RUN conda activate decompdiff
# RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install tensorboard pyyaml easydict lmdb 
RUN pip3 install openbabel-wheel==3.1.1.20
# For decomposition
RUN pip3 install mdtraj
RUN pip3 install alphaspace2
RUN pip3 install scikit-learn
# RUN apt-get install -y libboost-all-dev \
# RUN apt-get install -y swig
# # For Vina Docking
RUN pip3 install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
RUN python3 -m pip3 install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

ENTRYPOINT ["python3"]
