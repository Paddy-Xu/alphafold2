conda create -n af2_local python==3.11
conda install --quiet --yes --channel conda-forge openmm=8.0.0 pdbfixer
conda clean --all --force-pkgs-dirs --yes

# only if you have a compatible NVIDIA GPU and want to use GPU acceleration for AlphaFold
    # && conda install --quiet --yes --channel nvidia cuda=${CUDA_VERSION} \

pip3 install --upgrade pip --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir


pip3 install --upgrade --no-cache-dir \
      jaxlib==0.4.26
# only if you have a compatible NVIDIA GPU and want to use GPU acceleration for AlphaFold
#pip3 install --upgrade --no-cache-dir      cuda12.cudnn89 \
#    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#


wget -q -P alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt



For Linux:
  if have root access, you can install hh-suite using your package manager. For example, on Ubuntu:
    sudo apt-get update
    sudo apt-get install -y cmake build-essential libomp-dev

    sudo apt install -y \
        build-essential \
        cmake \
        git \
        hmmer \
        kalign \
        tzdata \
        wget


  if not then use conda
    conda install -c conda-forge cmake make
    conda install -y -c conda-forge \
    gcc_linux-64 \
    gxx_linux-64



For macOS, you can install hh-suite using Homebrew:
# Install dependencies
  brew install cmake

# For OpenMP support (recommended)
  brew install libomp


## Install kalign
  For LUMI:L
    note that for LUMI, kalign needs to be installed avoiding cray compiler wrappers
    1. Get out of the Cray environment

    module purge
    module load gcc


  # Clone and build
    git clone https://github.com/TimoLassmann/kalign.git
    cd kalign
    mkdir build && cd build
    # cmake ..
    mkdir -p install
    cmake .. -DCMAKE_INSTALL_PREFIX="$PWD/../install" -DCMAKE_C_COMPILER=$(which gcc)
    make
    make test
    make install

    check ldd /scratch/project_465001728/kalign/install_nocray/bin/kalign
    : this output should not contain libsci_cray_* or libxpmem. You should only see standard things like libm.so, libc.so, maybe libgomp.so, etc.


    # Make the kalign binary available in your PATH
    # add the following line to your ~/.bashrc !!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    export PATH="/scratch/project_465001728/kalign/install/bin:$PATH"
    export LD_LIBRARY_PATH="/scratch/project_465001728/kalign/install/lib64:$LD_LIBRARY_PATH"
    export KALIGN_PREFIX=/scratch/project_465001728/kalign/install

    Note that cmake need install prefix to a local directory
    Note that kalign and jackhmmer need to be installed separately.

  For MacOS:

    brew install kalign
    brew install jackhmmer


jackhmmer is installed before

hhsuite

git clone --branch v3.3.0 --single-branch https://github.com/soedinglab/hh-suite.git  hh-suite \
    && mkdir  hh-suite/build \
    && pushd  hh-suite/build \
    && cmake  -DCMAKE_POLICY_VERSION_MINIMUM=3.5  \
     -DCMAKE_INSTALL_PREFIX="$PWD/../install" \
     -DCMAKE_C_COMPILER=$(which gcc) .. \
    &&  make -j &&  make install

if error, 在 hh-suite/src/a3m_compress.h 加
#include <cstdint>



    # Make the hhsuite binary available in your PATH
# add the following line to your ~/.bashrc !!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    export HHSUITE_ROOT=/scratch/project_465001728/hh-suite/install
    export PATH="$HHSUITE_ROOT/bin:$PATH"


    export HHSUITE_ROOT=/home/greenfold/PythonProjects/alphafold2/hh-suite/install
    export PATH="$HHSUITE_ROOT/bin:$PATH"



