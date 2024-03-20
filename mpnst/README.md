## Setup: Conda
conda create --name CDRP python=3.8
conda activate CDRP
pip install -r requirements.txt

## Setup dependencies with docker
docker build -f Dockerfile -t deeptta . --build-arg HTTPS_PROXY=$HTTPS_PROXY

## run to test mpnst
docker run -v $PWD:/tmp deeptta /opt/venv/bin/python run_deep_TTA.py

## interactive session
docker run -it --name my_deeptta_container deeptta /bin/bash        


## To train models - old

`python run_models.py --config configs/config_ccle.yaml`

## process the mpnst dataset into the transformer model