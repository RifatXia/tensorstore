- create an env with only Pytorch, only things necessary to use OPT-125 model
- then create a notebook (ipynb) which would contain some basic codes to load the model

conda create -n opt125 python=3.10 -y
conda activate opt125
pip install torch torchvision torchaudio transformers notebook
jupyter notebook