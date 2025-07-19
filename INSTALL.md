
# Installation

## Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 
  Note, please check PyTorch version matches that is required by Detectron2.
- Detectron2: follow Detectron2 installation instructions.
- OpenCV ≥ 4.6 is needed by demo and visualization.

## Example conda environment setup

```bash
conda create --name cutler python=3.8 -y
conda activate cutler
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone --recursive git@github.com:facebookresearch/CutLER.git
cd CutLER
pip install -r requirements.txt
pip install sahi==0.11.18 pycocotools

# Move the following files to the location of your conda environments (.e.g, miniconda/envs/cutler/lib/python3.8/site-packages/detectron2/)
cp detectron2_modifications/defaults.py /miniconda/envs/cutler/lib/python3.8/site-packages/detectron2/engine/defaults.py

cp detectron2_modifications/augmentation_impl.py /miniconda/envs/cutler/lib/python3.8/site-packages/detectron2/data/transforms/augmentation_impl.py


# Move the following files to miniconda/envs/cutler/lib/python3.8/site-packages/sahi/:
# Change lines 16-17 in sahi_modifications/detectron2.py to: sys.path.insert(0, "<absolute path to CutLER>"); sys.path.insert(0, "<absolute path to CutLER/cutler>")
cp sahi_modifications/detectron2.py /miniconda/envs/cutler/lib/python3.8/site-packages/sahi/models/detectron2.py

cp sahi_modifications/annotation.py /miniconda/envs/cutler/lib/python3.8/site-packages/sahi/annotation.py
```

## datasets
If you want to train/evaluate on the datasets, please see [datasets/README.md](datasets/README.md) to see how we prepare datasets for this project.