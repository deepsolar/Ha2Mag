# Ha2Mag

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.

## Dataset
The data we used in the paper is in `./dataset`. 
The 2048 x 1024 images in our dataset are pairs of $H\alpha$ images and the corresponding SDO/HMI magnetograms. The dataset mode is aligned.

## Train
You can train a model as the following instruction: 
```bash
python train.py --dataroot ./dataset --name Ha2Mag_pix2pix --model pix2pix
```
Models are saved to `./checkpoints/`.

See `opt` in files(base_options.py and train_options.py) for additional training options.

## Test
You can test the model as the following instruction:
```bash
python test.py --dataroot ./dataset --name Ha2Mag_pix2pix --model pix2pix
```
See `opt` in files(base_options.py and test_options.py) for additional testing options.

Testing results are saved in `./results/`.

## Acknowledgments
Code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
