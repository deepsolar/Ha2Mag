# Ha2Mag

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.

## Dataset
The data we used in the paper is in `./dataset`. The size of training image is 1024 x 1024, but they are randomly crop to 512 x 512 in training process. And the images we generated is 1024 x 1024. 

## Train
You can train a model as the following instruction: 
```bash
python train.py --dataroot ./datasets --name Ha2Mag_pix2pix --model pix2pix --dataset_mode aligned 
```
Models are saved to `./checkpoints/`.

See `opt` in files(base_options.py and train_options.py) for additional training options.

## Test
You can test the model as the following instruction:
```bash
python test.py --dataroot ./datasets/result_new --name Ha2Mag_pix2pix --model pix2pix --dataset_mode aligned --num_test 130
```
See `opt` in files(base_options.py and test_options.py) for additional testing options.

Testing results are saved in `./results/` and a html file here: `./results/Ha2Mag_pix2pix/latest_test/index.html`.

## Acknowledgments
Code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
