# Ha2Mag

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Train
You can train a model: 
```bash
python train.py --dataroot ./datasets --name Ha2Mag_pix2pix --model pix2pix --dataset_mode aligned 
```
Models are saved to `./checkpoints/`.

See `opt` in files(base_options.py and train_options.py) for additional training options.

## Test
You can test the model:
```bash
python test.py --dataroot ./datasets/result_new --name Ha2Mag_pix2pix --model pix2pix --dataset_mode aligned --num_test 130
```
See `opt` in files(base_options.py and test_options.py) for additional training options.

Testing results are saved in `./result/`.

## Acknowledgments
Code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
