# Errorneous MNIST with DAC loss

This repo is to test **Deep Abstaining Classifier** proposed by [Sunil Thulasidasan et al., "Combating Label Noise in Deep Learning Using Abstention"](https://arxiv.org/abs/1905.10964) for MNIST example, with enforced label noise.

The `dac_loss.py` is slightly modifed from its [original implementation](https://github.com/thulas/dac-label-noise).

### Requirements
- Python 3+
- PyTorch 1.1+
- TorchVision 0.3+

### How to train & test
```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```
