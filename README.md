# Errorneous MNIST with DAC loss

This repo is to test [DAC loss](https://arxiv.org/abs/1905.10964) for MNIST example, with enforced label noise.

The `dac_loss.py` is slightly modifed from its [original implementation](https://github.com/thulas/dac-label-noise).

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```
