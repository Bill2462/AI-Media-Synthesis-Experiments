# AI dreams experiments

This repository contains random notebooks connected with generating content using AI.

## Experiments

Text guided image

 - `experiments/image_init.ipynb` -> Demonstration of how image can be used to initialize Z in text guided image synthesis with VQGAN + CLIP.

## Setting up the environment

First setup environment containing pytorch, torchvision and preferrably cuda toolkit so
GPU can be used for acceleration.

For example here is command to create environment using conda.

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install omegaconf pytorch-lightning jupyterlab matplotlib
pip install git+https://github.com/bfirsh/taming-transformers.git
pip install git+https://github.com/openai/CLIP.git
```

Then download models:

CLIP:

```
curl -o ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

VQGAN models:

```
curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1'
curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1'

curl -L -o coco.yaml -C - 'https://dl.nmkd.de/ai/clip/coco/coco.yaml'
curl -L -o coco.ckpt -C - 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt'

curl -L -o wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt'
curl -L -o wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml'

curl -L -o sflckr.yaml -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1'
curl -L -o sflckr.ckpt -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1'
```

Environment should also contain environment variable called `MODEL_STORE` which should contain path to location where model files are stored.
It can be defined using the following command:

```
export MODEL_STORE="/home/skynet/AI/dream_weights"
```
