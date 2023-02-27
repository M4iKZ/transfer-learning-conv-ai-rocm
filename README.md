# Building a Conversational AI with Transfer Learning for ROCm

This repo contains the code from my blog post [Conversational AI with Transfer Learning for ROCm](https://medium.com/@m4ikz/conversational-ai-with-transfer-learning-for-rocm-b4c095990e0e).


This model can be trained in about two hours on a AMD 6900 xT with epoch set to 1, I advise to set at least to 3 to improve the loss.

## Installation

To install and use the training and inference scripts clone the repo and follow those commands inside the folder created after cloning:

```bash
pythom -m venv env

source bin/bin/activate

pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm5.3
pip install --pre pytorch-ignite 
pip install transformers
```

## Using the training script

The training script can be used in single GPU, at the moment multi-GPU/Cloud isn't supported.

```bash
python trainchat.py --n_epochs 3
```

## Using the interaction script

The training script saves all the experiments and checkpoints in a sub-folder named with the timestamp of the experiment in the `./runs` folder of the repository base folder.

You can then use the interactive script to interact with the model simply by using this command line to run the interactive script:

```bash
python interact.py --modelpath "runs/Feb27_13-06-51_m4ikz-linux_gpt2"
```

## Credits

My code is an edit version for ROCm based on the work described on this blog post [🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/@Thomwolf/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)