# Bristol-Myers Squibb – Molecular Translation
## Introduction
This repository is the 50th place solution for the [Bristol-Myers Squibb – Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation/overview) competition.

## Requirements
This project requires the below libraries:
- torch==1.8.1+cu111
- torchvision==0.9.1+cu111
- numpy
- albumentations
- pytorch_lightning
- wandb
- pandas
- opencv_python
- tokenizers
- python_Levenshtein
- PyYAML

You can simply install them by using:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

This repository contains training and prediction scripts. You can reproduce our results by using this project.
Note that the subword tokenization is performed with [huggingface tokenizers](https://github.com/huggingface/tokenizers) and the detailed training code is in [this notebook](https://www.kaggle.com/affjljoo3581/bms-molecular-translation-train-inchi-tokenizer).
Of course, do not forget to download the dataset from the competition. You can download both the dataset and the tokenizer by using:
```bash
kaggle competitions download -c bms-molecular-translation
unzip -qq bms-molecular-translation.zip -d res
rm bms-molecular-translation.zip

kaggle kernels output bms-molecular-translation-train-inchi-tokenizer
mv tokenizer.json res/
```

### Train a model

First of all, you need to make a training configuration file. For example:

```yaml
data:
  datasets:
    main:
      image_dir: res/train
      label_csv_path: res/train_labels.csv

  tokenizer_path: res/tokenizer.json
  val_ratio: 0.01

model:
  image_size: 224
  patch_size: 16
  max_seq_len: 256

  num_encoder_layers: 6
  num_decoder_layers: 6

  hidden_dim: 512
  num_attn_heads: 8
  expansion_ratio: 4

  encoder_dropout_rate: 0.1
  decoder_dropout_rate: 0.1

train:
  epochs: 10
  warmup_steps: 10000

  accumulate_grads: 8
  train_batch_size: 128
  val_batch_size: 128

  learning_rate: 1.e-4
  learning_rate_decay: linear

  weight_decay: 0.05
  max_grad_norm: 1.0

  grad_ckpt_ratio: 0.0

environ:
  name: [your model name]

  num_gpus: 1
  precision: 16
```
After writing your own training configuration file,  login to the [wandb](https://wandb.ai/) by using the below command to log the training and validation metrics.
```bash
wandb login [your wandb key]
```
Now you can train the model by:
```bash
python src/train.py [your config file]
```
If you want to use [apex](https://github.com/NVIDIA/apex) in training, use `--use_apex_amp` option. Note that the `apex` should be installed in your system.
It also supports resuming from checkpoint file and using pretrained weights. The detailed usage is as follows:
```
usage: train.py [-h] [--use_apex_amp] [--resume RESUME] [--checkpoint CHECKPOINT]
                [--pretrained PRETRAINED]
                config

positional arguments:
  config

optional arguments:
  -h, --help            show this help message and exit
  --use_apex_amp
  --resume RESUME
  --checkpoint CHECKPOINT
  --pretrained PRETRAINED
```

### Make a prediction
After training the model, you can make a prediction.
```yaml
data:
  image_dir: res/test
  label_csv_path: res/sample_submission.csv
  tokenizer_path: res/tokenizer.json

model:
  image_size: 224
  patch_size: 16
  max_seq_len: 256

  num_encoder_layers: 6
  num_decoder_layers: 6

  hidden_dim: 512
  num_attn_heads: 8
  expansion_ratio: 4

predict:
  batch_size: 1024
  weight_path: [trained model weight]

environ:
  name: [your model name]
  precision: 16
```
Create a configuration and run the below command:
```bash
python src/predict.py [your config file]
```

### Some utility scripts
This project also contains useful utility scripts.

#### weight averaging
```bash
python scripts/average_weights.py model1.pth model2.pth model3.pth ... --output averaged.pth
```

#### combining encoder and decoder transformers
```bash
python scripts/combine_encoder_and_decoder.py --encoder vit.pth --decoder gpt2.pth --output model.pth
```

#### download pretrained ViT encoder
```bash
python scripts/download_pretrained_encoder.py vit-large --output ViT-encoder.pth --include_embeddings
```

#### resize input image resolution
```bash
python scripts/resize_encoder_resolution.py model-224.pth --output model-384.pth --image_size 384
```

#### stack transformer layers
```bash
python scripts/stack_transformer_layers.py model-12.pth --output model-24.pth --num_encoder_layers 24 --num_decoder_layers 6 --modify_mode repeat-first
```

#### visualize embedding and projection layers
```bash
python scripts/visualize_embeddings.py model.pth
```

#### create external dataset
```bash
python scripts/create_external_dataset.py res/extra_inchi.csv --output_path . --num_folds 4 --fold_index 0
```