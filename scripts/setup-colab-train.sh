apt install -y vim screen

# Download `BMS Molecular Translation` training images.
pip install -U kaggle

export KAGGLE_USERNAME=[your kaggle username]
export KAGGLE_KEY=[your kaggle key]

kaggle competitions download -c bms-molecular-translation
unzip -qq bms-molecular-translation.zip -d res
rm bms-molecular-translation.zip

# Download pretrained InChI tokenizer.
kaggle kernels output bms-molecular-translation-train-inchi-tokenizer
mv tokenizer.json res/

# Install requirements.
pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
python -m wandb login [your wandb key]

# Install Nvidia apex. Note that minor-version check will be removed.
git clone https://github.com/NVIDIA/apex
sed -i "s/or (bare_metal_minor != torch_binary_minor)//g" apex/setup.py
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
rm -rf apex
