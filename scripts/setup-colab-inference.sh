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

# Download trained model weights
kaggle datasets download bms-molecular-translation-mot-trained-weights
unzip -qq bms-molecular-translation-mot-trained-weights.zip
rm bms-molecular-translation-mot-trained-weights.zip

# Install requirements.
pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
