# DEKP: a deep learning model for enzyme kinetic parameter prediction based on pretrained models and graph neural networks
![workflow](./workflow.png)
## Preparation
Here are the pretrained models used in DEKP

[ProtT5-XL-U50](https://zenodo.org/records/4644188)

[SMILES Transformer](https://github.com/DSPsleeporg/smiles-transformer)

[PST](https://github.com/BorgwardtLab/PST)

[MolFormer](https://github.com/IBM/molformer)

Extension libraries that need to be installed separately
[PyTorch geometric (PyG)](https://pytorch-geometric.com/whl/torch-2.1.2%2Bcu118.html)

## Usage
```
git clone https://github.com/wang-yi-zhen/DEKP
conda env create -f environment.yml
```
The files in the **Encode** folder are used to extract different features. You can run the corresponding code file based on your needs, but make sure that the dataset required by the file has been prepared.

