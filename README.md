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

cd-hit-2d is used to calculate the sequence similarity of proteins
[cd-hit-2d](https://github.com/weizhongli/cdhit/releases)

PDBFixer is used to repair and prepare Protein Data Bank (PDB) files for molecular simulations by automatically fixing missing atoms, residues, and hydrogens while removing unwanted molecules.
```
conda install -c conda-forge pdbfixer
pdbfixer
```

## Dataset
[kcat](https://drive.google.com/file/d/19kjQP5AyNqxfBHwAOxlqjWxcKXaoyr3q/view?usp=sharing)

[Km](https://drive.google.com/file/d/1e4HC1fjqbwZgyJiFbPt1V9XYi5Cto3z9/view?usp=sharing)

[Protein Structure Datasets](https://zenodo.org/records/15081759?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImRiMGNiNWE5LTUwOTUtNDFjYi05ZDIxLWMzZTIxYTMwMWE4ZCIsImRhdGEiOnt9LCJyYW5kb20iOiI0ODU0ZTg0MjQ0ZDllOWJmMDdkZjAxMmRhNDAxOTdiMiJ9.CFyXcG_1ED_izVaGK3KLxMUO-Pp9SXfJDoD1-qSayF_EN2g7kTMckGIJfhgpEBW5fQAgFgiFdJ2t6xzP2Azzxg)





