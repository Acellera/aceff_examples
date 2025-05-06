# Examples for using AceFF neural network potential.

This python package provides an ASE calculator for AceFF and a OpenMM-ML potential implementaion.


There are 3 example notebooks in the examples folder.
They can be run in Google Colab or locally.
- Single point energy calculator with ASE.
    - `./notebooks/aceff_single_point_calculation.ipynb`
- Running MD for a small molecule in vacuum with OpenMM.
    - `./notebooks/aceff_MD_example.ipynb`
- Running MD for a protein ligand complex using MM/ML mechanical embedding scheme with OpenMM. 
    - `./notebooks/aceff_protein_ligand.ipynb`


## Python Environment for running locally
You will need to install:
- `torchmd-net`

and depending on wich example
- `openmm-torch openmm-ml`
- `ase`

For the MD examples you will need a CUDA GPU.

The AceFF model files need to be downloaded from HuggingFace: 
https://huggingface.co/Acellera
