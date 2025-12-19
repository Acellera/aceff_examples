# Examples for using AceFF neural network potential.

There are 5 example notebooks
They can be run in Google Colab or locally.
- Single point energy calculator with ASE.
    - `./notebooks/aceff_single_point_calculation.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_single_point_calculation.ipynb)
- Running MD for a small molecule in vacuum with OpenMM.
    - `./notebooks/aceff_MD_example.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_MD_example.ipynb)
- Running MD for a protein ligand complex using MM/ML mechanical embedding scheme with OpenMM. 
    - `./notebooks/aceff_protein_ligand.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_protein_ligand.ipynb)
- Batched minimization
    - `./notebooks/aceff_batched_minimization.ipynb`
- Batched MD
    - `./notebooks/aceff_batched_MD.ipynb`



## Python Environment for running locally
You will need to install:
- `torchmd-net`

and depending on which example
- `openmm-torch openmm-ml`
- `ase`
- `rdkit`

For the MD examples you will need a CUDA GPU.

The AceFF model files need to be downloaded from HuggingFace: 
https://huggingface.co/Acellera
