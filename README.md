# Examples for using AceFF neural network potential.

There are 4 example notebooks
They can be run in Google Colab or locally.
- Single point energy calculator with ASE.
    - `./notebooks/aceff_single_point_calculation.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_single_point_calculation.ipynb)

- Batch minimization and molecular dynamics with PyTorch.
    - `./notebooks/aceff_batched_minimization_and_md.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_batched_minimization_and_md.ipynb)


- Running MD for a small molecule in vacuum with OpenMM.
    - `./notebooks/aceff_openmm_md_example.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_openmm_md_example.ipynb)
- Running MD for a protein ligand complex using MM/ML mechanical embedding scheme with OpenMM. 
    - `./notebooks/aceff_openmm_protein_ligand.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_openmm_protein_ligand.ipynb)


    
## Python Environment for running locally
You will need to install:
- `torchmd-net`

and depending on which example
- `openmm-torch openmm-ml`
- `ase`
- `rdkit`

For the MD examples you will need a CUDA GPU.

The AceFF model files can be downloaded from HuggingFace: 
https://huggingface.co/Acellera
