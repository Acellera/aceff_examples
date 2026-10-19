# Examples for using AceFF neural network potential.

The ASE and batched PyTorch examples use [AceFF-2-OMOL](https://huggingface.co/Acellera/AceFF-2-OMOL). The OpenMM examples use [AceFF-2.0](https://huggingface.co/Acellera/AceFF-2.0) until AceFF-2-OMOL support is released in OpenMM-ML.

There are 4 example notebooks
They can be run in Google Colab or locally.
- Single point energy calculator with ASE.
    - `./notebooks/aceff_single_point_calculation.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_single_point_calculation.ipynb)

- Batch minimization and molecular dynamics with PyTorch and TorchMD.
    - `./notebooks/aceff_batched_minimization_and_md.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_batched_minimization_and_md.ipynb)


- Running MD for a small molecule in vacuum with OpenMM.
    - `./notebooks/aceff_openmm_md_example.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_openmm_md_example.ipynb)
- Running MD for a protein ligand complex using MM/ML mechanical embedding scheme with OpenMM. 
    - `./notebooks/aceff_openmm_protein_ligand.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Acellera/aceff_examples/blob/main/notebooks/aceff_openmm_protein_ligand.ipynb)


    
## Python Environment for running locally
You will need to install:
- `torchmd-net` (3.1.0 or newer for AceFF-2-OMOL)

and depending on which example
- `openmm-torch openmm-ml`
- `ase`
- `rdkit`
- `torchmd`

For the MD examples you will need a CUDA GPU.

The AceFF model files can be downloaded from HuggingFace: 
https://huggingface.co/Acellera
