# Reproducible Research Example

This is a simple example of training a machine learning model for reproducible research. 


## Installation

```bash
micromamba env create -f environment.yml
```

## Usage


```bash
torchrun main_pretrain.py --config configs/simclr_stl.yml --data_path ~/gent/data --output_dir outputs/simclr_stl 
```