# Effects of long-time training
export WANDB_ENRITY=erow
export WANDB_PROJECT=reproducibility

WANDB_NAME=simclr_stl_e100 torchrun main_pretrain.py --config configs/simclr_stl.yml --data_path ~/gent/data --output_dir outputs/simclr_stl_e100 --epochs 100 

WANDB_NAME=simclr_stl_e200 torchrun main_pretrain.py --config configs/simclr_stl.yml --data_path ~/gent/data --output_dir outputs/simclr_stl_e200 --epochs 200