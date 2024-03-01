#!/bin/bash -l
# SLURM SUBMIT SCRIPT

#SBATCH --nodelist=node-gpu01
#SBATCH --gres=gpu:1    # Request N GPUs per machine

lr=1e-4
batch_size=32
accumulate_grad_batches=4 #8

python main.py fit \
--data=TwentyQuestions \
--data.batch_size=$batch_size \
--data.n_traj_eval=64 \
--model=FilteredBehaviouralCloning \
--model.lr=$lr \
--model.filter=0.1 \
--trainer.fast_dev_run=True \
--trainer.max_epoch=100 \
--trainer.accumulate_grad_batches=$accumulate_grad_batches \
--trainer.logger=WandbLogger \
--trainer.logger.init_args.project="TwentyQuestions" \
--trainer.logger.init_args.name="FBC-lr$lr" \
--trainer.val_check_interval=500