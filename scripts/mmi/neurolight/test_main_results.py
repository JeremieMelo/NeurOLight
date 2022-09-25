'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-18 00:48:16
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-09-24 19:49:19
'''
import os
import numpy as np
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "mmi"
model = "neurolight"
exp_name = "train_random"
# exp_name = "train_random_slot"
root = f"log/{dataset}/{model}/{exp_name}_main_test"
script = "test.py"
config_file = f"configs/{dataset}/{model}/{exp_name}/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    dataset, n_data, data_dir, n_layer, test_mode, mixup, ckpt, id = args
    # n_layer = 12
    # n_data = 5
    # data_list = [f"rHz_{i}" for i in range(n_data)]

    data_list = [f"{dataset}_{i}" for i in range(n_data)]
    with open(
        os.path.join(root, f"{dataset}{n_data}_{test_mode}_mixup-{mixup}_id-{id}.log"), "w"
    ) as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir={data_dir}",
            f"--dataset.test_ratio=0.1",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--plot.interval=50",
            f"--plot.dir_name=test_{model}_{exp_name}_{dataset}{n_data}_mixup-{mixup}_cmae_exp_nl-{n_layer}_{test_mode}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob=0",
            f"--dataset.augment.random_vflip_ratio=0",
            f"--criterion.name=cmae",
            f"--criterion.norm=True",
            f"--aux_criterion.tv_loss.weight=0",
            f"--aux_criterion.tv_loss.norm=False",
            f"--optimizer.lr=0.002",
            f"--model.pos_encoding=exp",
            f"--model.kernel_list={[64]*n_layer}",
            f"--model.kernel_size_list={[1]*n_layer}",
            f"--model.padding_list={[1]*n_layer}",
            f"--model.mode_list={[(40, 70)]*n_layer}",
            f"--model.with_cp=False",
            f"--checkpoint.resume=1",
            f"--checkpoint.restore_checkpoint={ckpt}",
            f"--run.test_mode={test_mode}",
            f"--run.test_split=test",
            f"--run.test_random_state=10000",
            f"--model.aug_path=False",
        ]
        logger.info(f"running command {' '.join(pres + exp)}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first


    tasks = [
        ("rHz", 5, "random_size5", 12, "mm", 1, "./checkpoint/mmi/neurolight/train_random/NeurOLight2d.pt", 1),
    ]
    # tasks = [
    #     ("slot_rHz", 10, "random_size10_slot", 16, "mm", 1, "./checkpoint/mmi/neurolight/train_random_etched/NeurOLight2d.pt", 1),
    # ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
