'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-09-24 19:47:30
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "mmi"
model = "fno"
exp_name = "train_random"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    mixup, loss, loss_norm, aux_loss, aux_loss_w, aux_loss_norm, lr, enc, n_data, n_layer, id = args
    data_list = [f"rHz_{i}" for i in range(n_data)]
    with open(os.path.join(root, f'rHz{n_data}_mixup-{mixup}_loss-{loss}_enc-{enc}_nl-{n_layer}_tv-{aux_loss_w:.4f}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir=random_size{n_data}",
            f"--dataset.test_ratio=0.1",
            f"--plot.interval=50",
            f"--plot.dir_name={model}_{exp_name}_rHz{n_data}_mixup-{mixup}_{loss}_{enc}_nl-{n_layer}_tv-{aux_loss_w:.4f}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob={mixup}",
            f"--criterion.name={loss}",
            f"--criterion.norm={loss_norm}",
            f"--aux_criterion.{aux_loss}.weight={aux_loss_w}",
            f"--aux_criterion.{aux_loss}.norm={aux_loss_norm}",
            f"--optimizer.lr={lr}",
            f"--model.pos_encoding={enc}",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--model.kernel_list={[32]*n_layer}",
            f"--model.kernel_size_list={[1]*n_layer}",
            f"--model.padding_list={[0]*n_layer}",
            f"--model.mode_list={[(10, 32)]*n_layer}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [[1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 1]] 


    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
