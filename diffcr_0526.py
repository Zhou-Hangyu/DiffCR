print("Begin srcipt")
import os
os.environ["WANDB_API_KEY"] = "5f04d2ce100707f23b71379f67f28901d496edda"
# os.environ["WANDB_MODE"] = "disabled"

import argparse
import warnings
import torch
import torch.multiprocessing as mp
import wandb
from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

print("Complete loading packages")

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config', type=str, default='config/ours_sigmoid_allclear.json', help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
parser.add_argument('-b', '--batch', type=int, default=4, help='Batch size in every gpu')
parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-P', '--port', default='21012', type=str)
parser.add_argument('--train_dataset', default='sen2mtc', type=str, choices=["sen2mtc", "allclear"])

''' parser configs '''
args, _ = parser.parse_known_args()
opt = Praser.parse(args)

print(f"Using Dataset: {args.train_dataset}")

''' cuda devices '''
gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
opt['world_size'] = 1 

gpu = 0
ngpus_per_node = 1

Util.set_seed(opt['seed'])

''' set logger '''
phase_logger = InfoLogger(opt)
phase_writer = VisualWriter(opt, phase_logger)  
phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

run_name = f"DiffCR_{args.train_dataset}_bs{args.batch}_0526"
wandb.init(project="allclear-diffcr-v1", name=run_name, config=opt)


print('''set networks and dataset''')
'''set networks and dataset'''
phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.

# ======== Using allclear dataset ========
if args.train_dataset == "allclear":
    import sys, os
    if "ck696" in os.getcwd():
        sys.path.append("/share/hariharan/ck696/allclear")
    else:
        sys.path.append("/share/hariharan/cloud_removal/allclear")
    
    from dataset.dataloader_v1 import CRDataset
    from torch.utils.data import DataLoader, Dataset
    
    class CRDatasetWrapper(Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset
    
        def __len__(self):
            return 2376
    
        def __getitem__(self, idx):
            batch = self.original_dataset[idx]
            cond_image = batch["input_images"][(2,1,0),...].reshape(9,256,256) * 2 - 1
            gt_image = batch["target"][(2,1,0),...].reshape(3,256,256) * 2 - 1
            return {"gt_image": gt_image, "cond_image": cond_image, "path": ["", ""]}
    
    import json
    with open('/share/hariharan/cloud_removal/metadata/v3/s2p_tx3_train_2k_v1.json') as f:
        metadata = json.load(f)
        
    for i in range(len(metadata)):
        metadata[f"{i}"]["target"][0][1] = "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4/" + metadata[f"{i}"]["target"][0][1].split("dataset_30k_v4")[1]
        metadata[f"{i}"]["s2_toa"][0][1] = "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4/" + metadata[f"{i}"]["s2_toa"][0][1].split("dataset_30k_v4")[1]
        metadata[f"{i}"]["s2_toa"][1][1] = "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4/" + metadata[f"{i}"]["s2_toa"][1][1].split("dataset_30k_v4")[1]
        metadata[f"{i}"]["s2_toa"][2][1] = "/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4/" + metadata[f"{i}"]["s2_toa"][2][1].split("dataset_30k_v4")[1]
    
    train_data = CRDataset(metadata, 
                        selected_rois="all", 
                        main_sensor="s2_toa", 
                        aux_sensors=[],
                        aux_data=[],
                        format="stp",
                        target="s2p",
                        tx=3,
                        s2_toa_channels=[2,3,4]
                          )
    wrapped_train_data = CRDatasetWrapper(train_data)
    phase_loader = DataLoader(wrapped_train_data, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
# ======== Using allclear dataset ========

networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

print('''set metrics, loss, optimizer and  schedulers''')
''' set metrics, loss, optimizer and  schedulers '''
metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

print('''set model''')
model = create_model(
    opt = opt,
    networks = networks,
    phase_loader = phase_loader,
    val_loader = None,
    losses = losses,
    metrics = metrics,
    logger = phase_logger,
    writer = phase_writer,
)
model.save_current_results_flag = 0
model.wandb = wandb

# ==== For evaluation ==== 
# model.load_networks()
# params = torch.load("./pretrained/diffcr_new.pth")
# model.netG.load_state_dict(params,strict=False)
# model.test()

# ==== For training ==== 
model.train()
model.test()