import json
import random
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import torch
torch.cuda.empty_cache()

from model_utils import UNET_SERESNEXT101
from image_param_utils import ImageParam
from dream_utils import Dreamer


with open('config.json') as config_file:
    config = json.load(config_file)

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
fix_seed(2020)

def build_model(resolution, deepsupervision, clfhead, load_weights):
    model = UNET_SERESNEXT101(resolution, deepsupervision, clfhead, load_weights)
    return model

LOAD_LOCAL_WEIGHT_PATH_LIST = {}
for seed in config['split_seed_list']:
    LOAD_LOCAL_WEIGHT_PATH_LIST[seed] = []
    for fold in config['FOLD_LIST']:
        LOAD_LOCAL_WEIGHT_PATH_LIST[seed].append(opj(config['BASE_PATH']+config['MODEL_PATH'],f'model_seed{seed}_fold{fold}_bestscore.pth'))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model_list = {}
for seed in config['split_seed_list']:
    model_list[seed] = []
    for path in LOAD_LOCAL_WEIGHT_PATH_LIST[seed]:
        print("Loading weights from %s" % path)
        
        model = build_model(resolution=(None,None), #config['resolution'], 
                            deepsupervision=config['deepsupervision'], 
                            clfhead=config['clfhead'],
                            load_weights=False)
        model.load_state_dict(torch.load(path))
        model.eval()
        model_list[seed].append(model) 

dream_model = Dreamer(model, device = "cpu")
layers  = ['encoder4.2.se_module.avg_pool']
metadata = pd.read_csv(config['BASE_PATH'] + config['DATA_PATH'] + config['METADATA'])
print (len(metadata))

image_path = "/N/slate/soodn/kaggle_data_package/kaggle_data_multiftu/data/images/largeintestine/A001-C-224_patch_1_0_largeintestine.tif"
print (image_path)
param = ImageParam(image = image_path, device= "cuda")

image_param = dream_model.render(
    image_parameter= param,
    layers = layers,
    lr = 2e-4,
    # weight_decay= 1e-1,
    iters = 20,  ## mild
    custom_func = None
)

plt.savefig(image_param, 'img_param.png')
