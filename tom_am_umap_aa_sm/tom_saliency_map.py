import json
import random
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import torch
import tifffile
from model_utils import UNET_SERESNEXT101
from image_param_utils import ImageParam
from saliency_map_utils import Dreamer
import matplotlib.pyplot as plt

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

image_path = "/N/slate/soodn/kaggle_data_package/kaggle_data_multiftu/data/images/lung/41507_88065_A_2_4_lung.tif"
# org_image = tifffile.imread(image_path)
# image = torch.Tensor(org_image).permute(-1,0,1).unsqueeze(0)
# mask = model(image)
param = ImageParam(image = image_path, device= "cuda")

image_param = dream_model.render_saliency(
    image_parameter= param,
    layers = layers,
    lr = 2e-4,
    # weight_decay= 1e-1,
    iters = 56,  ## mild
    custom_func = None
)

if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        image_param = tf.transpose(image_param, [0, 2, 1])
im = keras.preprocessing.image.array_to_img(
        image_param, data_format="channels_last")
keras.preprocessing.image.save_img(
        "image_param.png", im, data_format="channels_last")

