import json
import random
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import torch
torch.cuda.empty_cache()
import tifffile
import tensorflow as tf

from model_utils import UNET_SERESNEXT101
from hook_utils import Hook
from image_param_utils import ImageParam
from feature_extractor_utils import FeatureExtractor
from umap_utils import normalize_layout, reduce_dim_umap, plot_umap

with open('config.json') as config_file:
    config = json.load(config_file)
    
device = torch.device("cpu")

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

def pad_image(image, side):
        image_h = image.shape[0]
        image_w = image.shape[1]
        if image_h < side and image_w < side:
                new_image = np.zeros(shape=(side, side, 3))
                new_image[0:image_h, 0:image_w, :] = image
                padding = image[0: (side-image_h), 0: (side-image_w), :]
                new_image[image_h:side, image_w:side, :] = padding
        elif image_h > side and image_w > side:
                new_image = image[0:side, 0:side, :]
        else:
                new_image = image
        return new_image

LOAD_LOCAL_WEIGHT_PATH_LIST = {}
for seed in config['split_seed_list']:
    LOAD_LOCAL_WEIGHT_PATH_LIST[seed] = []
    for fold in config['FOLD_LIST']:
        LOAD_LOCAL_WEIGHT_PATH_LIST[seed].append(opj(config['BASE_PATH']+config['MODEL_PATH'],f'model_seed{seed}_fold{fold}_bestscore.pth'))

model_list = {}
for seed in config['split_seed_list']:
    model_list[seed] = []
    for path in LOAD_LOCAL_WEIGHT_PATH_LIST[seed]:
        print("Loading weights from %s" % path)
        
        model = build_model(resolution=(None,None), #config['resolution'], 
                            deepsupervision=config['deepsupervision'], 
                            clfhead=config['clfhead'],
                            load_weights=False).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        model_list[seed].append(model) 

# Print all layers
# layers = []
# for name, mod in model.named_modules():
#     print (name)
#     layers.append(name)

# Get one specific layer
layers  = ['encoder4.2.se_module.avg_pool']

hooks = []
for layer in layers:
    hook = Hook(model, layer)
    hooks.append(hook)

def make_custom_func(layer_number = 0): 
    def custom_func(layer_outputs):
        loss = layer_outputs[layer_number].mean()
        return -loss
    return custom_func

custom_func = make_custom_func(layer_number = 0)

metadata = pd.read_csv(config['BASE_PATH'] + config['DATA_PATH'] + config['METADATA'])
print (len(metadata))

organs = []
data_sources = []
file_names = []
all_features = []
for idx,row in metadata.iterrows():
    image_path = config['BASE_PATH'] + config['DATA_PATH'] + "images/" + row['tissue_name'] + "/" + row['filename'] + ".tif"
    print ("Image ",image_path)
    org_image = tifffile.imread(image_path)
    image = torch.Tensor(org_image).permute(-1,0,1).unsqueeze(0).to(device)
    resnet_features = FeatureExtractor(model, layers)
    # Tom's model can't process some images of K2 data, use exception handling for that
    try:
        features = resnet_features(image)
    except:
        print ("Image couldn't be processed. Image shape ", org_image.shape)
        continue
    all_features.append(features[layers[0]].detach().numpy())
    organs.append(row['tissue_name'])
    if row['data_type'] == "public" or row['data_type'] == "private":
        data_sources.append("HPA")
    else:
        data_sources.append("HUBMAP")
    file_names.append(row['filename'])

activations = np.array(all_features)
print (activations.shape)
shape = 1
for s in activations.shape[1:]:
    shape *= s

flattened_activations = activations.reshape(activations.shape[0], shape)
activations_reduced = reduce_dim_umap(flattened_activations)
activations_normalized = normalize_layout(activations_reduced)

with open('activations_all.npy', 'wb') as f:
    np.save(f, activations_normalized)

xs = activations_normalized[:, 0]
ys = activations_normalized[:, 1]
# activations_normalized = np.load('activations_5.npy')
plot_umap(xs, ys, data_sources, organs, file_names, layers[0])
print ("Activations plotted.")




    
    
