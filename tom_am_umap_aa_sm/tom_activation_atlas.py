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
from feature_extractor_utils import FeatureExtractor
from umap_utils import normalize_layout, reduce_dim_umap, plot_umap
from render_utils import render_layout


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

canvas = render_layout(model, layers, activations, xs, ys, n_steps=128, grid_size=(3, 3))
print (canvas.shape, np.max(canvas), np.min(canvas))

im = deprocess_image(canvas)
if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        im = tf.transpose(im, [0, 2, 1])
img = keras.preprocessing.image.array_to_img(
        im, data_format="channels_last")
keras.preprocessing.image.save_img(
        "canvas.png", img, data_format="channels_last")





    
    
