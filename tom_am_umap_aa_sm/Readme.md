Repository structure:
|---tom_am_umap_aa_sm
      |---tom_umap.py : Plot activations from tom’s model in 2D
      |---tom_activation_maximization.py : Use activation maximization on random image or training image using tom’s model
      |---tom_activation_atlas.py : Create activation atlas using tom’s model
      |---tom_saliency_map.py : Create a saliency map for an intermediate layer
      
      
Naming scheme:
  All the files starting with "tom" are the main files, and the name mentions what the end result of running that file should be. 
  The remaining file, generally ending with utils, contains functions used by the main files. 
