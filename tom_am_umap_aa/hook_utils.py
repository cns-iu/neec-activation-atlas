import torch
import numpy as np

class Hook():
    def __init__(self, model, layer_id, backward=False):
        self.model = model
        module = dict([*self.model.named_modules()])[layer_id]
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()