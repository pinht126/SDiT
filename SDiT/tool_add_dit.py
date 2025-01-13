import sys
import os

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import *
from SDiT.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

def verify_requires_grad(target_dict, pretrained_weights_keys):
    for k in pretrained_weights_keys:
        if k in target_dict:
            tensor = target_dict[k]
            if tensor.requires_grad:
                pass#print(f'Error: Weight {k} has requires_grad=True')
            else:
                pass
                #print(f'Correct: Weight {k} has requires_grad=False')
        else:
            print(f'Error: Weight {k} not found in target_dict')

model = create_model(config_path='./models/sdit.yaml')
pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()
target_dict = {}
for k in scratch_dict.keys():
    is_dit, name = get_node_name(k, 'dit_')
    #print(k)
    #import pdb; pdb.set_trace()
    if is_dit:
        weight_tensor = scratch_dict[k].clone()
        weight_tensor.requires_grad = True  # Ensure requires_grad is False
        target_dict[k] = weight_tensor
        print(f'These weights are newly added: {k}')
    elif k in pretrained_weights:
        weight_tensor = pretrained_weights[k].clone()
        weight_tensor.requires_grad = False  # Ensure requires_grad is False
        target_dict[k] = weight_tensor
        #import pdb; pdb.set_trace()
        print(f'These weights are added: {k}')
    else:
        weight_tensor = scratch_dict[k].clone()
        weight_tensor.requires_grad = True  # Ensure requires_grad is False
        target_dict[k] = weight_tensor
        print(f'????: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
#import pdb; pdb.set_trace()
verify_requires_grad(target_dict, pretrained_weights)
print('Done.')
