import json
import os
# import torch

CSV_DELIMETER = ';'


# Create directorie
def create_directorie(d):
    if d and not os.path.exists(d):
        os.makedirs(d)

    return d


# Save configs as json
def save_config(log_path, cfg, filename):
    dic = convert_config_to_dic(cfg)
    path = os.path.join(log_path, '%s.json' % filename)
    with open(path, 'w') as f:
        json.dump(vars(dic), f)


# Convert configs to dictionary
def convert_config_to_dic(cfg):
    d = dict(cfg._sections)
    for k in d:
        d[k] = dict(d[k])

    return d
