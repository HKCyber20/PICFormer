import yaml
import argparse
import os

class Config:
    def __init__(self, cfg_path="config/config.yaml"):
        with open(cfg_path, "r", encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

    def __getattr__(self, name):
        return self.cfg.get(name)
    
    def __getitem__(self, key):
        return self.cfg[key]
    
    def get(self, key, default=None):
        return self.cfg.get(key, default)

def parse_args():
    parser = argparse.ArgumentParser(description='PICFormer Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoint',
                        help='path to save checkpoints')
    parser.add_argument('--resume', type=str, default='',
                        help='path to resume checkpoint')
    return parser.parse_args()

cfg = Config() 