#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import json
import os


current_dir = os.getcwd()

with open(os.path.join(current_dir, "src/envs/hok/hok_game/conf/gamecore_conf.json"), 'r') as f:
    GC_CONFIG = json.load(f)
    
with open(os.path.join(current_dir, "src/envs/hok/hok_game/conf/natureclient_conf.json"), 'r') as f:
    NC_CONFIG = json.load(f)
    