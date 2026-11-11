# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2025-present, Kim et al.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

import os

from models.build import build_model, load_best_model, build_norm_module
from utils.parser import parse_args, load_config
from datasets.build import update_cfg_from_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices
from tta.tafas import build_adapter
import tta.cosa as cosa
import tta.petsa as petsa
import tta.dynatta as dynatta
from config import get_norm_module_cfg


def checkpoint_result_suffix(checkpoint_dir):
    normalized = checkpoint_dir.replace("\\", "/").rstrip("/")
    marker = "./checkpoints/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return os.path.join(parts[-2], parts[-1])
    return parts[0] if parts else "checkpoint"


def main():
    args = parse_args()
    cfg = load_config(args)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    
    cfg.RESULT_DIR = os.path.join(cfg.RESULT_DIR, checkpoint_result_suffix(cfg.TRAIN.CHECKPOINT_DIR))
    
    if not os.path.exists(cfg.RESULT_DIR):
        os.makedirs(cfg.RESULT_DIR)


    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)


    with open(os.path.join(cfg.RESULT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    
    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None

    if cfg.TRAIN.ENABLE:
        # build trainer
        trainer = build_trainer(cfg, model, norm_module=norm_module)
        trainer.train()
        
    if cfg.TTA.ENABLE or cfg.TEST.ENABLE:
        model = load_best_model(cfg, model)
        if cfg.NORM_MODULE.ENABLE:
            norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
    
    if cfg.TTA.ENABLE:
        if 'TAFAS' in cfg.RESULT_DIR:
            # print("TAFAS")
            adapter = build_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            adapter.count_parameters()
        elif 'PETSA' in cfg.RESULT_DIR:
            # print("PETSA")
            adapter = petsa.build_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            adapter.count_parameters()
        elif 'SIMPLE' in cfg.RESULT_DIR:
            # print("SIMPLE ADAPTER")
            adapter = cosa.build_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            adapter.count_parameters()
        elif 'DYNATTA' in cfg.RESULT_DIR:
            # print("DYNATTA ADAPTER")
            adapter = dynatta.build_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            adapter.count_parameters()


    if cfg.TEST.ENABLE:
        predictor = Predictor(cfg, model, norm_module=norm_module)
        predictor.predict()


if __name__ == '__main__':
    main()
