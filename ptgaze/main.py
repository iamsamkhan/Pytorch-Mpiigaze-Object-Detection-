import argparse
import logging
import pathlib
import warnings

import torch
from omegaconf import DictConfig, OmegaConf

from .demo import Demo
from .utils import (check_path_all, download_dlib_pretrained_model,
                    download_ethxgaze_model, download_mpiifacegaze_model,
                    download_mpiigaze_model, expanduser_all,
                    generate_dummy_camera_params)

logger = logging.getLogger(__name__)


def load_config(video_path,output_dir,display):

    download_ethxgaze_model()

    package_root = pathlib.Path(__file__).parent.resolve()
    path = package_root / 'data/configs/eth-xgaze.yaml'
    config = OmegaConf.load(path)
    config.PACKAGE_ROOT = package_root.as_posix()

    if config.face_detector.mode == 'dlib':
        download_dlib_pretrained_model()

    config.device = 'cpu'
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        warnings.warn('Run on CPU because CUDA is not available.')

    config.demo.use_camera = False
    config.demo.video_path = video_path

    config.gaze_estimator.use_dummy_camera_params = True
    if config.gaze_estimator.use_dummy_camera_params:
        generate_dummy_camera_params(config)
 
    config.demo.data_output_dir = output_dir
    config.demo.display_on_screen = display
    
    expanduser_all(config)
    
    OmegaConf.set_readonly(config, True)
    logger.info(OmegaConf.to_yaml(config))

    check_path_all(config)

    return config

def main():    
    video_path = "videos\\An Important Advice-AI Is Advancing Just Start Learning AI Before Its Late.mp4"
    output_dir = "C:\\Users\\Shamshad ahmed\\pytorch_mpiigaze_demo-master\\output"

    config = load_config(video_path,output_dir,display=False)

    demo = Demo(config)
    demo.run()
    check_path_all(config)

    demo = Demo(config)
    demo.run()
