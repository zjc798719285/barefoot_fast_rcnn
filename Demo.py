from models import FootNet_v3
from utils import *
from config import cfg
from train2 import train

model = FootNet_v3.FootNet_v3()
train_x, train_roi, test_x, test_roi = load_data(cfg.train_path, cfg.test_path)
train(img=train_x, ground_truth=train_roi, test_img=test_x, test_roi=test_roi, model=model, params=cfg)
