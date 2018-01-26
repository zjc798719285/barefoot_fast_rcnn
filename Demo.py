from models import FootNet_v3
from utils import *
from config import cfg
from train import train

model = FootNet_v3.FootNet_v3()
train_img, train_roi, test_img, test_roi = load_data(cfg.train_path, cfg.test_path)
# print(test_roi)
train(img=train_img, ground_truth=train_roi, test_img=test_img, test_roi=test_roi, model=model, params=cfg)
