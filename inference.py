import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from Model.MFBRNet import Net
from config import Config
from utils.dataLoader import TestDataset


def inference(datasets):
	global model, cfg
	model.eval()

	for dataset in datasets:
		assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
		save_path = os.path.join('prediction_maps', dataset)
		os.makedirs(save_path, exist_ok=True)

		test_dataset = TestDataset(
			image_root=getattr(cfg.dp, f'test_{dataset}_imgs'),
			gt_root=getattr(cfg.dp, f'test_{dataset}_masks'),
			depth_root=getattr(cfg.dp, f'test_{dataset}_depth'),
			testsize=cfg.trainsize,
			edge_root=None
		)
		for idx, (img, depth, _, gt, _, name) in enumerate(tqdm(test_dataset)):
			img = img.unsqueeze(0).to(cfg.device)
			depth = depth.unsqueeze(0).to(cfg.device)

			out1, out2, out3,obj, edge = model(img, depth)
			out1 = F.interpolate(out1, size=gt.shape[1:], mode='bilinear', align_corners=False)
			out1 = torch.sigmoid(out1) * 255
			out1 = out1.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

			cv2.imwrite(os.path.join(save_path, name), out1)




if __name__ == '__main__':

	pth_path = 'best.pth'
	cfg = Config()
	model = Net().to(cfg.device)

	state_dict = torch.load(pth_path, map_location=cfg.device)
	model.load_state_dict(state_dict)

	datasets = ["CHAMELEON", "CAMO", "COD10K", "NC4K"]
	inference(datasets)