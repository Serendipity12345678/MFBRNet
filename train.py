import torch
import random
import numpy as np
import torch.nn.functional as F
from utils.dataLoader import TrainDataset
from utils.LRScheduler import CosineDecay
from config import Config
from tqdm import tqdm
import os


def structure_loss(logits, target, eps=1e-7, alpha=0.6, beta=0.4, k=5.0):
	target = target.float()
	weit = 1.0 + k * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
	bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
	wbce = (weit * bce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + eps)
	prob = torch.sigmoid(logits)
	inter = (weit * prob * target).sum(dim=(2, 3))
	union = (weit * (prob + target - prob * target)).sum(dim=(2, 3))
	wiou = 1.0 - (inter + eps) / (union + eps)

	return (alpha * wbce + beta * wiou).mean()

def edgeLoss(logits, edge_target):

    edge_target = edge_target.float()
    bce = F.binary_cross_entropy_with_logits(logits,edge_target,reduction="none")
    dims = tuple(range(2, edge_target.ndim))
    pos_ratio = edge_target.mean(dim=dims, keepdim=True)
    neg_ratio = 1.0 - pos_ratio
    eps = torch.finfo(edge_target.dtype).eps
    has_pos = (pos_ratio > 0).to(edge_target.dtype)
    has_neg = (neg_ratio > 0).to(edge_target.dtype)
    num_classes = (has_pos + has_neg).clamp_min(1.0)

    w_pos = has_pos / (num_classes * pos_ratio.clamp_min(eps) )
    w_neg = has_neg / (num_classes * neg_ratio.clamp_min(eps))

    weight = (w_pos * edge_target+ w_neg * (1.0 - edge_target))
    loss = (bce * weight).mean()

    return loss

def train():#
	global model, train_datald, optimizer, cfg, scheduler
	save_dir = "save_pth"
	os.makedirs(save_dir, exist_ok=True)
	best_loss=float('inf')

	for epoch in range(cfg.epochs):
		model.train()
		loss_iter = []

		for step, (img, depth, mask, edge) in enumerate(tqdm(train_datald), start=1):

			optimizer.zero_grad()
			img = img.to(cfg.device)
			depth = depth.to(cfg.device)
			mask = mask.to(cfg.device)
			edge = edge.to(cfg.device)

			pred1_up, pred2_up, pred3_up, obj_map, edge_map = model(img, depth)

			pred1_up = F.interpolate(pred1_up, size=mask.shape[2:], mode="bilinear", align_corners=False)
			pred2_up = F.interpolate(pred2_up, size=mask.shape[2:], mode="bilinear", align_corners=False)
			pred3_up = F.interpolate(pred3_up, size=mask.shape[2:], mode="bilinear", align_corners=False)

			sal_loss1 = structure_loss(pred1_up, mask)
			sal_loss2 = structure_loss(pred2_up, mask)
			sal_loss3 = structure_loss(pred3_up, mask)

			obj_map_up = F.interpolate(obj_map, size=mask.shape[2:], mode='bilinear', align_corners=False)
			edge_map_up = F.interpolate(edge_map, size=edge.shape[2:], mode='bilinear', align_corners=False)

			obj_loss = structure_loss(obj_map_up, mask, k=2.0)
			edge_loss = edgeLoss(edge_map_up, edge)
			sal_loss = 1.0 * sal_loss1 + 0.5 * sal_loss2 + 0.35 * sal_loss3
			sal_loss = sal_loss / (1.0 + 0.5 + 0.35)

			loss = sal_loss + 0.2 * obj_loss + edge_loss

			loss.backward()
			optimizer.step()
			loss_iter.append(loss.item())

		print(f'Epoch: {epoch + 1}, LR: {np.round(scheduler.get_lr(), 8)}, '
			  f': {np.round(np.mean(loss_iter), 8)}')

		scheduler.step()
		current_loss = np.mean(loss_iter)
		if current_loss < best_loss:
			best_loss = current_loss
			save_path = os.path.join(save_dir, "best.pth")
			torch.save(model.state_dict(), save_path)
			print(f'New best model saved at epcoh {epoch + 1} with best loss{best_loss}')

		if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
			torch.save(model.state_dict(), f'save_pth/epoch_{epoch + 1}.pth')




if __name__ == '__main__':
	seed = 123456
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False

	cfg = Config()

	from Model.MFBRNet import Net


	model = Net().to(cfg.device)
	model.encoder_rgb.load_state_dict(torch.load('./smt_tiny.pth',weights_only=False)['model'])
	model.encoder_d.load_state_dict(torch.load('./smt_tiny.pth', weights_only=False)['model'])


	train_dataset = TrainDataset(image_root=cfg.dp.train_imgs, gt_root=cfg.dp.train_masks,
								 depth_root=cfg.dp.train_depth, trainsize=cfg.trainsize,
								 edge_root=cfg.dp.train_edges)
	train_datald = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=cfg.batch_size,
											   shuffle=True,
											   num_workers=cfg.num_workers,
											   pin_memory=True)


	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
	scheduler = CosineDecay(optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs)

	train()
