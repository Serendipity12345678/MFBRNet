import os
import cv2
from tqdm import tqdm
from config import Config
from utils.metrics import EvaluationMetrics


def evaluate(pred_root, dataset, cfg):
    metric = EvaluationMetrics()

    pred_dir = os.path.join(pred_root, dataset)
    mask_dir = getattr(cfg.dp, f"test_{dataset}_masks")

    names = sorted(os.listdir(pred_dir))

    for name in tqdm(names, desc=dataset):
        pred = cv2.imread(os.path.join(pred_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)

        assert pred.shape == mask.shape
        metric.step(pred, mask)

    return metric.get_results()


if __name__ == "__main__":
    cfg = Config()

    pred_root = 'prediction_maps'

    datasets = ["CHAMELEON", "CAMO", "COD10K", "NC4K"]

    for dataset in datasets:
        res = evaluate(pred_root, dataset, cfg)

        if dataset == "CHAMELEON":
            print('path:', pred_root)
        print(f"\n{dataset}:")
        print(f"  SM      : {res['sm']:.4f}")
        print(f"  EM-Mean : {res['emMean']:.4f}")
        print(f"  EM-Max  : {res['emMax']:.4f}")
        print(f"  EM-Adp  : {res['emAdp']:.4f}")
        print(f"  FM-Mean : {res['fmMean']:.4f}")
        print(f"  FM-Max  : {res['fmMax']:.4f}")
        print(f"  FM-Adp  : {res['fmAdp']:.4f}")
        print(f"  WFM     : {res['wfm']:.4f}")
        print(f"  MAE     : {res['mae']:.4f}")