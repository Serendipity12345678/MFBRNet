import py_sod_metrics as metrics
import numpy as np


class EvaluationMetrics:
    def __init__(self):
        self.SM = metrics.Smeasure()
        self.EM = metrics.Emeasure()
        self.FM = metrics.Fmeasure()
        self.WFM = metrics.WeightedFmeasure()
        self.MAE = metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        """
        pred, gt: uint8 [0, 255]
        """
        self.SM.step(pred=pred, gt=gt)
        self.EM.step(pred=pred, gt=gt)
        self.FM.step(pred=pred, gt=gt)
        self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

    def get_results(self):
        # ---------- S-measure ----------
        sm = self.SM.get_results()["sm"]

        # ---------- E-measure ----------
        em_res = self.EM.get_results()["em"]
        emMean = em_res["curve"].mean()
        emMax = em_res["curve"].max()
        emAdp = em_res["adp"]

        # ---------- F-measure ----------
        fm_res = self.FM.get_results()["fm"]
        fmMean = fm_res["curve"].mean()
        fmMax = fm_res["curve"].max()
        fmAdp = fm_res["adp"]

        # ---------- Weighted F ----------
        wfm = self.WFM.get_results()["wfm"]

        # ---------- MAE ----------
        mae = self.MAE.get_results()["mae"]

        return {
            "sm": sm,

            "emMean": emMean,
            "emMax": emMax,
            "emAdp": emAdp,

            "fmMean": fmMean,
            "fmMax": fmMax,
            "fmAdp": fmAdp,

            "wfm": wfm,
            "mae": mae,
        }


class EvaluationMetricsV2:
    def __init__(self):
        self.SM = metrics.Smeasure()
        self.EM = metrics.Emeasure()
        self.FM = metrics.Fmeasure()
        self.WFM = metrics.WeightedFmeasure()
        self.MAE = metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        """
        pred: [0, 255]
        gt: [0, 255]
        """
        self.SM.step(pred=pred, gt=gt)
        self.EM.step(pred=pred, gt=gt)
        self.FM.step(pred=pred, gt=gt)
        self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

    def get_results(self):
        # S-measure, default alpha=0.5
        sm = self.SM.get_results()["sm"]
        # mean E-measure and E-measure Curve
        _em = self.EM.get_results()["em"]
        em_curve = np.flip(_em["curve"])
        emMean = _em['curve'].mean()
        emMax = _em['curve'].max()
        emAdp = _em['adp']
        # F-measure curve and PR-curve
        _fm = self.FM.get_results()
        # F-measure
        fm = _fm["fm"]
        fmMean = fm["curve"].mean()
        fmMax = fm["curve"].max()
        fmAdp = fm["adp"]
        fm_curve = np.flip(fm["curve"])
        pr = _fm["pr"]
        p = np.flip(pr["p"])
        r = np.flip(pr["r"])
        # weighted F-measure
        wfm = self.WFM.get_results()["wfm"]
        # mean Absolute Error
        mae = self.MAE.get_results()["mae"]
        # return sm, emMean, emAdp, wfm, mae
        return {
            'sm': sm,

            'emMean': emMean,
            'emAdp': emAdp,
            'emMax': emMax,
            'em_curve': em_curve,

            'fmMean': fmMean,
            'fmMax': fmMax,
            'fmAdp': fmAdp,
            'fm_curve': fm_curve,

            'wfm': wfm,
            'mae': mae,

            'p': p,
            'r': r
        }

class EvaluationMetricsV3:
    def __init__(self):
        self.SM = metrics.Smeasure()
        self.EM = metrics.Emeasure()
        self.FM = metrics.Fmeasure()
        self.WFM = metrics.WeightedFmeasure()
        self.MAE = metrics.MAE()

        # +++ 新增：用于 mDice / mIoU +++
        self._dice_sum = 0.0
        self._iou_sum = 0.0
        self._n = 0

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        """
        pred: [0, 255] uint8 gray
        gt:   [0, 255] uint8 gray
        """
        self.SM.step(pred=pred, gt=gt)
        self.EM.step(pred=pred, gt=gt)
        self.FM.step(pred=pred, gt=gt)
        self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

        # +++ 新增：Dice / IoU（用 0.5 阈值 -> 128）+++
        pred_bin = (pred > 128)
        gt_bin = (gt > 128)

        tp = np.logical_and(pred_bin, gt_bin).sum()
        fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
        fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)

        self._dice_sum += float(dice)
        self._iou_sum += float(iou)
        self._n += 1

    def get_results(self):
        sm = self.SM.get_results()["sm"]

        _em = self.EM.get_results()["em"]
        em_curve = np.flip(_em["curve"])
        emMean = _em["curve"].mean()
        emMax = _em["curve"].max()
        emAdp = _em["adp"]

        _fm = self.FM.get_results()
        fm = _fm["fm"]
        fmMean = fm["curve"].mean()
        fmMax = fm["curve"].max()
        fmAdp = fm["adp"]
        fm_curve = np.flip(fm["curve"])
        pr = _fm["pr"]
        p = np.flip(pr["p"])
        r = np.flip(pr["r"])

        wfm = self.WFM.get_results()["wfm"]
        mae = self.MAE.get_results()["mae"]

        # +++ 新增：mDice / mIoU +++
        mDice = self._dice_sum / max(self._n, 1)
        mIoU = self._iou_sum / max(self._n, 1)

        return {
            "mDice": mDice,
            "mIoU": mIoU,

            "sm": sm,

            "emMean": emMean,
            "emAdp": emAdp,
            "emMax": emMax,
            "em_curve": em_curve,

            "fmMean": fmMean,
            "fmMax": fmMax,
            "fmAdp": fmAdp,
            "fm_curve": fm_curve,

            "wfm": wfm,
            "mae": mae,

            "p": p,
            "r": r
        }
