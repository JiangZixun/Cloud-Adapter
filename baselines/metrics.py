"""Dataset-wide confusion-matrix metrics, reported in percent like MMSeg."""

import torch


METRIC_NAMES = ("aAcc", "mIoU", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall")


def confusion_matrix(prediction, target, num_classes=2, ignore_index=255):
    valid = target != ignore_index
    target, prediction = target[valid].long(), prediction[valid].long()
    if ((target < 0) | (target >= num_classes)).any():
        raise ValueError("Target contains a class outside the configured range")
    return torch.bincount(
        target * num_classes + prediction, minlength=num_classes**2
    ).reshape(num_classes, num_classes)


def segmentation_metrics(matrix, classes):
    matrix = matrix.detach().cpu().double()
    tp = matrix.diag()
    actual, predicted = matrix.sum(1), matrix.sum(0)

    def divide(numerator, denominator):
        return torch.where(denominator > 0, numerator / denominator, float("nan"))

    iou = divide(tp, actual + predicted - tp)
    recall = divide(tp, actual)
    precision = divide(tp, predicted)
    dice = divide(2 * tp, actual + predicted)
    # Algebraically equivalent to MMSeg's precision/recall F-score, including
    # its undefined result when precision or recall has a zero denominator.
    fscore = divide(2 * precision * recall, precision + recall)

    def scalar(value):
        return float(value * 100) if torch.isfinite(value) else None

    values = dict(IoU=iou, Acc=recall, Dice=dice, Fscore=fscore,
                  Precision=precision, Recall=recall)
    result = {"aAcc": scalar(divide(tp.sum(), matrix.sum()))}
    result.update({"m" + key: scalar(torch.nanmean(value)) for key, value in values.items()})
    result["per_class"] = {
        name: {key: scalar(value[i]) for key, value in values.items()}
        for i, name in enumerate(classes)
    }
    result["confusion_matrix"] = matrix.long().tolist()
    return result
