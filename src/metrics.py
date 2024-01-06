import numpy as np

class PixelwiseMetrics(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.count = 0
        self.data = {
            "pixelclass_" + str(i): {"correct": 0, "predicted": 0, "total": 0}
            for i in range(num_classes)
        }

    def add_batch(self, y, y_hat):
        self.count += 1

        for c in range(self.num_classes):
            class_data = self.data["pixelclass_" + str(c)]
            preds_c = y_hat == c
            targs_c = y == c
            num_correct = (preds_c * targs_c).sum().cpu().detach().numpy()
            num_predicted = preds_c.sum().cpu().detach().numpy()
            num_pixels = targs_c.sum().cpu().detach().numpy()

            class_data["correct"] += num_correct
            class_data["predicted"] += num_predicted
            class_data["total"] += num_pixels

    def get_classwise_accuracy(self):
        cw_acc = {
            k: el['correct'] / el['total'] if el['total'] > 0 else 0.0
            for k, el in self.data.items()
        }
        return cw_acc

    def get_average_accuracy(self):
        cw_acc = self.get_classwise_accuracy()
        return np.mean(list(cw_acc.values()))

    def get_IoU(self):
        iou_per_class = {}
        for c in range(self.num_classes):
            class_data = self.data["pixelclass_" + str(c)]
            intersection = class_data["correct"]
            union = class_data["predicted"] + class_data["total"] - intersection
            iou_per_class["class_" + str(c)] = intersection / union if union != 0 else 0
        return iou_per_class

    def get_mIoU(self):
        iou_per_class = self.get_IoU()
        return np.mean(list(iou_per_class.values()))

    def get_conf_matrix(self):
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for c_true in range(self.num_classes):
            for c_pred in range(self.num_classes):
                preds_c = y_hat == c_pred
                targs_c = y == c_true
                conf_matrix[c_true, c_pred] += (preds_c * targs_c).sum().cpu().detach().numpy()
        return conf_matrix

