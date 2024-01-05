from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import albumentations as A
from pathlib import Path
import random
from PIL import Image, ImageFilter

class AlbumentationsToTorchTransform:
    """Take a list of Albumentation transforms and apply them
    s.t. it is compatible with a Pytorch dataloader"""

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, x):
        x_t = self.augmentations(image=x)

        return x_t["image"]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomGrayscale(A.ToGray):
    def __init__(self, always_apply=False, p=0.5):
        super(A.ToGray, self).__init__(always_apply, p)

    def apply(self, img, **params):
        if torch.rand(1).item() < self.p:
            img = np.repeat(img.mean(axis=2, keepdims=True), 12, axis=2)

        return img


def get_batch_corrrelations(scan_embeds_1, scan_embeds_2, device):
    """gets correlations between scan embeddings"""
    batch_size, channels, h, w = scan_embeds_2.shape

    scan_embeds_1 = F.normalize(scan_embeds_1, dim=1).to(device)
    scan_embeds_2 = F.normalize(scan_embeds_2, dim=1).to(device)
    correlation_maps = F.conv2d(scan_embeds_1, scan_embeds_2) / (h * w)
    return correlation_maps


class NCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, batch_similarities):
        ax1_softmaxes = F.softmax(batch_similarities / self.temperature, dim=1)
        ax2_softmaxes = F.softmax(batch_similarities / self.temperature, dim=0)
        softmax_scores = torch.cat(
            (-ax1_softmaxes.diag().log(), -ax2_softmaxes.diag().log())
        )
        loss = softmax_scores.mean()
        return loss


def normalise_channels(scan_img, eps=1e-5):
    # normalize each channel
    scan_min = scan_img.flatten(start_dim=-2).min(dim=-1)[0][:, None, None]
    scan_max = scan_img.flatten(start_dim=-2).max(dim=-1)[0][:, None, None]
    return (scan_img - scan_min) / (scan_max - scan_min + eps)


def save_checkpoint_single_model(
    model, optimiser, val_stats, epochs, save_weights_path
):

    print(f"==> Saving Model Weights to {save_weights_path}")
    state = {
        "model_weights": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "val_stats": val_stats,
        "epochs": epochs,
    }
    # if not os.path.isdir(save_weights_path):
    #    os.mkdir(save_weights_path)
    # previous_checkpoints = glob.glob(save_weights_path + '/ckpt*.pt', recursive=True)
    torch.save(state, save_weights_path)  # + '/ckpt' + str(epochs) + '.pt')
    # for previous_checkpoint in previous_checkpoints:
    #    os.remove(previous_checkpoint)
    return


def save_checkpoint(
    s1_model, s2_model, optimiser, val_stats, epochs, save_weights_path
):

    print(f"==> Saving Model Weights to {save_weights_path}")
    state = {
        "s1_model_weights": s1_model.state_dict(),
        "s2_model_weights": s2_model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "val_stats": val_stats,
        "epochs": epochs,
    }
    # if not os.path.isdir(save_weights_path):
    #    os.mkdir(save_weights_path)
    # previous_checkpoints = glob.glob(save_weights_path + '/ckpt*.pt', recursive=True)
    torch.save(state, save_weights_path)  # + '/ckpt' + str(epochs) + '.pt')
    # for previous_checkpoint in previous_checkpoints:
    #    os.remove(previous_checkpoint)
    return


def get_rank_statistics(similarities_matrix):
    sorted_similarities_values, sorted_similarities_idxs = similarities_matrix.sort(
        dim=1, descending=True
    )
    ranks = []
    for idx, row in enumerate(tqdm(sorted_similarities_idxs)):
        rank = torch.where(row == idx)[0][0]
        ranks.append(rank.cpu())
    ranks = np.array(ranks)
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    top_10 = np.sum(ranks < 10) / len(ranks)
    top_5 = np.sum(ranks < 5) / len(ranks)
    top_1 = np.sum(ranks < 1) / len(ranks)

    ranks_stats = {
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "top_10": top_10,
        "top_5": top_5,
        "top_1": top_1,
    }

    return ranks_stats


def get_dataset_similarities(scan_embeds_1, scan_embeds_2, device, batch_size=50):
    """Gets similarities for entire dataset.
    Splits job into batches to reduce GPU memory"""
    ds_size, channels, h, w = scan_embeds_2.shape
    ds_similarities = torch.zeros(ds_size, ds_size)

    for batch_1_start_idx in tqdm(range(0, ds_size, batch_size)):
        for batch_2_start_idx in range(0, ds_size, batch_size):

            batch_1_end_idx = batch_1_start_idx + batch_size
            batch_2_end_idx = batch_2_start_idx + batch_size
            if batch_2_end_idx >= ds_size:
                batch_2_end_idx = ds_size
            if batch_1_end_idx >= ds_size:
                batch_1_end_idx = ds_size

            correlations = get_batch_corrrelations(
                scan_embeds_1[batch_1_start_idx:batch_1_end_idx],
                scan_embeds_2[batch_2_start_idx:batch_2_end_idx],
                device,
            )
            similarities, _ = torch.max(correlations.flatten(start_dim=2), dim=-1)
            ds_similarities[
                batch_1_start_idx:batch_1_end_idx, batch_2_start_idx:batch_2_end_idx
            ] = similarities
    return ds_similarities


class AverageMeter(object):
    """Computes and stores the average and current values"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def multi_acc(pred, label):
    """compute pixel-wise accuracy across a batch"""
    _, tags = torch.max(pred, dim=1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc


def class_wise_acc(pred, label, results, num_classes=10):
    """add number of correctly classified pixels and total number of pixels
    for each class to `results`"""
    _, tags = torch.max(pred, dim=1)

    for l in range(num_classes):
        if label[label == l].numel() == 0:
            continue
        else:
            corrects = (tags[label == l] == label[label == l]).float()
            results[str(l) + "_correct"] += corrects.sum()
            results[str(l) + "_total"] += corrects.numel()
            # acc = acc * 100
            # results[str(l)].extend(corrects.detach().cpu().numpy().tolist())

    return results


def class_wise_acc_per_img(pred, label, num_classes=10):
    """return class wise accuracy independently for each img in the batch
    assumes pred and label of dim bxnum_classesxhxw and bx1xhxw"""

    _, tags = torch.max(pred, dim=1)
    batch_size = pred.shape[0]

    results = []
    for b in range(batch_size):
        img_results = {}
        for l in range(num_classes):
            if label[b][label[b] == l].numel() == 0:
                # this class is not present in the current image
                continue
            else:
                corrects = (tags[b][label[b] == l] == label[b][label[b] == l]).float()
                img_results[str(l)] = (corrects.sum() / corrects.numel()).item() * 100

        results.append(img_results)

    return results


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dotdictify(d):
    """recursively wrap a dictionary and
    all the dictionaries that it contains
    with the dotdict class
    """
    d = dotdict(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dotdictify(v)
    return d

from sklearn.metrics import confusion_matrix
class ConfMatrix():

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.state = np.zeros((self.num_classes, self.num_classes))

    def calc(self, gt, pred):
        """ calcs and returns the CM without saveing it to state """
        return confusion_matrix(gt.flatten(),
                                pred.flatten(),
                                labels=np.arange(self.num_classes))

    def get_existing_classes(self):
        return sum(np.sum(self.state, axis=1) > 0)

    def add(self, gt, pred):
        """ adds one label mask to the confusion matrix """

        assert gt.shape == pred.shape
        assert gt.shape == (256, 256)

        gt = gt.flatten()
        pred = pred.flatten()
        pred = pred[gt != 255]
        gt = gt[gt != 255]

        if not gt.size == 0:
            self.state += confusion_matrix(gt, pred,
                                           labels=np.arange(self.num_classes))

        return None

    def add_batch(self, gt, pred):
        """ adds a batch of label masks to the confusion matrix """

        # convert pytorch tensors to numpy arrays
        if not isinstance(gt, np.ndarray):
            gt = gt.cpu().numpy()
            pred = pred.cpu().numpy()

        assert len(gt.shape) == 3       # assert C x W x H

        noc = gt.shape[0]               # number of channels
        for batchindex in range(noc):   # iterate over batch
            self.add(gt[batchindex], pred[batchindex])

        return None

    def norm_on_lines(self):
        """ norms along the lines of the matrix """

        a = self.state
        b = np.sum(self.state, axis=1)[:, None]
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def get_aa(self):
        confmatrix = self.norm_on_lines()
        return np.diagonal(confmatrix).sum() / self.get_existing_classes()

    def get_IoU(self):
        res = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            cm = self.state
            a = cm[i, i]
            b = (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            res[i] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return res

    def get_mIoU(self):
        return np.mean(self.get_IoU())


def AA(gt, pred, num_classes):
    """ This is the mean over the diagonal of the confusion
    matrix when it's normed """

    cm = ConfMatrix(num_classes)
    cm.add(gt, pred)
    confmatrix = cm.norm_on_lines()

    return np.mean(np.diagonal(confmatrix))


def IoU(gt, pred, num_classes):
    """
    the intersection over union for class i can be calculated as follows:


    get the intersection:
        >>> thats the element [i,i] of the confusion matrix (cm)

    the union:
        >>> is the sum over row with index i plus the sum over line with index
        i minux the diagonal element [i,i] (otherwise its counted twice)

    """

    cm = ConfMatrix(num_classes).calc(gt, pred)

    res = np.zeros(num_classes)
    for i in range(num_classes):
        res[i] = cm[i, i] / (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])

    return res


def mIoU(gt, pred, num_classes):
    return np.mean(IoU(gt, pred, num_classes))

class PixelwiseMetrics(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

        # Initialize dictionaries to store class-wise statistics
        self.data = {"pixelclass_" + str(i): {"correct": 0, "total": 0} for i in range(num_classes)}

    def add_batch(self, y, y_hat):
        for c in range(self.num_classes):
            class_data = self.data["pixelclass_" + str(c)]
            preds_c = y_hat == c
            targs_c = y == c
            num_correct = (preds_c * targs_c).sum().cpu().detach().numpy()
            num_pixels = np.sum(targs_c.cpu().detach().numpy())

            # Update class-wise statistics
            class_data["correct"] += num_correct
            class_data["total"] += num_pixels

    def get_classwise_accuracy(self):
        cw_acc_ = {k: el['correct'] / el['total'] if el['total'] > 0 else 0.0 for k, el in self.data.items()}
        return cw_acc_

    def get_average_accuracy(self):
        cw_acc = self.get_classwise_accuracy()
        return np.mean(list(cw_acc.values()))
    
import re
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix


def generate_miou(path_truth: str, path_pred: str) -> list:

    #################################################################################################
    def get_data_paths (path, filter):
        for path in Path(path).rglob(filter):
             yield path.resolve().as_posix()

    def calc_miou(cm_array):
        m = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
        m = np.nansum(ious[:-1]) / (np.logical_not(np.isnan(ious[:-1]))).sum()
        return m.astype(float), ious[:-1]

    #################################################################################################

    truth_images = sorted(list(get_data_paths(Path(path_truth), 'mask*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
    preds_images  = sorted(list(get_data_paths(Path(path_pred), 'image*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
    if len(truth_images) != len(preds_images):
        print('[WARNING !] mismatch number of predictions and test files.')

    patch_confusion_matrices = []

    for u in range(len(truth_images)):
        target = np.array(Image.open(truth_images[u]))-1 # -1 as model predictions start at 0 and turth at 1.
        target[target>12]=12  ### remapping masks to reduced baseline nomenclature.
        preds = np.array(Image.open(preds_images[u]))
        patch_confusion_matrices.append(confusion_matrix(target.flatten(), preds.flatten(), labels=list(range(13))))

    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    mIou, ious = calc_miou(sum_confmat)

    return mIou, ious

def print_metrics(miou, ious):
    classes = ["Forest", "Shrubland", "Grassland", "Wetlands", "Croplands", "Urban/Built-up",
               "Barren", "Water","Invalid"]
    print('\n')
    print('-'*40)
    print(' '*8, 'Model mIoU : ', round(miou, 4))
    print('-'*40)
    print ("{:<25} {:<15}".format('Class','iou'))
    print('-'*40)
    for k, v in zip(classes, ious):
        print ("{:<25} {:<15}".format(k, v))
    print('\n\n')


def print_metrics_Flair1(miou, ious):
    classes = ['building', 'pervious surface','impervious surface',
               'bare soil','water','coniferous','deciduous','brushwood','vineyard',
               'herbaceous vegetation','agricultural land','plowed land','other']
    print('\n')
    print('-'*40)
    print(' '*8, 'Model mIoU : ', round(miou, 4))
    print('-'*40)
    print ("{:<25} {:<15}".format('Class','iou'))
    print('-'*40)
    for k, v in zip(classes, ious):
        print ("{:<25} {:<15}".format(k, v))
    print('\n\n')


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def predict_and_save(test_data_path, save_path, model, device='cuda'):
    """
    Function to perform predictions on a test dataset and save the results.

    Args:
    test_data_path (str): Path to the test dataset.
    save_path (str): Path where to save the prediction results.
    model (torch.nn.Module): Pre-trained model for making predictions.
    device (str): Device to run the predictions on ('cuda' or 'cpu').

    Returns:
    None
    """

    # Set the model to evaluation mode and to the specified device
    model.eval()
    model.to(device)

    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Check if the save path exists, if not create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Perform predictions and save them
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Convert predictions to a suitable format (e.g., numpy array)
            predicted = outputs.cpu().numpy()

            # Save the results
            save_file_path = os.path.join(save_path, f'prediction_{i}.npy')
            np.save(save_file_path, predicted)

    print("Predictions successfully saved.")

