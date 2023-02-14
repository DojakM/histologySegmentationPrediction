import cv2
import torch
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from alive_progress import alive_bar

def label2rgb(alpha, img, mask):
    labeled_img = cv2.addWeighted(decode_segmap(mask).transpose(1, 2, 0).astype(int), alpha, img.astype(int), 1 - alpha,
                                  0)
    return torch.from_numpy(labeled_img.transpose(2, 0, 1))


def decode_segmap(image: np.array, num_classes: int = 5):
    label_colors = np.array([(255, 255, 255), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for cls_idx in range(0, num_classes):
        idx = image == cls_idx
        r[idx] = label_colors[cls_idx, 0]
        g[idx] = label_colors[cls_idx, 1]
        b[idx] = label_colors[cls_idx, 2]
    rgb = np.stack([r, g, b], axis=0)
    return rgb


def unnormalize(img, mean=0.6993, std=0.4158):
    img = img * std
    img = img + mean
    return img * 255.0


def set_dropout(model, drop_rate=0.5):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout2d):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def standard_prediction(model, X):
    model = model.eval()
    logits = model(Variable(X))[0]
    pred = torch.argmax(logits.squeeze(), dim=0).cpu().detach().float().unsqueeze(0)

    return pred


def predict_dist(model, X, T=100):
    model = model.train()

    softmax_out_stack = []

    with alive_bar(T, title=f' MC-Dropout:') as bar:
        for mc_i in range(T):
            logits = model(Variable(X))[0]
            softmax_out = F.softmax(logits, dim=1)

            del logits
            torch.cuda.empty_cache()

            # remove batch dim
            softmax_out = softmax_out.squeeze(0)

            softmax_out_stack.append(softmax_out)

            bar.text('[MC-it: ' + str(mc_i + 1) + ']')
            bar()

    softmax_out_stack = torch.stack(softmax_out_stack)

    return softmax_out_stack


def monte_carlo_dropout_proc(model, x, T=1000, dropout_rate=0.5):
    standard_pred = standard_prediction(model, x)

    set_dropout(model, drop_rate=dropout_rate)

    softmax_dist = predict_dist(model, x, T)

    pred_std = torch.std(softmax_dist, dim=0)

    del softmax_dist
    torch.cuda.empty_cache()

    pred_std = pred_std.gather(0, standard_pred.long()).squeeze(0)

    model = model.eval()

    return pred_std
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


