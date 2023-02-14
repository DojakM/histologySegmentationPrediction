import abc
from argparse import ArgumentParser
from typing import Any, Optional

import cv2
import pytorch_lightning as pl
import torch

from utils import unnormalize, label2rgb
from losses.FocalLosses import FocalLoss


class UnetSuper(pl.LightningModule):
    def __init__(self, len_test_set: int, hparams, **kwargs):
        super(UnetSuper, self).__init__()
        self.num_classes = kwargs["num_classes"]
        self.metric = iou_fnc
        self.save_hyperparameters(hparams)
        self.args = kwargs
        self.len_test_set = len_test_set
        self.weights = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.criterion = FocalLoss(apply_nonlin=None, alpha=self.weights, gamma=2)
        self._to_console = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers', type=int, default=10, metavar='N', help='number of workers (default: 2)')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
        parser.add_argument('--gamma-factor', type=float, default=0.02)
        parser.add_argument('--training-batch-size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--training-epochs', type=int, default=15, help='training epochs')
        parser.add_argument('--test-batch-size', type=int, default=8, help='Input batch size for testing')
        parser.add_argument('--test-percent', type=float, default=0.15, help='dataset percent for testing')
        parser.add_argument('--test-epochs', type=int, default=10, help='epochs before testing')

        return parser

    @abc.abstractmethod
    def forward(self, x):
        pass

    def loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        labels = labels.long()
        return self.criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        output = {}

        x, y = train_batch
        prob_mask = self.forward(x)
        loss = self.criterion(prob_mask, y.type(torch.long))

        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['num_classes'])
        for i in range(self.args['num_classes']):
            output['iou_' + str(i)] = torch.tensor(iter_iou[i])
            output['iou_cnt_' + str(i)] = torch.tensor(iter_count[i])

        output['loss'] = loss

        return output

    def training_epoch_end(self, training_step_outputs):
        """
        On each training epoch end, log the average training loss
        """
        train_avg_loss = torch.stack([train_output['loss'] for train_output in training_step_outputs]).mean().item()

        train_iou_sum = torch.zeros(self.args['num_classes'])
        train_iou_cnt_sum = torch.zeros(self.args['num_classes'])
        for i in range(self.args['num_classes']):
            train_iou_sum[i] = torch.stack(
                [train_output['iou_' + str(i)] for train_output in training_step_outputs]).sum()
            train_iou_cnt_sum[i] = torch.stack(
                [train_output['iou_cnt_' + str(i)] for train_output in training_step_outputs]).sum()
        iou_scores = train_iou_sum / (train_iou_cnt_sum + 1e-10)

        iou_mean = iou_scores[~torch.isnan(iou_scores)].mean().item()

        self.log('train_avg_loss', train_avg_loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_mean_iou', iou_mean, sync_dist=True, on_step=False, on_epoch=True)
        for c in range(self.args['num_classes']):
            if train_iou_cnt_sum[c] == 0.0:
                iou_scores[c] = 0
            self.log('train_iou_' + str(c), iou_scores[c].item(), sync_dist=True, on_step=False, on_epoch=True)

        if self._to_console:
            print('epoch {0:.1f} - loss: {1:.15f} - meanIoU: {3:.15f}'.format(self.current_epoch, train_avg_loss,
                                                                              iou_mean))
            for c in range(self.args['num_classes']):
                print('class {} IoU: {}'.format(c, iou_scores[c].item()))

    def validation_step(self, test_batch, batch_idx):
        """
        Predicts on the test dataset to compute the current performance of the model.
        :param test_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - Validation performance
        """

        output = {}

        x, y = test_batch
        prob_mask = self.forward(x)
        loss = self.criterion(prob_mask, y.type(torch.long))

        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['num_classes'])
        for i in range(self.args['num_classes']):
            output['val_iou_' + str(i)] = torch.tensor(iter_iou[i])
            output['val_iou_cnt_' + str(i)] = torch.tensor(iter_count[i])

        output['val_loss'] = loss

        return output

    def validation_epoch_end(self, outputs):
        """
        Computes validation
        :param outputs: outputs after every epoch end
        :return: output - average validation loss
        """

        test_avg_loss = torch.stack([test_output['val_loss'] for test_output in outputs]).mean().item()

        test_iou_sum = torch.zeros(self.args['num_classes'])
        test_iou_cnt_sum = torch.zeros(self.args['num_classes'])
        for i in range(self.args['num_classes']):
            test_iou_sum[i] = torch.stack([test_output['val_iou_' + str(i)] for test_output in outputs]).sum()
            test_iou_cnt_sum[i] = torch.stack([test_output['val_iou_cnt_' + str(i)] for test_output in outputs]).sum()
        iou_scores = test_iou_sum / (test_iou_cnt_sum + 1e-10)

        iou_mean = iou_scores[~torch.isnan(iou_scores)].mean().item()

        self.log('val_avg_loss', test_avg_loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_mean_iou', iou_mean, sync_dist=True, on_step=False, on_epoch=True)
        for c in range(self.args['num_classes']):
            if test_iou_cnt_sum[c] == 0.0:
                iou_scores[c] = 0
            self.log('val_iou_' + str(c), iou_scores[c].item(), sync_dist=True, on_step=False, on_epoch=True)

        if self._to_console:
            print('eval ' + str(self.current_epoch) + ' ..................................................')
            print('eLoss: {0:.15f} - eMeanIoU: {2:.15f}'.format(test_avg_loss,
                                                                iou_mean))
            for c in range(self.args['num_classes']):
                print('class {} IoU: {}'.format(c, iou_scores[c].item()))

    def test_step(self, test_batch, batch_idx):
        """
        Predicts on the test dataset to compute the current accuracy of the models.

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """

        output = {}

        x, y = test_batch
        prob_mask = self.forward(x)
        loss = self.criterion(prob_mask, y.type(torch.long))

        iter_iou, iter_count = iou_fnc(torch.argmax(prob_mask, dim=1).float(), y, self.args['num_classes'])
        for i in range(self.args['num_classes']):
            output['test_iou_' + str(i)] = torch.tensor(iter_iou[i])
            output['test_iou_cnt_' + str(i)] = torch.tensor(iter_count[i])

        output['test_loss'] = loss

        return output

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score

        :param outputs: outputs after every epoch end

        :return: output - average test loss
        """
        test_avg_loss = torch.stack([test_output['test_loss'] for test_output in outputs]).mean().item()

        test_iou_sum = torch.zeros(self.args['num_classes'])
        test_iou_cnt_sum = torch.zeros(self.args['num_classes'])
        for i in range(self.args['num_classes']):
            test_iou_sum[i] = torch.stack([test_output['test_iou_' + str(i)] for test_output in outputs]).sum()
            test_iou_cnt_sum[i] = torch.stack([test_output['test_iou_cnt_' + str(i)] for test_output in outputs]).sum()
        iou_scores = test_iou_sum / (test_iou_cnt_sum + 1e-10)

        iou_mean = iou_scores[~torch.isnan(iou_scores)].mean().item()

        self.log('test_avg_loss', test_avg_loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_mean_iou', iou_mean, sync_dist=True, on_step=False, on_epoch=True)
        for c in range(self.args['num_classes']):
            if test_iou_cnt_sum[c] == 0.0:
                iou_scores[c] = 0
            self.log('test_iou_' + str(c), iou_scores[c].item(), sync_dist=True, on_step=False, on_epoch=True)

        if self._to_console:
            print('eval ' + str(self.current_epoch) + ' ..................................................')
            print('eLoss: {0:.15f} -  eMeanIoU: {2:.15f}'.format(test_avg_loss, iou_mean))
            for c in range(self.args['num_classes']):
                print('class {} IoU: {}'.format(c, iou_scores[c].item()))

    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """
        return {}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'])
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6, verbose=True,
            ),
            'monitor': 'train_avg_loss',
        }
        return [self.optimizer], [self.scheduler]

    def log_tb_images(self, batch_idx, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, index=0):
        img = cv2.cvtColor(unnormalize(x[index].cpu().detach().numpy().squeeze()), cv2.COLOR_GRAY2RGB).astype(int)
        pred = y_hat[index].cpu().detach().numpy()
        mask = y[index].cpu().detach().numpy()
        alpha = 0.7
        # Performing image overlay
        gt = label2rgb(alpha, img, mask)
        prediction = label2rgb(alpha, img, pred)
        log = torch.stack([gt, prediction], dim=0)
        self.logger.experiment.add_images(f'Images and Masks Batch: {batch_idx}', log, self.current_epoch)


def iou_fnc(pred, target, n_classes=7):
    import numpy as np
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    count = np.zeros(n_classes)

    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().cpu().item()
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().cpu().item() - intersection  # .data.cpu()[0] - intersection

        if union == 0:
            ious.append(0.0)
        else:
            count[cls] += 1
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious), count
