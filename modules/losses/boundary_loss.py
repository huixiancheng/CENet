import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    divce = label.device
    one_hot_label = torch.eye(
        n_classes, device='cuda', requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        # pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt


        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred


        # reshape
#         gt_b = gt_b[:, 1:, :, :]
#         pred_b = pred_b[:, 1:, :, :]
#         c = c-1

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


# for debug
# if __name__ == "__main__":
#     import torch.optim as optim
#     from torchvision.models import segmentation
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     img = torch.randn(1, 3, 224, 224).to(device)
#     gt = torch.randint(0, 10, (1, 224, 224)).to(device)
#     model = segmentation.fcn_resnet50(num_classes=10).to(device)
#
#     pic = cv2.imread(r"C:\Users\Aragaki Yui\Desktop\tu\02.jpg")
#     print(pic.shape)
#     cv2.imshow('a',pic)
#     cv2.waitKey(0)
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
#     criterion = BoundaryLoss()
#
#     y = model(img)
#
#     loss = criterion(y['out'], gt)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     print(loss)
