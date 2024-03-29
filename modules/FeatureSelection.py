import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FeatureSelectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1. / (1. + torch.exp(-10. * x))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()

        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            tl = tl if tl > (in_size / 3) else in_size / 3
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size - tl else in_size - tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size - tl else in_size - tl

            w_off = int(tx - tl) if (tx - tl) > 0 else 0
            h_off = int(ty - tl) if (ty - tl) > 0 else 0
            w_end = int(tx + tl) if (tx + tl) < in_size else in_size
            h_end = int(ty + tl) if (ty + tl) < in_size else in_size

            mk = (h(x - w_off) - h(x - w_end)) * (h(y - h_off) - h(y - h_end))
            xatt = images[i] * mk
            before_upsample = Variable(xatt.unsqueeze(0))
            xamp = F.interpolate(before_upsample, size=(224, 224), mode='bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_tensors[0], self.saved_tensors[1]
        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size / 3 * 2)
        short_size = (in_size / 3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size) + (x >= long_size) + (y < short_size) + (y >= long_size)) > 0).float() * 2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret


class FeatureSelectionLayer(nn.Module):
    def forward(self, images, locs):
        return FeatureSelectionFunction.apply(images, locs)
