# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib


class EMDFunction(Function):
    def forward(self, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        self.xyz1 = xyz1
        self.xyz2 = xyz2
        match = torch.zeros(batchsize, n , m)
        cost = torch.zeros(batchsize, )

        if not xyz1.is_cuda:
            my_lib.emd_forward(xyz1, xyz2, match, cost)
        else:
            match = match.cuda()
            cost = cost.cuda()
            temp = torch.zeros(batchsize, 2 * (m+n)).cuda()
            my_lib.emd_forward_cuda(xyz1, xyz2, match, cost, temp)


        self.match = match
        #print(batchsize, n, m)

        return cost

    def backward(self, gradcost):

        gradxyz1 = torch.zeros(self.xyz1.size())
        gradxyz2 = torch.zeros(self.xyz2.size())

        if not gradcost.is_cuda:
            my_lib.emd_backward(self.xyz1, self.xyz2, gradxyz1, gradxyz2, self.match)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            my_lib.emd_backward_cuda(self.xyz1, self.xyz2, gradxyz1, gradxyz2, self.match)

        return gradxyz1, gradxyz2
