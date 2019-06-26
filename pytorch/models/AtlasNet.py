import _init_paths
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from common import PointNetfeat, weights_init
from dist_chamfer import chamferDist as chamfer

def AtlasNet_setup(args):
    args.odir = 'results/%s/AtlasNet_%s' % (args.dataset, args.dist_fun)
    grain = int(np.sqrt(args.npts/args.nb_primitives))
    grain = grain*1.0
    n = ((grain + 1)*(grain + 1)*args.nb_primitives)
    if n < args.npts:
        grain += 1
    args.npts = (grain + 1)*(grain + 1)*args.nb_primitives
    args.odir += '_npts%d' %  (args.npts)
    args.odir += '_NBP%d' % (args.nb_primitives)
    args.odir += '_lr%.4f' % (args.lr)
    args.odir += '_' + args.optim
    args.odir += '_B%d' % (args.batch_size)
    args.classmap =''

    #generate regular grid
    vertices = []
    for i in range(0,int(grain + 1 )):
            for j in range(0,int(grain + 1 )):
                vertices.append([i/grain,j/grain])

    grid = [vertices for i in range(0,args.nb_primitives)]
    print("grain", grain, 'number vertices', len(vertices)*args.nb_primitives)
    args.grid = grid

def AtlasNet_create_model(args):
    """ Creates model """
    model = nn.DataParallel(AtlasNet(args, num_points=args.npts, nb_primitives=args.nb_primitives))
    args.enc_params = sum([p.numel() for p in model.module.encoder.parameters()])
    args.dec_params = sum([p.numel() for p in model.module.decoder.parameters()])
    args.nparams = sum([p.numel() for p in model.module.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    print(model)
    model.cuda()
    model.apply(weights_init)
    return model

def AtlasNet_step(args, targets_in, clouds_data):
    targets = Variable(torch.from_numpy(targets_in), requires_grad=False).float().cuda()
    targets = targets.transpose(2, 1).contiguous()
    inp = Variable(torch.from_numpy(clouds_data[1]), requires_grad=False).float().cuda()
    outputs =  args.model.forward(inp, args.grid)
    targets = targets.transpose(2, 1).contiguous()
    N = targets.size()[1]
    dist1, dist2 = eval(args.dist_fun)()(outputs, targets)
    # EMD not working in pytorch (see pytorch-setup.md)
    #emd_cost = args.emd_mod(outputs[:, 0:N,:], targets)/N
    #emd_cost = emd_cost.data.cpu().numpy()
    emd_cost = 0#args.emd_mod(outputs[:, 0:N, :], targets)/N
    emd_cost = np.array([0]*args.batch_size)#emd_cost.data.cpu().numpy()

    loss = torch.mean(dist2) + torch.mean(dist1)
    dist1 = dist1.data.cpu().numpy()
    dist2 = dist2.data.cpu().numpy()

    if args.model.training:
        return loss, dist1, dist2, emd_cost, outputs.data.cpu().numpy()
    else:
        return loss.item(), dist1, dist2, emd_cost, outputs.data.cpu().numpy()

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class AtlasNet(nn.Module):
    def __init__(self, args, num_points=2048, bottleneck_size=1024, nb_primitives=1):
        super(AtlasNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(args, num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0,self.nb_primitives)])

    def forward(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
