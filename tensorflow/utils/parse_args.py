import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='3D Object Point Cloud Completion')

    # Optimization arguments
    parser.add_argument('--optim', default='adagrad', help='Optimizer: sgd|adam|adagrad|adadelta|rmsprop')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=0.5e-2, type=float, help='Initial learning rate')

    #Training/Testing arguments
    parser.add_argument('--dist_fun', default='chamfer', help='Point Cloud Distance used in training')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=300, type=int, help='Number of training epochs')
    parser.add_argument('--resume', type=int, default=0, help='If 1, resume training')
    parser.add_argument('--eval', default=0, type=int, help='If 1, evaluate using best model')
    parser.add_argument('--benchmark', default=0, type=int, help='If 1 get output results in benchmark format')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--nworkers', default=4, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')


    # Model
    parser.add_argument('--NET', default='TopNet', help='Network used')
    parser.add_argument('--code_nfts', default=1024, type=int, help='Encoder output feature size')
    parser.add_argument('--ENCODER_ID', default=1, type=int, help='TopNet only - If 0, use PointNet encoder, if 1 use PCN encoder')
    parser.add_argument('--NLEVELS', default=6, type=int, help="Number of tree levels in TopNet")
    parser.add_argument('--NFEAT', default=8, type=int, help="Node feature dimension in TopNet")

    # Dataset
    parser.add_argument('--dataset', default='shapenet', help='Dataset name: shapenet')
    parser.add_argument('--pc_augm_scale', default=0, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=0, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=0, type=int, help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--npts', default=2048, type=int, help='Number of output points generated')
    parser.add_argument('--inpts', default=2048, type=int, help='Number of input points')
    parser.add_argument('--ngtpts', default=2048, type=int, help='Number of ground-truth points')

    args = parser.parse_args()
    args.start_epoch = 0
    if args.dataset == 'shapenet16384':
        args.ngtpts = 16384

    return args
