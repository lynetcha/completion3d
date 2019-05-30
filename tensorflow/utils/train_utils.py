import time
import numpy as np
import json
import logging
import random
from tqdm import tqdm
import os
from collections import defaultdict
import h5py
from multiprocessing import Queue
from data_process import kill_data_processes
from shapenet import ShapenetDataProcess
from PCN import *
from Folding import *
from TopNet import *

def check_overwrite(fname):
    if os.path.isfile(fname):
        valid = ['y', 'yes', 'no', 'n']
        inp = None
        while inp not in valid:
            inp = input(
                '%s already exists. Do you want to overwrite it? (y/n)'
                % fname)
            if inp.lower() in ['n', 'no']:
                raise Exception('Please create new experiment.')

def parse_experiment(odir):
    stats = json.loads(open(odir + '/trainlog.txt').read())
    valloss = [k['loss_val'] for k in stats if 'loss_val' in k.keys()]
    epochs = [k['epoch'] for k in stats if 'loss_val' in k.keys()]
    last_epoch = max(epochs)
    idx = np.argmin(valloss)
    best_val_loss = float('%.6f' % (valloss[idx]))
    best_epoch = epochs[idx]
    val_results = odir + '/results_val_%d' % (best_epoch)
    val_results = open(val_results).readlines()
    first_line = val_results[0]
    num_params = int(first_line.rstrip().split(' ')[-1])
    enc_params = int(val_results[1].rstrip().split(' ')[-1])
    dec_params = int(val_results[2].rstrip().split(' ')[-1])

    return last_epoch, best_epoch, best_val_loss, num_params, enc_params, dec_params

def model_at(args, i):
    return os.path.join(args.odir, 'models/model' + str(i) + '.ckpt')

def tf_resume(args, i):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))

    model = eval(args.NET + '_create_model')(args)
    args.saver.restore(args.sess, args.resume)
    args.start_epoch = i
    try:
        stats = json.loads(open(os.path.join(os.path.dirname(args.resume), 'trainlog.txt')).read())
    except:
        stats = []
    return model, stats

def set_optim(args):
    lr = tf.constant(args.lr, name='lr')
    if args.optim=='sgd':
        optimizer = tf.train.MomentumOptimizer(lr, args.momentum)
    elif args.optim=='adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif args.optim=='adagrad':
        optimizer = tf.train.AdagradOptimizer(lr, initial_accumulator_value=1e-13)
    elif args.optim=='adadelta':
        optimizer = tf.train.AdadeltaOptimizer(lr, rho=0.9, epsilon=1e-6)
    elif args.optim=='rmsprop':
        optimizer = tf.train.RMSPropOptimizer(alr, decay=0.99, epsilon=1e-8)
    return optimizer

def set_seed(seed):
    """ Sets seeds"""
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def data_setup(args, phase, num_workers, repeat):
    if args.dataset in ['shapenet', 'shapenet16384']:
        DataProcessClass = ShapenetDataProcess
    # Initialize data processes
    data_queue = Queue(4 * num_workers)
    data_processes =[]
    for i in range(num_workers):
        data_processes.append(DataProcessClass(data_queue, args, phase, repeat=repeat))
        data_processes[-1].start()
    return data_queue, data_processes

def train(args, epoch, data_queue, data_processes):
    """ Trains for one epoch """
    print("Training....")
    args.training = True

    N = len(data_processes[0].data_paths)
    Nb = int(N/args.batch_size)
    if Nb*args.batch_size < N:
        Nb += 1

    meters = []
    lnm = ['loss']
    Nl = len(lnm)
    for i in range(Nl):
        meters.append(AverageValueMeter())
    t0 = time.time()

    # iterate over dataset in batches
    for bidx in tqdm(range(Nb)):
        targets, clouds_data = data_queue.get()
        t_loader = 1000*(time.time()-t0)
        t0 = time.time()

        loss, dist1, dist2, emd_cost, outputs = args.step(args, targets, clouds_data)

        t_trainer = 1000*(time.time()-t0)
        losses = [loss, ]
        for ix, l in enumerate(losses):
            meters[ix].add(l)

        if (bidx % 50) == 0:
            prt = 'Train '
            for ix in range(Nl):
                prt += '%s %f, ' % (lnm[ix], losses[ix])
            prt += 'Loader %f ms, Train %f ms.\n' % (t_loader, t_trainer)
            print(prt)
        logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss, t_loader, t_trainer)
        t0 = time.time()
    return [meters[ix].value()[0] for ix in range(Nl)]

def test(split, args):
    """ Evaluated model on test set """
    print("Testing....")
    args.training = False

    data_queue, data_processes = data_setup(args, split, num_workers=1, repeat=False)

    meters = []
    lnm = ['loss',]
    Nl = len(lnm)
    for i in range(Nl):
        meters.append(AverageValueMeter())

    t0 = time.time()

    N = len(data_processes[0].data_paths)
    Nb = int(N/args.batch_size)
    if Nb*args.batch_size < N:
        Nb += 1
    # iterate over dataset in batches
    for bidx in tqdm(range(Nb)):
        targets, clouds_data = data_queue.get()
        t_loader = 1000*(time.time()-t0)
        t0 = time.time()

        loss, dist1, dist2, emd_cost, outputs = args.step(args, targets, clouds_data)

        t_trainer = 1000*(time.time()-t0)
        losses = [loss, ]
        for ix, l in enumerate(losses):
            meters[ix].add(l)

        logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss, t_loader, t_trainer)
        t0 = time.time()
    kill_data_processes(data_queue, data_processes)
    return [meters[ix].value()[0] for ix in range(Nl)]

def benchmark_results(split, args):
    data_queue, data_processes = data_setup(args, split, num_workers=1, repeat=False)
    L = len(data_processes[0].data_paths)
    Nb = int(L/args.batch_size)
    if Nb*args.batch_size < L:
        Nb += 1

    # iterate over dataset in batches
    for bidx in tqdm(range(Nb)):
        targets, clouds_data = data_queue.get()
        loss, dist1, dist2, emd_cost, outputs = args.step(args, targets, clouds_data)
        for idx in range(targets.shape[0]):
            fname = clouds_data[0][idx][:clouds_data[0][idx].rfind('.')]
            synset = fname.split('/')[-2]
            outp = outputs[idx:idx+1,...].squeeze()
            odir = args.odir + '/benchmark/%s' % (synset)
            if not os.path.isdir(odir):
                print("Creating %s ..." % (odir))
                os.makedirs(odir)
            ofile = os.path.join(odir, fname.split('/')[-1])
            print("Saving to %s ..." % (ofile))
            with h5py.File(ofile, "w") as f:
                f.create_dataset("data", data=outp)

    kill_data_processes(data_queue, data_processes)

def samples(split, args, N):
    print("Sampling ...")
    args.training=False

    collected = defaultdict(list)
    predictions = {}
    class_samples = defaultdict(int)
    if hasattr(args, 'classmap'):
        for val in args.classmap:
            class_samples[val[0]] = 0
    else:
        count = 0

    data_queue, data_processes = data_setup(args, split, num_workers=1, repeat=False)
    L = len(data_processes[0].data_paths)
    Nb = int(L/args.batch_size)
    if Nb*args.batch_size < L:
        Nb += 1

    # iterate over dataset in batches
    for bidx in tqdm(range(Nb)):
        targets, clouds_data = data_queue.get()
        run_net = False
        for idx in range(targets.shape[0]):
            if hasattr(args, 'classmap'):
                fname = clouds_data[0][idx][:clouds_data[0][idx].rfind('.')]
                synset = fname.split('/')[-2]
                if class_samples[synset] <= N:
                    run_net = True
                    break
            elif count <= N:
                run_net = True
                break
        if run_net:
            loss, dist1, dist2, emd_cost, outputs = args.step(args, targets, clouds_data)
            for idx in range(targets.shape[0]):
                if hasattr(args, 'classmap'):
                    fname = clouds_data[0][idx][:clouds_data[0][idx].rfind('.')]
                    synset = fname.split('/')[-2]
                    if class_samples[synset] > N:
                        continue
                    class_samples[synset] += 1
                else:
                    fname = str(bidx)
                    if count > N:
                        break
                    count +=1
                collected[fname].append((outputs[idx:idx+1,...], targets[idx:idx+1,...],
                                        clouds_data[1][idx:idx+1,...]))
    kill_data_processes(data_queue, data_processes)

    for fname, lst in collected.items():
        o_cpu, t_cpu, inp= list(zip(*lst))
        o_cpu = o_cpu[0]
        t_cpu, inp = t_cpu[0], inp[0]
        predictions[fname] = (inp, o_cpu, t_cpu)
    return predictions

def batch_instance_metrics(args, dist1, dist2):
    dgen = np.mean(dist1, 1) + np.mean(dist2, 1)
    return dgen

def metrics(split, args, epoch=0):
    print("Metrics ....")
    db_name = split
    args.training = False
    Gerror = defaultdict(list)
    Gerror_emd = defaultdict(list)
    data_queue, data_processes = data_setup(args, split, num_workers=1, repeat=False)
    N = len(data_processes[0].data_paths)
    Nb = int(N/args.batch_size)
    if Nb*args.batch_size < N:
        Nb += 1
    # iterate over dataset in batches
    for bidx in tqdm(range(Nb)):
        targets, clouds_data = data_queue.get()
        loss, dist1, dist2, emd_cost, outputs = args.step(args, targets, clouds_data)
        dgens = batch_instance_metrics(args, dist1, dist2)
        for idx in range(targets.shape[0]):
            if hasattr(args, 'classmap'):
                classname = args.classmap[clouds_data[0][idx].split('/')[0]]
            Gerror[classname].append(dgens[idx])
            Gerror_emd[classname].append(emd_cost[idx])
    kill_data_processes(data_queue, data_processes)
    Gm_errors = []
    Gm_errors_emd = []
    outfile = args.odir + '/results_%s_%d' % (db_name, epoch + 1)
    if args.eval:
        outfile = args.odir + '/eval_%s_%d' % (db_name, epoch)
    print("Saving results to %s ..." % (outfile))
    with open(outfile, 'w') as f:
        f.write('#ParametersTotal %d\n' % (args.nparams))
        f.write('#ParametersEncoder %d\n' % (args.enc_params))
        f.write('#ParametersDecoder %d\n' % (args.dec_params))
        for classname in list(Gerror.keys()):
            Gmean_error = np.mean(Gerror[classname])
            Gm_errors.append(Gmean_error)
            Gmean_error_emd = np.mean(Gerror_emd[classname])
            Gm_errors_emd.append(Gmean_error_emd)
            f.write('%s Generator_emd %.6f\n' % (classname, Gmean_error_emd))
            f.write('%s Generator_dist %.6f\n' % (classname, Gmean_error))
        f.write('Generator Class Mean EMD %.6f\n' % (np.mean(Gm_errors_emd)))
        f.write('Generator Class Mean DIST %.6f\n' % (np.mean(Gm_errors)))

def cache_pred(predictions, db_name, args):
    with h5py.File(os.path.join(args.odir, 'inp_'+ db_name +'.h5'), 'w') as hf:
        with h5py.File(os.path.join(args.odir, 'predictions_'+db_name+'.h5'), 'w') as hf1:
            with h5py.File(os.path.join(args.odir, 'gt_'+db_name+'.h5'), 'w') as hf2:
                for fname, o_cpu in predictions.items():
                    hf.create_dataset(name=fname, data=o_cpu[0])
                    hf1.create_dataset(name=fname, data=o_cpu[1])
                    hf2.create_dataset(name=fname, data=o_cpu[2])

def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

class AverageValueMeter():
    def __init__(self):
        self.avg = 0.0
        self.N = 0

    def add(self, k):
        self.avg = ((self.avg * self.N) + k)/(self.N + 1)
        self.N = self.N + 1

    def value(self):
        return self.avg, self.N


