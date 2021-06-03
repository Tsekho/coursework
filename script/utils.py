import sys
import time
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler

import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator

import pickle
from io import BytesIO

#####
# OUTPUTS
#####


"""
Output formatting
"""
TIME_NOW = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
ARGS_FORMAT = \
    "training net={net} on f={p} fraction of train set\n" \
    "  cuda on parameter server  cuda:{cuda}\n\n" \
    "e={e} epochs:\n" \
    "  steps per spoch            spe={spe}\n" \
    "  params pull every       sppull={sppull} epochs\n" \
    "  logs push every            spr={spr} steps\n" \
    "  using checkpoints every      c={c} epochs\n" \
    "  milestones                  ms={ms}\n" \
    "    learning rate             lr={lr}\n" \
    "    learning rate reduction    g={g}\n" \
    "    staleness compensation     a={a}\n" \
    "  fraction of train set \\\n" \
    "  to train BN layers         bnf={bnp}\n\n" \
    "batch size                     b={b}\n" \
    "weight decay                  wd={wd}\n" \
    "gradient clipping max norm    cn={cn}\n" \
    "momentum                       m={m}\n" \
    "  nesterov                     n:{n}\n" \
    "compression rate              cr={cr}\n\n" \
    "using tensorboard for logging tb:{tb}\n" \
    "using csv for logging        csv:{csv}\n" \
    "silent                         s:{s}\n\n\n"
STDOUT_CLIENT = \
    "{H:3s}    {V:8s} | STEP   {stp:8d} | ACCURACY      {trn_a:5.4f} |" \
    " RECV TIME  {time_recv:8.5f}s |\n" \
    "                | NORM {grd_n:10.4f} | LOSS      {trn_l:10.4f} |"\
    " SEND TIME  {time_send:8.5f}s |"
STDOUT_HEADER = \
    "{:15s} |                 | {:20s} | "
STDOUT_FORMAT = \
    "EPOCH      {epc:4d} | STEP   {stp:8d} | ACCURACY      {tst_a:5.4f} |" \
    " RECV TIME  {time_recv:8.5f}s |\n" \
    "    {time_epch:10.2f}s |     {time_btch:8.5f}s/S | LOSS      {tst_l:10.4f} |"\
    " SEND TIME  {time_send:8.5f}s |\n" \
    "                |                 | LR       {lr:10.9f} | "
STDOUT_FINAL = \
    "!!!!!!!!!!!!!!!!!!!! |\n" \
    "FINISHED        | EPOCH      {:4d} | BEST ACCURACY {:5.4f} |                      |\n" \
    "    {:10.2f}s |   {:10.2f}s/E |                {:4d}E |                      |\n"
CSV_TRAIN_HEADER = \
    " param_v,     stp,     trn_a,     trn_l, time_send, time_recv, " \
    "time_calc, time_eval,     grd_n\n"
CSV_TRAIN_FORMAT = \
    "{prmv:8d},{stp:8d},{trn_a:9.8f},{trn_l:10.4f},  {time_send:8.5f},  {time_recv:8.5f}," \
    "  {time_calc:8.5f},  {time_eval:8.5f},{grd_n:10.4f}\n"
CSV_TEST_HEADER = \
    "     epc,     tst_a,     tst_l, time_epch,time_btch,time_send,time_recv,time_ugrd," \
    "time_ubnl,time_eval,time_logw,time_chck,        lr\n"
CSV_TEST_FORMAT = \
    "{epc:8d},{tst_a:9.8f},{tst_l:10.4f},{time_epch:10.2f}, {time_btch:8.5f}," \
    " {time_send:8.5f}, {time_recv:8.5f}, {time_ugrd:8.5f}," \
    " {time_ubnl:8.5f}, {time_eval:8.5f}, {time_logw:8.5f}, {time_chck:8.5f},{lr:9.8f}\n"


def fpath(*path):
    """
    Pathing respective to train.py
    """
    fcp = os.path.dirname(__file__)
    fup = os.path.dirname(fcp)
    return os.path.join(fup, *path)


def fmdir(*path):
    """
    Directory creation respective to train.py
    """
    p = fpath(*path)
    if not os.path.isdir(p):
        os.makedirs(p)
    return p

#####
# COMMUNICATION
#####


def enum(*sequential, **named):
    """
    Easier enumeration
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type("Enum", (), enums)


"""
Message tags
"""
# REQ ->
# ...<- REP
TAGS = enum("INIT", "ARGS",
            "PULL", "PARAMS",
            "GRADS", "CONFIRM_GRADS",
            "STATS", "CONFIRM_STATS")


def memoryleak_fix(map_loc):
    return lambda b: torch.load(BytesIO(b), map_location=map_loc)


class MappedUnpickler(pickle.Unpickler):
    def __init__(self, *args, map_loc="cpu", **kwargs):
        self._map_location = map_loc
        self.allowed = {("argparse", "Namespace"),
                        ("torch._utils", "_rebuild_parameter"),
                        ("torch._utils", "_rebuild_tensor_v2"),
                        ("torch.storage", "_load_from_bytes"),
                        ("collections", "OrderedDict"),
                        ("torch", "Size"),
                        ("torch", "float32"),
                        ("torch", "int32")}
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise pickle.UnpicklingError(
                "Restricted pickle. {} {}".format(module, name))
        if module == "torch.storage" and name == "_load_from_bytes":
            return memoryleak_fix(self._map_location)
        return getattr(sys.modules[module], name)


class Communicator:
    """
    ZMQ communication
    """

    def __init__(self, wd, args):
        context = zmq.Context()
        self.silent = args.s
        self.dev = "cuda" if args.cuda else "cpu"
        if args.srv:
            auth = ThreadAuthenticator(context)
            auth.start()

            pubksd = os.path.join(wd, "public")
            auth.configure_curve(domain="*", location=pubksd)

            seck = os.path.join(wd, "private", "server.key_secret")
            pk, sk = zmq.auth.load_certificate(seck)
            self.sock = context.socket(zmq.REP)
            self.sock.curve_publickey = pk
            self.sock.curve_secretkey = sk
            self.sock.setsockopt(zmq.RCVTIMEO, 60 * 60 * 1000)  # 1h
            self.sock.curve_server = True

            self.sock.bind("tcp://*:{}".format(args.port))
        else:
            seck = os.path.join(wd, "private", "client.key_secret")
            pk, sk = zmq.auth.load_certificate(seck)
            self.sock = context.socket(zmq.REQ)
            self.sock.curve_publickey = pk
            self.sock.curve_secretkey = sk
            self.sock.setsockopt(zmq.RCVTIMEO, 300 * 1000)  # 5m

            pubkd = os.path.join(wd, "public", "server.key")
            mk, _ = zmq.auth.load_certificate(pubkd)
            self.sock.curve_serverkey = mk

            self.sock.connect("tcp://{}:{}".format(args.ip, args.port))
        self.sock.setsockopt(zmq.SNDTIMEO, 300 * 1000)
        self.sock.setsockopt(zmq.LINGER, 0)

    def send(self, obj, tag):
        p = pickle.dumps([tag, obj], 4)
        self.sock.send(p)

    def recv(self):
        try:
            p = self.sock.recv()
        except zmq.Again as err:
            if not self.silent:
                print("Failed to receive, timeout.")
                sys.stdio.flush()
            exit(1)
        up = MappedUnpickler(BytesIO(p), map_loc=self.dev).load()
        return up


#####
# GRADIENT SPARSIFICATION
#####


def compress(params):
    """
    Passing sparse gradients as 1d arrays with lighter indexation
    """
    compressed = []
    for p in params:
        t = p.view(-1)
        i = t.nonzero().view(1, -1)
        compressed.append((i.to(torch.int32),
                           t[i].view(-1), t.size(), t.dtype))
    return compressed


def decompress(message):
    """
    Building sparse tensors back
    """
    for i, v, s, dt in message:
        t = torch.sparse_coo_tensor(i, v, size=s, dtype=dt)
        yield t

#####
# DATALOADERS
#####


def get_train_dataloader(args):
    transformer = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010))])
    dataset = torchvision.datasets.CIFAR10(root=fpath("data"), train=True, download=True,
                                           transform=transformer)
    if args.p != 1.0:
        count = max(1, int(len(dataset) * args.p))
        g = torch.Generator()
        g.manual_seed(0)
        indexes = torch.randperm(len(dataset), generator=g)[:count]
    else:
        indexes = list(range(len(dataset)))
    subset = torch.utils.data.Subset(dataset, indexes)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.b,
                                         shuffle=True, num_workers=2)
    return loader


def get_train_light_dataloader(args):
    """
    Uses fraction of training dataset to learn BN parameters locally
    without reducing them from workers
    """
    transformer = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010))])
    dataset = torchvision.datasets.CIFAR10(root=fpath("data"), train=True, download=False,
                                           transform=transformer)
    sampler = WeightedRandomSampler(torch.ones(len(dataset)),
                                    max(1, int(len(dataset) * args.bnp)),
                                    replacement=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.b,
                                         shuffle=False, num_workers=2,
                                         sampler=sampler)
    return loader


def get_test_dataloader(args):
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010))])
    dataset = torchvision.datasets.CIFAR10(root=fpath("data"), train=False, download=True,
                                           transform=transformer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=False, num_workers=2)
    return loader

#####
# MODEL
#####


def get_network(args):
    if args.net == "resnet18":
        from script.models import ResNet18
        net = ResNet18()
    if args.net == "resnet34":
        from script.models import ResNet34
        net = ResNet34()
    elif args.net == "resnet50":
        from script.models import ResNet50
        net = ResNet50()
    elif args.net == "simplenetv1":
        from script.models import SimpleNetV1
        net = SimpleNetV1()
    if args.cuda:
        net = net.cuda()
    return net
