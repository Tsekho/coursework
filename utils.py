import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler


"""
Output formatting
"""
TIME_NOW = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
ARGS_FORMAT = \
    "training {net} with {ws} processes on {p} fraction of CIFAR-10\n" \
    "  cuda on parameter server    {cuda_s}\n" \
    "  cuda on workers             {cuda_c}\n\n" \
    "{e} epochs ({lr} learning rate, {cr} compression rate):\n" \
    "  {w} warming up epochs with {wcr} starting compression rate\n" \
    "  {ms} milestones with {g} learning rate reduction\n\n" \
    "batch size                    {b}\n" \
    "weight decay                  {wd}\n" \
    "momentum                      {m}\n" \
    "  nesterov                    {n}\n" \
    "gradient clipping max norm    {cn}\n\n" \
    "using tensorboard for logging {tb}\n" \
    "using csv for logging         {csv}\n" \
    "silent                        {s}\n" \
    "using checkpoints every       {c} epochs\n\n\n"
STDOUT_HEADER = \
    "{:15s} |    {:4d} WORKERS | {:20s} | "
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
    "     stp,     trn_a,     trn_l, time_send, time_recv, " \
    "time_calc, time_eval,     grd_n,        cr\n"
CSV_TRAIN_FORMAT = \
    "{stp:8d},{trn_a:9.8f},{trn_l:10.4f},  {time_send:8.5f},  {time_recv:8.5f}," \
    "  {time_calc:8.5f},  {time_eval:8.5f},{grd_n:10.4f},{cr:9.8f}\n"
CSV_TEST_HEADER = \
    "      epc,    tst_a,     tst_l,time_epch,time_btch,time_send,time_recv,time_ugrd," \
    "time_ubnl,time_eval,time_logw,time_chck,        lr\n"
CSV_TEST_FORMAT = \
    "{epc:8d},{tst_a:9.8f},{tst_l:10.4f}, {time_epch:8.5f}, {time_btch:8.5f}," \
    " {time_send:8.5f}, {time_recv:8.5f}, {time_ugrd:8.5f}," \
    " {time_ubnl:8.5f}, {time_eval:8.5f}, {time_logw:8.5f}, {time_chck:8.5f},{lr:9.8f}\n"


def fpath(*path):
    """
    Pathing respective to train.py
    """
    return os.path.join(os.path.dirname(__file__), *path)


def fmdir(*path):
    """
    Directory creation respective to train.py
    """
    p = fpath(*path)
    if not os.path.isdir(p):
        os.makedirs(p)


def enum(*sequential, **named):
    """
    Easier enumeration
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


"""
Message tags
"""
TAGS = enum("START", "PARAMS", "GRADS", "STATS", "FINISH")


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
    sampler = DistributedSampler(subset, args.ws - 1, args.r - 1)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.b,
                                         shuffle=False, num_workers=0,
                                         sampler=sampler)
    return loader


def get_train_light_dataloader(args):
    """
    Uses 0.05 of training dataset to learn BN parameters locally
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
                                    len(dataset) // 20,
                                    replacement=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.b,
                                         shuffle=False, num_workers=0,
                                         sampler=sampler)
    return loader


def get_test_dataloader(args):
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                           (0.2023, 0.1994, 0.2010))])
    dataset = torchvision.datasets.CIFAR10(root=fpath("data"), train=False, download=True,
                                           transform=transformer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=False, num_workers=0)
    return loader


def get_network(args):
    if args.net == "resnet18":
        from models import ResNet18
        net = ResNet18()
    if args.net == "resnet34":
        from models import ResNet34
        net = ResNet34()
    elif args.net == "resnet50":
        from models import ResNet50
        net = ResNet50()
    elif args.net == "simplenetv1":
        from models import SimpleNetV1
        net = SimpleNetV1()
    if args.cuda:
        net = net.cuda()
    return net


def compress(params):
    """
    Passing sparse gradients as 1d arrays with lighter indexation
    """
    compressed = []
    for p in params:
        t = p.view(-1)
        i = t.nonzero().view(1, -1)
        compressed.append((i.cpu(), t[i].view(-1).cpu(), t.size(), p.dtype))
    return compressed


def decompress(message):
    """
    Building sparse tensors back
    """
    for i, v, s, dt in message:
        t = torch.sparse_coo_tensor(i, v, size=s, dtype=dt)
        yield t
