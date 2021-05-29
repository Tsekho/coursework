import argparse

from mpi4py import MPI

import torch

torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument("-net", "--network", default="simplenetv1", type=str,
                        help="network, one of (simplenetv1), resnet18, resnet34, resnet50",
                        dest="net")
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="total epochs (50)",
                        dest="e")
    parser.add_argument("-w", "--warmup_epochs", type=int, default=5,
                        help="warming epochs (5) [-1 for None]",
                        dest="w")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="batch size (64)",
                        dest="b")

    # usage of cuda acceleration
    parser.add_argument("-cuda_s", "--cuda_server", action="store_true", default=False,
                        help="use cuda on server",
                        dest="cuda_s")
    parser.add_argument("-cuda_c", "--cuda_clients", action="store_true", default=False,
                        help="use cuda on clients",
                        dest="cuda_c")

    # (!) applied according to global epoch
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05,
                        help="base learning rate (0.05)",
                        dest="lr")
    parser.add_argument("-ms", "--milestones", nargs="*", type=int, default=[30, 40],
                        help="milestones (30, 40) [-1 for None]",
                        dest="ms")
    parser.add_argument("-g", "--gamma", type=float, default=0.1,
                        help="gamma (0.1) [-1 for None]",
                        dest="g")

    # (!) applied according to local epoch as it's applied locally
    parser.add_argument("-cr", "--compression_rate", type=float, default=0.01,
                        help="compression rate (0.01) [-1 for None]",
                        dest="cr")
    parser.add_argument("-wcr", "--warmup_compression_rate", type=float, default=0.5,
                        help="warming compression rate (0.5) [-1 for None]",
                        dest="wcr")

    # other local parameters
    parser.add_argument("-cn", "--clip_gradient_norm", type=float, default=1,
                        help="gradient max norm (1) [-1 for None]",
                        dest="cn")
    parser.add_argument("-m", "--momentum", type=float, default=0.7,
                        help="momentum (0.7) [-1 for None]",
                        dest="m")
    parser.add_argument("-n", "--nesterov", action="store_true", default=False,
                        help="use nesterov momentum",
                        dest="n")
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4,
                        help="weight decay (5e-4) [-1 for None]",
                        dest="wd")

    # logging and checkpoints creation
    parser.add_argument("-s", "--silent", action="store_true", default=False,
                        help="silent mode",
                        dest="s")
    parser.add_argument("-tb", "--tensorboard", action="store_true", default=False,
                        help="use tensorboard",
                        dest="tb")
    parser.add_argument("-csv", "--csv", action="store_true", default=False,
                        help="save to csv",
                        dest="csv")
    parser.add_argument("-c", "--checkpoints", default=-1, type=int,
                        help="checkpoints interval (None)",
                        dest="c")

    # handy for fast testing
    parser.add_argument("-p", "--partial", default=1.0, type=float,
                        help="dataset fraction (1) [-1 for None]",
                        dest="p")

    args = parser.parse_args()

    args.ws = MPI.COMM_WORLD.Get_size()
    args.r = MPI.COMM_WORLD.Get_rank()

    assert args.ws > 1

    args.cr = None if args.cr <= 0 else args.cr
    args.cn = None if args.cn <= 0 else args.cn
    args.m = None if args.m <= 0 else args.m
    args.wd = None if args.wd < 0 else args.wd

    args.c = None if args.c <= 0 else args.c
    args.p = None if args.p <= 0 else args.p

    if (args.w <= 0) or (args.wcr <= 0):
        args.w = None
        args.wcr = None

    if (args.g <= 0) or not args.ms:
        args.g = None
        args.ms = None

    # start jobs
    if args.r != 0:
        args.cuda = args.cuda_c
        from client import Client
        entity = Client(args)
    else:
        args.cuda = args.cuda_s
        from server import Server
        entity = Server(args)
    entity.run()


if __name__ == "__main__":
    main()
