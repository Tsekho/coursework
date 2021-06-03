from script.server_core import Server
import argparse
import torch

torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()

    # TODO: probably should load defaults from conf.py or passed file

    parser.add_argument("-p", "--port", default="51821", type=str,
                        help="master node port",
                        dest="port")
    parser.add_argument("-sppull", "--steps_per_pull", type=int, default=5,
                        help="number of steps per pull (5)",
                        dest="sppull")
    parser.add_argument("-spr", "--steps_per_report", type=int, default=50,
                        help="number of steps per stats push (50) [-1 for None]",
                        dest="spr")

    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="total epochs (50)",
                        dest="e")
    parser.add_argument("-spe", "--steps_per_epoch", type=int, default=1000,
                        help="number of steps per epoch (1000)",
                        dest="spe")
    parser.add_argument("-net", "--network", default="simplenetv1", type=str,
                        help="network, one of (simplenetv1), resnet18, resnet34, resnet50",
                        dest="net")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="batch size (64)",
                        dest="b")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05,
                        help="base learning rate (0.05)",
                        dest="lr")
    parser.add_argument("-ms", "--milestones", nargs="*", type=int, default=[30, 40],
                        help="milestones (30, 40) [-1 for None]",
                        dest="ms")
    parser.add_argument("-g", "--gamma", type=float, default=0.1,
                        help="gamma (0.1)",
                        dest="g")
    parser.add_argument("-a", "--alpha", type=float, default=0.03,
                        help="alpha (0.03)",
                        dest="a")
    parser.add_argument("-bnf", "--batch_norm_fraction", default=0.1, type=float,
                        help="batchnorm layers learner samples fraction (0.1)",
                        dest="bnp")
    parser.add_argument("-f", "--fraction", default=1.0, type=float,
                        help="dataset fraction (1) [-1 for None]",
                        dest="p")
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4,
                        help="weight decay (5e-4) [-1 for None]",
                        dest="wd")
    parser.add_argument("-cn", "--clip_gradient_norm", type=float, default=1,
                        help="gradient max norm (1) [-1 for None]",
                        dest="cn")
    parser.add_argument("-m", "--momentum", type=float, default=0.7,
                        help="momentum (0.7) [-1 for None]",
                        dest="m")
    parser.add_argument("-n", "--nesterov", action="store_true", default=False,
                        help="use nesterov momentum",
                        dest="n")
    parser.add_argument("-cr", "--compression_rate", type=float, default=0.01,
                        help="compression rate (0.01) [-1 for None]",
                        dest="cr")

    parser.add_argument("-s", "--silent", action="store_true", default=False,
                        help="silent mode",
                        dest="s")
    parser.add_argument("-csv", "--csv", action="store_true", default=False,
                        help="save to csv",
                        dest="csv")
    parser.add_argument("-tb", "--tensorboard", action="store_true", default=False,
                        help="use tensorboard",
                        dest="tb")
    parser.add_argument("-c", "--checkpoints", default=-1, type=int,
                        help="checkpoints interval (None)",
                        dest="c")
    parser.add_argument("-cuda", "--cuda", action="store_true", default=False,
                        help="use cuda",
                        dest="cuda")

    args = parser.parse_args()

    args.cr = None if args.cr <= 0 else args.cr
    args.cn = None if args.cn <= 0 else args.cn
    args.m = None if args.m <= 0 else args.m
    args.wd = None if args.wd < 0 else args.wd
    args.a = None if args.a < 0 else args.a

    args.c = None if args.c <= 0 else args.c
    args.p = None if args.p <= 0 else args.p
    args.bnp = 0.1 if args.bnp <= 0 else args.bnp

    args.spr = None if args.spr <= 0 else args.spr
    args.sppull = 1 if args.sppull <= 0 else args.sppull

    if (args.g <= 0) or not args.ms:
        args.g = None
        args.ms = None

    entity = Server(args)

    entity.run()


if __name__ == "__main__":
    main()
