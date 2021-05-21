
import argparse
import os
import time


import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import accuracy_score

from optimizer import DGC

TIME_NOW = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def train(args):
    # DATALOADERS
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                           transform=transform_test)
    if args.ws != 1:
        if args.r != 0:
            sampler = DistributedSampler(trainset, args.ws - 1, args.r - 1)
        # else:
        #   start server
    else:
        sampler = DistributedSampler(trainset, 1, 0)
        # start server
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b,
                                              shuffle=False, num_workers=2,
                                              sampler=sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False, num_workers=2)

    # NETWORK
    if args.net == "resnet18":
        from models import ResNet18
        net = ResNet18()
    if args.net == "resnet34":
        from models import ResNet34
        net = ResNet34()
    elif args.net == "resnet50":
        from models import ResNet50
        net = ResNet50()
    if args.cuda:
        net = net.cuda()

    # OPTIMIZER
    optimizer = DGC(net, args=args)

    # SCHEDULER
    scheduler = MultiStepLR(optimizer, milestones=args.ms, gamma=args.g)
    net.train()

    # TENSORBOARD LOGGER
    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(
            "runs", args.net, TIME_NOW))
        input_tensor = torch.Tensor(1, 3, 32, 32)

        if args.cuda:
            input_tensor = input_tensor.cuda()
        writer.add_graph(net, input_tensor)

    # TRAIN
    mloss = 0
    step = 0
    for epoch in range(args.e):
        lr = optimizer.get_lr()
        cr = optimizer.get_cr(epoch)
        if not args.s:
            print("= Epoch {} | LR: {} | CR: {}".format(epoch + 1, lr, cr))
        net.train()
        trainloader.sampler.set_epoch(epoch)
        start = time.time()
        logger = {}
        for i, data in enumerate(trainloader):
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # DO CALCULATIONS
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step(epoch)

            # DO SCORING
            _, predicted = torch.max(outputs, 1)
            step += 1
            mloss = 0.9 * mloss + 0.1 * loss.item()
            accuracy = accuracy_score(predicted.cpu(), labels.cpu())

            # DO LOGGING
            logger = {"timestamp": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
                      "epoch": epoch + 1, "step": i,
                      "train_loss": loss.item(), "train_mloss": mloss, "train_acc": accuracy}
            if args.tb:
                writer.add_scalar("step/train_mloss",
                                  logger["train_mloss"], step)
                writer.add_scalar("step/train_loss",
                                  logger["train_loss"], step)
                writer.add_scalar("step/train_acc",
                                  logger["train_acc"], step)
                writer.add_scalar("norm/grad_norm", optimizer.out.norm(), step)
            if not i % 10 and not args.s:
                print("== {timestamp} | "
                      "Step: {step:6} | "
                      "Loss: {train_loss:6.4f} | "
                      "MLoss: {train_mloss:6.4f} | "
                      "Accuracy: {train_acc:6.4f}".format(**logger))

        scheduler.step()
        end = time.time()
        delta = end - start
        delta_avg = delta / len(trainloader)

        # MORE SCORING
        net.eval()
        total = 0
        correct = 0
        test_loss = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if args.cuda:
                    images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                test_loss += F.cross_entropy(outputs, labels).item()
                total += labels.size(0)
                correct += (predicted == labels).sum()

        # MORE LOGGING
        logger["test_loss"] = test_loss
        logger["test_acc"] = correct.item() / total
        if args.tb:
            writer.add_scalar("epoch/train_mloss",
                              logger["train_mloss"], epoch + 1)
            writer.add_scalar("epoch/test_loss",
                              logger["test_loss"], epoch + 1)
            writer.add_scalar("epoch/test_acc",
                              logger["test_acc"], epoch + 1)
        if not args.s:
            print("== {timestamp} | "
                  "Step: {step:6} | "
                  "Loss: {train_loss:6.4f} | "
                  "MLoss: {train_mloss:6.4f} | "
                  "Accuracy: {train_acc:6.4f}\n"
                  "=== Test Loss: {test_loss:6.4f} | "
                  "Test Accuracy: {test_acc:6.4f}".format(**logger))
        if not args.s:
            print("=== AVG Batch: {:6.3f}s | "
                  "Epoch: {:6.3f}s\n".format(delta_avg, delta))
    if args.tb:
        writer.close()
    if not args.s:
        print("Finished Training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=50,
                        help="total epochs (50)")
    parser.add_argument("-w", type=int, default=5,
                        help="warming epochs (5)")
    parser.add_argument("-b", type=int, default=64,
                        help="batch size (64)")
    parser.add_argument("-lr", type=float, default=0.1,
                        help="base learning rate (0.1)")
    parser.add_argument("-ms", nargs="*", type=int, default=[30, 40],
                        help="milestones (30, 40)")
    parser.add_argument("-g", type=float, default=0.1,
                        help="gamma (0.1)")
    parser.add_argument("-wcr", type=float, default=0.5,
                        help="warming compression rate (0.5)")
    parser.add_argument("-cr", type=float, default=0.01,
                        help="compression rate (0.01)")
    parser.add_argument("-cn", type=float, default=1,
                        help="gradient max norm (0.25)")
    parser.add_argument("-m",  type=float, default=0.7,
                        help="nesterov momentum (0.7)")
    parser.add_argument("-wd", type=float, default=5e-4,
                        help="weight decay (5e-4)")
    parser.add_argument("-cuda", action="store_true", default=False,
                        help="use cuda (False)")
    parser.add_argument("-net", default="resnet18", type=str,
                        help="network, one of (resnet18), resnet34, resnet50")
    parser.add_argument("-s", action="store_true", default=False,
                        help="silent")
    parser.add_argument("-tb", action="store_true", default=False,
                        help="use tensorboard")

    parser.add_argument("-ws", type=int, default=1,
                        help="world size")
    parser.add_argument("-r", type=int, default=0,
                        help="rank")

    args = parser.parse_args()

    args.w = None if args.w < 0 else args.w
    args.wcr = None if args.wcr < 0 else args.wcr
    args.cr = None if args.cr < 0 else args.cr
    args.cn = None if args.cn < 0 else args.cn
    args.m = None if args.m < 0 else args.m
    args.wd = None if args.wd < 0 else args.wd

    if not args.s:
        print(args)
    train(args)
