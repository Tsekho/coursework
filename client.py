import sys
from time import time

import torch
import torch.nn.functional as F

from mpi4py import MPI

from utils import get_train_dataloader, get_network, compress, TAGS


class Client:
    """
    Client side related work
    TODO: move logging to separate classes,
          decide on calling them either workers or clients
    """

    def __init__(self, args):
        self.loader = get_train_dataloader(args)
        self.net = get_network(args)
        self.net.train()
        self.comm = MPI.COMM_WORLD
        self.tags = TAGS

        self.n = list(filter(lambda x: x.requires_grad, self.net.parameters()))
        self.u = [torch.zeros_like(p) for p in self.net.parameters()]
        self.v = [torch.zeros_like(p) for p in self.net.parameters()]
        self.o = [torch.zeros_like(p) for p in self.net.parameters()]

        self.args = args

        self.cr = self.args.cr
        self.epoch = 0
        self.steps_finished = 0

        self.time = time()
        self.log_buf = []
        self.log_obj = {}

    def timer(self, label=None):
        """
        Handy for time loggings
        TODO: move to separate class in utils.py
        """
        if label is not None:
            self.log_obj["time_" + label] = time() - self.time
        self.time = time()

    @torch.no_grad()
    def copy_params(self, params):
        for n, p in zip(self.n, params):
            if self.args.cuda:
                p = p.cuda()
            n.copy_(p)

    @torch.no_grad()
    def apply_wd(self, p, l_v, l_u, l_o):
        """
        Applies weight decay
        """
        if self.args.wd is not None:
            p.grad.data.add_(p.data, alpha=self.args.wd)

    @torch.no_grad()
    def apply_cn(self, p, l_v, l_u, l_o):
        """
        Applies norm clipping
        """
        if self.args.cn is not None:
            c = self.args.cn / (p.grad.norm() + 1e-6)
            if c < 1:
                p.grad.mul_(c)

    @torch.no_grad()
    def apply_mc(self, p, l_v, l_u, l_o):
        """
        Applies momentum correction
        """
        if self.args.m is not None:
            if self.args.n:
                l_u.add_(p.grad).mul_(self.args.m)
                l_v.add_(l_u).add_(p.grad)
            else:
                l_u.mul_(self.args.m).add_(p.grad)
                l_v.add_(l_u)
        else:
            l_v.add_(p.grad)

    @torch.no_grad()
    def apply_cr(self, p, l_v, l_u, l_o):
        """
        Applies compression
        """
        mask = torch.tensor(True)
        if self.args.cr is not None:
            ne = p.data.numel()
            self.cr = self.args.cr
            if (self.args.w is not None) and (self.epoch < self.args.w):
                self.cr = max(self.args.cr, self.args.wcr ** (self.epoch + 1))
            l_v_abs = l_v.abs()
            s = l_v_abs.view(-1)[torch.randint(ne, (max(ne // 100, 1),))]
            t = torch.topk(s, max(1, int(s.numel() * self.cr)))[0][-1]
            mask = 0
            for _ in range(11):
                mask = l_v_abs > t
                selected = mask.sum()
                if selected > 1.3 * ne * self.cr:
                    t *= 1.3
                elif selected < 0.7 * ne * self.cr:
                    t *= 0.7
                else:
                    break
        l_o.copy_(l_v.mul(mask))
        l_v.mul_(~mask)
        l_u.mul_(~mask)

    @torch.no_grad()
    def step(self):
        for t in zip(self.n, self.u, self.v, self.o):
            self.apply_wd(*t)
            self.apply_cn(*t)
            self.apply_mc(*t)
            self.apply_cr(*t)

    @torch.no_grad()
    def evaluate(self, loss, outputs, labels):
        self.log_obj["trn_a"] = (outputs.max(1)[1] ==
                                 labels).sum().item() / len(outputs)
        self.log_obj["trn_l"] = loss.item()
        self.log_obj["grd_n"] = sum((x.norm().item() ** 2
                                     for x in self.o)) ** (0.5)
        self.log_obj["stp"] = self.steps_finished
        self.log_obj["cr"] = self.cr

    def run(self):
        # tell server the client is ready
        self.comm.send(len(self.loader), dest=0, tag=self.tags.START)
        for self.epoch in range(self.args.e):
            self.loader.sampler.set_epoch(self.epoch)
            for images, labels in self.loader:
                self.timer()
                sys.stdout.flush()

                params = self.comm.recv(source=0, tag=self.tags.PARAMS)
                # receive time
                self.timer("recv")

                self.copy_params(params)
                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()
                self.net.zero_grad()
                outputs = self.net(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.step()
                # calculation time
                self.timer("calc")

                self.steps_finished += 1
                self.evaluate(loss, outputs, labels)
                # evaluation time
                self.timer("eval")

                self.comm.send(compress(self.o), dest=0, tag=self.tags.GRADS)
                # send time
                self.timer("send")

                self.log_buf.append(self.log_obj)
                self.log_obj = {}
            self.comm.send(self.log_buf, dest=0, tag=self.tags.STATS)
            self.log_buf = []
        self.comm.send(None, dest=0, tag=self.tags.FINISH)
