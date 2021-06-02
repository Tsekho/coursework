from script.utils import get_train_dataloader, get_network, \
    compress, fpath, TAGS, Communicator, STDOUT_CLIENT

import sys
from time import time

import torch
import torch.nn.functional as F

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)


class Client:
    """
    Client side related work
    TODO: move logging to separate classes,
          decide on calling them either workers or clients
    """

    def __init__(self, args):
        self.args = args
        self.wid = -1
        self.param_version = 0
        if not torch.cuda.is_available():
            self.args.cuda = False
            self.handle_print("Cuda is not available, "
                              "CPU will be used instead.")

        self.loader = get_train_dataloader(self.args)
        self.net = get_network(self.args)
        self.net.train()
        self.comm = Communicator(fpath("keys_client"), self.args)
        self.tags = TAGS

        self.n = list(filter(lambda x: x.requires_grad, self.net.parameters()))
        self.u = [torch.zeros_like(p) for p in self.net.parameters()]
        self.v = [torch.zeros_like(p) for p in self.net.parameters()]
        self.o = [torch.zeros_like(p) for p in self.net.parameters()]

        self.steps_finished = 0

        self.time = time()
        self.log_buf = []
        self.log_obj = {"H": "", "V": ""}

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
            n.copy_(p)
        p.detach()

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
            l_v_abs = l_v.abs()
            s = l_v_abs.view(-1)[torch.randint(ne, (max(ne // 100, 1),))]
            t = torch.topk(s, max(1, int(s.numel() * self.args.cr)))[0][-1]
            mask = 0
            for _ in range(11):
                mask = l_v_abs > t
                selected = mask.sum()
                if selected > 1.3 * ne * self.args.cr:
                    t *= 1.3
                elif selected < 0.7 * ne * self.args.cr:
                    t *= 0.7
                else:
                    break
        l_o.copy_(l_v.mul(mask))
        l_v.mul_(~mask)
        l_u.mul_(~mask)

    @torch.no_grad()
    def apply_lu(self, p, l_v, l_u, l_o):
        """
        Applies local update (bad idea, skipped for now)
        """
        lr = self.args.lr
        e = self.param_version // self.args.spe
        if self.args.g is not None:
            for s in self.args.ms:
                if e >= s:
                    lr *= self.args.g
                else:
                    break
        p.add_(l_o, alpha=-lr)

    def step(self):
        self.steps_finished += 1
        for t in zip(self.n, self.u, self.v, self.o):
            self.apply_wd(*t)
            self.apply_cn(*t)
            self.apply_mc(*t)
            self.apply_cr(*t)
            if self.steps_finished % self.args.sppull:
                self.apply_lu(*t)

    @torch.no_grad()
    def evaluate(self, loss, outputs, labels):
        self.log_obj["trn_a"] = (outputs.max(1)[1] ==
                                 labels).sum().item() / len(outputs)
        self.log_obj["trn_l"] = loss.item()
        self.log_obj["grd_n"] = sum((x.norm().item() ** 2
                                     for x in self.o)) ** (0.5)
        self.log_obj["stp"] = self.steps_finished
        self.log_obj["prmv"] = self.param_version

    def run(self):
        while True:
            for images, labels in self.loader:
                self.timer()

                if not self.steps_finished % self.args.sppull:
                    self.comm.send(self.wid, self.tags.PULL)
                    tag, data = self.comm.recv()
                    pv, params, wid = data
                    self.wid = wid
                    self.param_version = pv
                    self.log_obj["H"] = "VER"
                    self.log_obj["V"] = "{:8d}".format(self.param_version)
                    self.copy_params(params)

                if self.args.spr is not None and not (self.steps_finished % self.args.spr) \
                        and self.steps_finished > 0:
                    self.comm.send([self.log_buf, self.wid], self.tags.STATS)
                    self.comm.recv()
                    self.log_buf = []

                # receive time
                self.timer("recv")

                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()
                self.net.zero_grad()
                outputs = self.net(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.step()
                # calculation time
                self.timer("calc")

                self.evaluate(loss, outputs, labels)
                # evaluation time
                self.timer("eval")
                self.comm.send([self.param_version, compress(self.o), self.wid],
                               self.tags.GRADS)
                self.comm.recv()
                # send time
                self.timer("send")

                self.log_buf.append(self.log_obj)
                if not self.args.s:
                    print(STDOUT_CLIENT.format(**self.log_obj))
                    self.log_obj = {"H": "", "V": ""}
                sys.stdout.flush()
