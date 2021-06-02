from script.utils import get_test_dataloader, get_train_light_dataloader, \
    get_network, decompress, fpath, fmdir, \
    TAGS, Communicator, \
    TIME_NOW, ARGS_FORMAT, STDOUT_HEADER, STDOUT_FORMAT, STDOUT_FINAL, \
    CSV_TRAIN_HEADER, CSV_TRAIN_FORMAT, CSV_TEST_HEADER, CSV_TEST_FORMAT

import sys
from time import time
import os

import torch
import torch.nn.functional as F

from pickle import UnpicklingError

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)


class Server:
    """
    Parameter server side related work
    TODO: redesign to pass optimizer and lr_scheduler,
          move logging to separate classes
    """

    def __init__(self, args):
        args.srv = True
        self.args = args
        if not torch.cuda.is_available():
            self.args.cuda = False
            self.handle_print("Cuda is not available, "
                              "CPU will be used instead.")

        self.loader = get_test_dataloader(self.args)
        sys.stdout.flush()
        self.net = get_network(self.args)
        self.net.eval()
        self.comm = Communicator(fpath("keys_server"), self.args)
        self.tags = TAGS

        self.n = list(filter(lambda x: x.requires_grad, self.net.parameters()))

        self.active = 0

        self.lr = self.args.lr
        self.epoch = 0
        self.steps_finished = 0

        self.init_logging()

    def init_logging(self):
        self.time = [0] * 5

        self.log_obj = {"time_logw": 0.0, "time_send": 0.0}

        self.best_acc = 0
        self.best_epoch = 0

        self.workpath = fpath("runs", self.args.net, TIME_NOW)
        self.nestpath = lambda *x: fpath(self.workpath, *x)
        fmdir(self.workpath)

        # desc.txt file
        self.stdout_writer = open(self.nestpath("desc.txt"),
                                  "w", encoding="utf-8")
        s = ARGS_FORMAT.format(**vars(self.args))
        self.stdout_writer.write(s)
        self.stdout_writer.flush()

        # tensorboard logging
        if self.args.tb:
            fmdir(self.nestpath("tensorboard"))
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(
                log_dir=self.nestpath("tensorboard"))
            input_tensor = torch.zeros((1, 3, 32, 32))
            if self.args.cuda:
                input_tensor = input_tensor.cuda()
            self.tb_writer.add_graph(self.net, input_tensor)

        # csv logging
        if self.args.csv:
            fmdir(self.nestpath("csv"))
            with open(self.nestpath("csv", "server.csv"),
                      "w", encoding="utf-8") as f:
                f.write(CSV_TEST_HEADER)

        # checkpoints
        if self.args.c is not None:
            fmdir(self.nestpath("checkpoints"))

    def timer(self, i=0, label=None, count=1):
        """
        Handy for time loggings
        TODO: move to separate class in utils.py
        """
        if label is not None:
            label = "time_" + label
            self.log_obj[label] = self.log_obj.get(label, 0)
            self.log_obj[label] += (time() - self.time[i]) / count
        self.time[i] = time()

    def handle_print(self, string):
        """
        Handles stdout output
        TODO: move to separate class in utils.py
        """
        if not self.args.s:
            print(string, end="")
            sys.stdout.flush()
        self.stdout_writer.write(string)
        self.stdout_writer.flush()

    def progress_bar(self):
        """
        Handy 5% precision epoch progress bar
        TODO: move to separate class in utils.py
        """
        s_count = 0
        while True:
            step = (self.steps_finished - 1) % self.args.spe + 1
            while ((s_count + 1) * self.args.spe <= 20 * step) and (s_count < 20):
                self.handle_print("^")
                s_count += 1
            if step == self.args.spe:
                s_count = 0
                self.handle_print(" |\n")
            yield

    def write_stats(self, wid=None, data=None):
        """
        TensorBoard and CSV output formatting
        TODO: move to separate class in utils.py
        """
        if wid is None:
            if self.args.tb:
                for t in ["tst_a", "tst_l", "time_recv", "time_epch"]:
                    self.tb_writer.add_scalar("server/" + t,
                                              self.log_obj[t],
                                              self.log_obj["epc"])
            if self.args.csv:
                fp = self.nestpath("csv", "server.csv")
                with open(fp, "a", encoding="utf-8") as f:
                    f.write(CSV_TEST_FORMAT.format(**self.log_obj))
        else:
            if self.args.tb:
                for lo in data:
                    for t in ["trn_a", "trn_l", "time_calc", "grd_n"]:
                        self.tb_writer.add_scalar("worker{}/{}".format(wid, t),
                                                  lo[t],
                                                  lo["stp"])
            if self.args.csv:
                for lo in data:
                    fp = self.nestpath("csv", "worker{}.csv".format(wid))
                    if not os.path.exists(fp):
                        with open(fp, "w", encoding="utf-8") as f:
                            f.write(CSV_TRAIN_HEADER)
                    with open(fp, "a", encoding="utf-8") as f:
                        f.write(CSV_TRAIN_FORMAT.format(**lo))

    def report_stats(self, start=False, end=False):
        """
        Stdout output formatting
        TODO: move to separate class in utils.py
        """
        if start:
            s = STDOUT_HEADER.format(self.args.net.upper(), TIME_NOW)
            self.handle_print(s)
        elif end:
            s = STDOUT_FINAL.format(self.epoch, self.best_acc, self.log_obj["time_l"],
                                    self.log_obj["time_l"] / self.args.e, self.best_epoch)
            self.handle_print(s)
        else:
            s = STDOUT_FORMAT.format(**self.log_obj)
            self.handle_print(s)

    @torch.no_grad()
    def update_params(self, pv, params, lr):
        """
        Applies received gradients
        """
        for n, g in zip(self.n, params):
            if self.args.cuda:
                g = g.cuda()
            k = self.args.a * (self.steps_finished - pv)
            n.view(-1).add_(g, alpha=-(lr / (1 + k)))

    @torch.no_grad()
    def update_bn_params(self):
        """
        Learn BN parameters locally
        """
        self.net.train()
        loader = get_train_light_dataloader(self.args)
        for images, labels in loader:
            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()
            self.net(images)
        self.net.eval()

    def update_lr(self):
        lr = self.args.lr
        if self.args.g is not None:
            for s in self.args.ms:
                if self.epoch >= s:
                    lr *= self.args.g
                else:
                    break
        self.lr = lr

    @torch.no_grad()
    def evaluate(self):
        total = 0
        correct = 0
        loss = 0
        for images, labels in self.loader:
            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = self.net(images)
            loss += F.cross_entropy(outputs, labels)
            total += len(outputs)
            correct += (outputs.max(1)[1] == labels).sum().item()

        self.log_obj["tst_a"] = correct / total
        self.log_obj["tst_l"] = loss.item()
        self.log_obj["stp"] = self.steps_finished
        self.log_obj["epc"] = self.epoch
        self.log_obj["lr"] = self.lr
        if self.log_obj["tst_a"] > self.best_acc:
            self.best_acc = self.log_obj["tst_a"]
            self.best_epoch = self.epoch

    def process_epoch(self):
        # average step time without epoch processing included
        self.timer(2, "btch", self.args.spe)

        self.update_bn_params()
        # batchnorm params training time (once per epoch)
        self.timer(3, "ubnl")

        self.epoch = self.steps_finished // self.args.spe
        self.update_lr()
        self.evaluate()
        # evaluation time (once per epoch)
        self.timer(3, "eval")

        if (self.args.c is not None) and not (self.epoch % self.args.c):
            torch.save(self.net.state_dict(),
                       self.nestpath("checkpoints",
                                     "regular-{}.pt".format(self.epoch)))
        if self.epoch == self.best_epoch and (self.args.c is not None):
            torch.save(self.net.state_dict(),
                       self.nestpath("checkpoints",
                                     "best.pt"))
        # checkpoint save time (once per epoch)
        self.timer(3, "chck")

        # epoch done time (needed before writing logs)
        self.timer(1, "epch")

        self.write_stats()
        self.report_stats()
        self.log_obj = {"time_logw": 0.0, "time_send": 0.0}
        # log writing total time after last write/report_stats() call
        self.timer(3, "logw")

        self.timer(2)

    def run(self):
        self.report_stats(start=True)
        self.bar = self.progress_bar()

        # init timers
        self.timer(0)  # learning timer
        self.timer(1)  # epoch done timer
        self.timer(2)  # steps done timer
        self.timer(3)  # general timer
        self.timer(4)  # reserved

        while self.steps_finished != self.args.spe * self.args.e:
            try:
                tag, data = self.comm.recv()
            except UnpicklingError:
                self.comm.send(None, -1)
                continue
            if tag == self.tags.INIT:
                self.comm.send(self.args, self.tags.ARGS)
            elif tag == self.tags.PULL:
                self.timer(3)

                wid = data
                if wid == -1:
                    self.active += 1
                    self.comm.send([self.steps_finished, self.n, self.active],
                                   self.tags.PARAMS)
                else:
                    self.comm.send([self.steps_finished, self.n, wid],
                                   self.tags.PARAMS)
                # average request + send_params time per step
                self.timer(3, "send", self.args.spe)

            elif tag == self.tags.GRADS:
                pv, params, wid = data
                self.comm.send(None, self.tags.CONFIRM_GRADS)
                # average receive_grads + reply time per step
                self.timer(3, "recv", self.args.spe)

                self.update_params(pv, decompress(params), self.lr)
                self.steps_finished += 1
                next(self.bar)
                # average parameter update time
                self.timer(3, "ugrd", self.args.spe)

                if self.steps_finished // self.args.spe > self.epoch:
                    self.process_epoch()
                self.timer(3)

            elif tag == self.tags.STATS:
                self.timer(3)

                logs, wid = data
                self.comm.send(None, self.tags.CONFIRM_STATS)
                self.write_stats(wid, logs)
                # second point of log writing
                self.timer(3, "logw")
            else:
                self.comm.send(None, -1)

        if self.args.tb:
            self.tb_writer.close()

        # total learning time without initializations included
        self.timer(0, "l")
        self.report_stats(end=True)

        self.stdout_writer.close()
