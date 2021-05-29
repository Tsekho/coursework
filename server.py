import sys
from time import time

import torch
import torch.nn.functional as F

from mpi4py import MPI

from utils import get_test_dataloader, get_train_light_dataloader, \
    get_network, decompress, fpath, fmdir, \
    TAGS, TIME_NOW, ARGS_FORMAT, STDOUT_HEADER, STDOUT_FORMAT, STDOUT_FINAL, \
    CSV_TRAIN_HEADER, CSV_TRAIN_FORMAT, CSV_TEST_HEADER, CSV_TEST_FORMAT


class Server:
    """
    Parameter server side related work
    TODO: redesign to pass optimizer and lr_scheduler,
          move logging to separate classes
    """

    def __init__(self, args):
        self.loader = get_test_dataloader(args)
        sys.stdout.flush()
        self.net = get_network(args)
        self.net.eval()
        self.comm = MPI.COMM_WORLD
        self.status = MPI.Status()
        self.tags = TAGS

        self.n = list(filter(lambda x: x.requires_grad, self.net.parameters()))

        self.args = args

        self.total = self.args.ws - 1
        self.active = 0

        self.lr = self.args.lr
        self.epoch = 0
        self.steps_total = 0
        self.steps_finished = 0
        self.steps_left = [0] * self.total

        self.init_logging()

    def init_logging(self):
        self.time = [0] * 5

        self.log_obj = {}

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
            for i in range(1, self.args.ws):
                with open(self.nestpath("csv", "worker{}.csv".format(i)),
                          "w", encoding="utf-8") as f:
                    f.write(CSV_TRAIN_HEADER)

        # checkpoints
        if self.args.c is not None:
            fmdir(self.nestpath("checkpoints"))

    def timer(self, i=0, label=None, s=1):
        """
        Handy for time loggings
        TODO: move to separate class in utils.py
        """
        if label is not None:
            label = "time_" + label
            self.log_obj[label] = self.log_obj.get(label, 0)
            self.log_obj[label] += (time() - self.time[i]) / s
        self.time[i] = time()

    def handle_print(self, s):
        """
        Handles stdout output
        TODO: move to separate class in utils.py
        """
        if not self.args.s:
            print(s, end="")
            sys.stdout.flush()
        self.stdout_writer.write(s)
        self.stdout_writer.flush()

    def progress_bar(self):
        """
        Handy 5% precision epoch progress bar
        TODO: move to separate class in utils.py
        """
        s_count = 0
        while True:
            step = (self.steps_finished - 1) % self.steps_total + 1
            while ((s_count + 1) * self.steps_total <= 20 * step) and (s_count < 20):
                self.handle_print("^")
                s_count += 1
            if step == self.steps_total:
                s_count = 0
                self.handle_print(" |\n")
            yield

    def write_stats(self, data=None, source=None):
        """
        TensorBoard and CSV output formatting
        TODO: move to separate class in utils.py
        """
        if source is None:
            if self.args.tb:
                for t in ["tst_a", "tst_l", "time_recv", "time_epch"]:
                    self.tb_writer.add_scalar("server/" + t,
                                              self.log_obj[t],
                                              self.log_obj["epc"])
            if self.args.csv:
                with open(self.nestpath("csv", "server.csv"),
                          "a", encoding="utf-8") as f:
                    f.write(CSV_TEST_FORMAT.format(**self.log_obj))
        else:
            if self.args.tb:
                for lo in data:
                    for t in ["trn_a", "trn_l", "time_calc", "grd_n"]:
                        self.tb_writer.add_scalar("worker{}/{}".format(source, t),
                                                  lo[t],
                                                  lo["stp"])
            if self.args.csv:
                for lo in data:
                    with open(self.nestpath("csv", "worker{}.csv".format(source)),
                              "a", encoding="utf-8") as f:
                        f.write(CSV_TRAIN_FORMAT.format(**lo))

    def report_stats(self, start=False, end=False):
        """
        Stdout output formatting
        TODO: move to separate class in utils.py
        """
        if start:
            s = STDOUT_HEADER.format(self.args.net.upper(),
                                     self.args.ws - 1, TIME_NOW)
            self.handle_print(s)
        elif end:
            s = STDOUT_FINAL.format(self.epoch, self.best_acc, self.log_obj["time_l"],
                                    self.log_obj["time_l"] / self.args.e, self.best_epoch)
            self.handle_print(s)
        else:
            s = STDOUT_FORMAT.format(**self.log_obj)
            self.handle_print(s)

    @ torch.no_grad()
    def update_params(self, params, lr):
        """
        Applies received gradients
        """
        for n, g in zip(self.n, params):
            if self.args.cuda:
                g = g.cuda()
            n.view(-1).add_(g, alpha=-lr)

    @ torch.no_grad()
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

    @ torch.no_grad()
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
        self.timer(2, "btch", self.steps_total)

        self.update_bn_params()
        # batchnorm params training time (once per epoch)
        self.timer(3, "ubnl")

        self.epoch = self.steps_finished // self.steps_total
        if (self.args.g is not None) and (self.epoch in self.args.ms):
            self.lr *= self.args.g
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
        self.log_obj = {}
        # log writing total time after last write/report_stats() call
        self.timer(3, "logw")

        self.timer(2)

    def run(self):
        # initializing communication
        while self.active != self.total:
            data = self.comm.recv(source=MPI.ANY_SOURCE,
                                  tag=self.tags.START,
                                  status=self.status)
            source = self.status.Get_source()
            self.steps_total += data
            self.steps_left[source - 1] = data * self.args.e
            self.comm.send(self.n, dest=source, tag=self.tags.PARAMS)
            self.active += 1

        self.report_stats(start=True)
        self.bar = self.progress_bar()

        # init timers
        self.timer(0)  # learning timer
        self.timer(1)  # epoch done timer
        self.timer(2)  # steps done timer
        self.timer(3)  # general timer
        self.timer(3, "logw")  # init value of 0
        self.timer(4)  # reserved

        while self.active != 0:
            data = self.comm.recv(source=MPI.ANY_SOURCE,
                                  tag=MPI.ANY_TAG,
                                  status=self.status)
            source = self.status.Get_source()
            tag = self.status.Get_tag()

            if tag == self.tags.GRADS:
                # average receive time
                self.timer(3, "recv", self.steps_total)

                self.steps_finished += 1
                self.steps_left[source - 1] -= 1
                next(self.bar)
                self.update_params(decompress(data), self.lr)
                # average parameter update time
                self.timer(3, "ugrd", self.steps_total)

                if self.steps_left[source - 1]:
                    self.comm.send(self.n, dest=source, tag=self.tags.PARAMS)
                # average send time
                self.timer(3, "send", self.steps_total)

                if self.steps_finished // self.steps_total > self.epoch:
                    self.process_epoch()
                self.timer(3)

            elif tag == self.tags.STATS:
                self.timer(3)
                self.write_stats(data, source)
                # second point of log writing
                self.timer(3, "logw")

            elif tag == self.tags.FINISH:
                self.active -= 1
                self.timer(3)

        if self.args.tb:
            self.tb_writer.close()

        # total learning time without initializations included
        self.timer(0, "l")
        self.report_stats(end=True)

        self.stdout_writer.close()
