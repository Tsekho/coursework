import torch

from torch.optim.optimizer import Optimizer


def compress(temp_param):
    indices = temp_param.nonzero()
    values = temp_param[indices]
    sparse_gradient = torch.cat((indices.double(), values.double())).view(-1)
    return sparse_gradient


def decompress(sparse_gradient, size):
    split = len(sparse_gradient) // 2
    i = sparse_gradient[:split]
    v = sparse_gradient[split:]
    size = torch.Size([size])
    dense_gradient = torch.sparse_coo_tensor(i.reshape(1, -1).long(),
                                             v.float(), size)
    return dense_gradient


def update(net, parameter_update, lr):
    ci = 0
    for parameter in net.parameters():
        ne = parameter.data.numel()
        size = parameter.data.size()
        parameter.data.add_(parameter_update[ci:ci + ne].view(size), alpha=-lr)
        ci += ne


class DGC(Optimizer):
    def __init__(self, net, args=None):
        self.net = net
        p = torch.zeros(0)
        if args.cuda:
            p = p.cuda()
        for parameter in net.parameters():
            p = torch.cat((p, parameter.data.view(-1)))

        self.out = p.clone()
        self.v = self.out.clone().zero_()
        self.u = self.out.clone().zero_()

        self.w = args.w
        self.wcr = args.wcr
        self.cr = args.cr

        self.cn = args.cn
        self.m = args.m
        self.wd = args.wd

        self.cuda = args.cuda
        super(DGC, self).__init__(net.parameters(), {"lr": args.lr})

    def get_lr(self):
        return self.param_groups[0]['lr']

    def get_cr(self, epoch):
        if self.w and (epoch < self.w):
            return max(self.cr, self.wcr ** (epoch + 1))
        return self.cr

    @torch.no_grad()
    def step(self, epoch):
        ci = 0
        for p in self.net.parameters():
            ne = p.data.numel()
            if p.grad is None:
                ci += ne
                continue
            grad = p.grad.data.view(-1)
            l_u = self.u[ci:ci + ne]
            l_v = self.v[ci:ci + ne]
            mask = torch.tensor(True)
            self.w = 0 if (self.w is None) else self.w

            # WEIGHT DECAY
            if self.wd is not None:
                p.grad.data.add_(p.data, alpha=self.wd)

            # GRADIENT CLIPPING
            if self.cn is not None:
                c = self.cn / (grad.norm() + 1e-6)
                if c < 1:
                    grad.data.mul_(c)

            # NESTEROV MOMENTUM CORRECTION
            if self.m is not None:
                l_u.mul_(self.m).add_(grad)
                grad.add_(l_u, alpha=self.m)

            # LOCAL GRADIENT ACCUMULATION
            l_v.add_(grad)

            # GRADIENT SPARSIFICATION WITH WARM UP COMPRESSION RATE
            if self.cr is not None:
                cr = self.get_cr(epoch)
                l_v_abs = l_v.abs()
                s = l_v_abs[torch.randint(ne, (max(ne // 100, 1),))]
                t = torch.topk(s, max(1, int(s.numel() * cr)))[0][-1]
                mask = 0
                for _ in range(11):
                    mask = l_v_abs > t
                    selected = mask.sum()
                    if selected > 1.3 * ne * cr:
                        t *= 1.3
                    elif selected < 0.7 * ne * cr:
                        t *= 0.7
                    else:
                        break
            self.out[ci:ci + ne].copy_(l_v.mul(mask))
            l_v.mul_(~mask)
            l_u.mul_(~mask)
            ci += ne
        msg = compress(self.out)

        # SEND - RECEIVE
        gradients = decompress(msg, self.out.numel())

        if self.cuda:
            gradients = gradients.cuda()
        gradients = gradients.to_dense()

        lr = self.get_lr()
        update(self.net, gradients, lr)
