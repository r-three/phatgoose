import random
import re
from collections import defaultdict

import gin
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule


def norm(v, dim=1):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def unit(v, dim=1, eps=1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
    [p, n] = tan_vec.size()
    tan_vec.t_()
    q, r = torch.linalg.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()

    return q


def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out


def Cayley_loop(X, W, tan_vec, t):  #
    [n, p] = X.size()
    Y = X + t * tan_vec
    for i in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))

    return Y.t()


def check_identity(X):  # n-by-p
    n, p = X.size()
    res = torch.eye(p).cuda() - torch.mm(X.t(), X)
    print("n={0}, p={1}, res norm={2}".format(n, p, torch.norm(res)))


class AdamW_Orth(optim.AdamW):
    def custom_step(self, router_param_groups, lr):
        # for group in router_param_groups:
        # need to combine expert_embeddings into a matrix based on group["params_name"]
        # momentum = group['betas'][0]
        # for index,p in enumerate(group['params']):
        #     if p.grad is None:
        #         continue
        # with torch.no_grad():
        #     p_remaining = torch.stack(group['other_params'][index], dim=0)
        #     p_grad_remaining = torch.zeros_like(p_remaining)
        #     p_full = torch.cat([p.unsqueeze(0), p_remaining], dim=0)
        # unity,_ = unit(p_full.data.view(p_full.size()[0],-1))
        # if unity.size()[0] <= unity.size()[1]:
        #     weight_decay = group['weight_decay']
        #     rand_num = random.randint(1,101)
        #     if rand_num==1:
        #         unity = qr_retraction(unity)

        #     with torch.no_grad():
        #         g_full = torch.cat([p.grad.data.unsqueeze(0), p_grad_remaining], dim=0)
        #     g_full = g_full.view(g_full.size()[0],-1)
        #     lr = group['lr']
        #     param_state = self.state[p]
        #     if 'momentum_buffer' not in param_state:
        #         param_state['momentum_buffer'] = torch.zeros(g_full.t().size())
        #         if p.is_cuda:
        #             param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()
        #     V = param_state['momentum_buffer']
        #     V = momentum * V - g_full.t()
        #     MX = torch.mm(V, unity)
        #     XMX = torch.mm(unity, MX)
        #     XXMX = torch.mm(unity.t(), XMX)
        #     W_hat = MX - 0.5 * XXMX
        #     W = W_hat - W_hat.t()
        #     t = 0.5 * 2 / (matrix_norm_one(W) + 1e-8)
        #     alpha = min(t, lr)

        #     p_new = Cayley_loop(unity.t(), W, V, alpha)
        #     V_new = torch.mm(W, unity.t()) # n-by-p
        # #                     check_identity(p_new.t())
        #     p_full.data.copy_(p_new.view(p_full.size()))
        #     p.data.copy_(p_full[0])
        #     V.copy_(V_new)
        for group in router_param_groups:
            momentum = group["betas"][0]
            weight_decay = group["weight_decay"]
            if group["orth"]:
                for index, p in enumerate(group["params"]):
                    # assuming all vectors in p_remaining are orthogonal
                    p_remaining = group["other_params"][index]
                    d_p = p.grad.data
                    d_p = d_p - sum(
                        torch.dot(d_p, u) / torch.dot(u, u) * u for u in p_remaining
                    )
                    # print(f"similarity of gradient is {[torch.dot(d_p, u) / torch.dot(u, u) for u in p_remaining]}")
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(d_p)
                        d_p = buf

                    # print(f"similarity of momentum is {[torch.dot(d_p, u) / torch.dot(u, u) for u in p_remaining]}")
                    # print(f"similarity before update is {[torch.dot(p.data, u) / torch.dot(u, u) for u in p_remaining]}")
                    p.data.add_(d_p, alpha=-lr)
                    similarity_final = [
                        torch.dot(p.data, u) / torch.dot(u, u) for u in p_remaining
                    ]
                    # print(f"similarity after update is {[torch.dot(p.data, u) / torch.dot(u, u) for u in p_remaining]}")
                    if similarity_final[0] > 1e-3:
                        import ipdb

                        ipdb.set_trace()
            else:
                for index, p in enumerate(group["params"]):
                    p_remaining = group["other_params"][index]
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(d_p)
                        d_p = buf
                    p.data.add_(d_p, alpha=-lr)

    def step(self, closure=None):
        if not hasattr(self, "router_param_groups"):
            print(f"creating router_param_groups")
            self.router_param_groups, param_groups = [], []
            for group in self.param_groups:
                if "orth" in group:
                    self.router_param_groups.append(group)
                else:
                    param_groups.append(group)
            self.param_groups = param_groups
        super().step(closure)
        self.custom_step(self.router_param_groups, lr=self.param_groups[0]["lr"] * 10)


def get_other_router_params(model, param_name):
    other_router_params = []
    pattern = r"(.*?)(\d+)?$"
    prefix = re.match(pattern, param_name).group(1)
    for name, param in model.torch_model.named_parameters():
        if name.startswith(prefix) and name != param_name:
            other_router_params.append(param.float())
    return other_router_params


@gin.configurable(
    allowlist=[
        "block_index",
    ]
)
def param_group_fn_orth(param_name, block_index):
    if "expert_embeddings" not in param_name:
        return "other_trainable"
    else:
        pattern = r"encoder\.block\.(\d+)\.layer\.\d+"
        match = re.match(pattern, param_name)
        if match:
            extracted_index = int(match.group(1))
            if extracted_index <= block_index:
                return "router_trainable"
            else:
                return "router_orth_trainable"
        else:
            return "router_orth_trainable"


@gin.configurable(
    allowlist=[
        "optimizer_class",
        "learning_rate",
        "weight_decay",
        "scale_parameter",
        "relative_step",
        "param_group_fn",
    ]
)
def get_optimizer(
    model,
    optimizer_class,
    learning_rate,
    weight_decay=0.0,
    scale_parameter=False,
    relative_step=False,
    param_group_fn=lambda x: "all_trainable",
):
    """
    Args:
        model: a model object
        optimizer_class: optimizer class, one of "adam", "sgd", "adamw", "adafactor"
        learning_rate: learning rate
        weight_decay: weight decay for sgd, adamw and adafactor optimizer
        scale_parameter: scale parameter in adafactor optimizer
        relative_step: relative step in adafactor optimizer
        param_group_fn: function to group parameters, default to group all trainable parameters into one group
    """
    param_groups = defaultdict(lambda: {"params": []})
    for param_name, param in model.named_trainable_parameters().items():
        param_group_name = param_group_fn(param_name)
        if "router" in param_group_name:
            if "other_params" not in param_groups[param_group_name]:
                param_groups[param_group_name]["other_params"] = []
            param_groups[param_group_name]["other_params"].append(
                get_other_router_params(model, param_name)
            )
        if param_group_name == "router_orth_trainable":
            param_groups[param_group_name]["orth"] = True
        elif param_group_name == "router_trainable":
            param_groups[param_group_name]["orth"] = False
        param_groups[param_group_name]["params"].append(param)
    param_groups = param_groups.values()
    if optimizer_class == "adam":
        optimizer = optim.Adam(param_groups, lr=learning_rate)
    elif optimizer_class == "sgd":
        optimizer = optim.SGD(param_groups, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_class == "adamw":
        optimizer = optim.AdamW(
            param_groups, lr=learning_rate, weight_decay=weight_decay, eps=1e-8
        )
    elif optimizer_class == "adamw_orth":
        optimizer = AdamW_Orth(
            param_groups, lr=learning_rate, weight_decay=weight_decay, eps=1e-8
        )
    elif optimizer_class == "adafactor":
        optimizer = Adafactor(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=False,
        )
    else:
        raise ValueError("Invalid optimizer class %s" % optimizer_class)

    return optimizer


@gin.configurable(
    allowlist=[
        "scheduler_class",
        "gamma",
        "warmup_ratio",
        "num_warmup_steps",
    ]
)
def get_scheduler(
    optimizer,
    num_steps,
    scheduler_class,
    gamma=None,
    warmup_ratio=None,
    num_warmup_steps=None,
):
    """
    Args:
        optimizer: an optimizer
        scheduler_class: scheduler class, one of "constant", "polynomial_decay_with_warmup", "exponential_decay", "linear_decay_with_warmup", "cosine_annealing", "adafactor"
        num_steps: total number of steps
        warmup_ratio: ratio of warmup steps
        num_warmup_steps: number of warmup steps
    """
    if num_warmup_steps is None and warmup_ratio is not None:
        num_warmup_steps = int(num_steps * warmup_ratio)
    if num_warmup_steps is None:
        num_warmup_steps = 0
    if scheduler_class == "constant_with_warmup":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif scheduler_class == "defrost":
        return get_defrost_schedule(optimizer, num_warmup_steps)
    elif scheduler_class == "polynomial_decay_with_warmup":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_steps
        )
    elif scheduler_class == "exponential_decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_class == "linear_decay_with_warmup":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_steps)
    elif scheduler_class == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    elif scheduler_class == "adafactor":
        return AdafactorSchedule(optimizer, initial_lr=optimizer.defaults["lr"])
    else:
        raise ValueError("Invalid scheduler class %s" % scheduler_class)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which
    the learning rate increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_defrost_schedule(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Like get_constant_schedule_with_warmup, but with an addition num_warmup_steps of zero learning rate in the beginning.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return 0.0
        elif current_step < 2 * num_warmup_steps:
            return float(current_step - num_warmup_steps) / float(
                max(1, num_warmup_steps)
            )
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=1e-7,
    power=1.0,
    last_epoch=-1,
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay
    from the initial lr set in the optimizer to end lr defined by `lr_end`,
    after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is
    based on the original BERT implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    assert (
        lr_init > lr_end
    ), f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)
