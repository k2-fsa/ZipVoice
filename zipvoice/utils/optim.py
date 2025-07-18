# Copyright      2022  Xiaomi Corp.        (authors: Daniel Povey)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.optim import Optimizer


class BatchedOptimizer(Optimizer):
    """
    This class adds to class Optimizer the capability to optimize parameters in batches:
    it will stack the parameters and their grads for you so the optimizer can work
    on tensors with an extra leading dimension.  This is intended for speed with GPUs,
    as it reduces the number of kernels launched in the optimizer.

    Args:
      params:
    """

    def __init__(self, params, defaults):
        super(BatchedOptimizer, self).__init__(params, defaults)

    @contextlib.contextmanager
    def batched_params(self, param_group, group_params_names):
        """
        This function returns (technically, yields) a list of
          of tuples (p, state), where
        p is a `fake` parameter that is stacked (over axis 0) from real parameters
        that share the same shape, and its gradient is also stacked;
        `state` is the state corresponding to this batch of parameters
        (it will be physically located in the "state" for one of the real
        parameters, the last one that has any particular shape and dtype).

        This function is decorated as a context manager so that it can
        write parameters back to their "real" locations.

        The idea is, instead of doing:
        <code>
          for p in group["params"]:
             state = self.state[p]
             ...
        </code>
        you can do:
        <code>
          with self.batched_params(group["params"]) as batches:
             for p, state, p_names in batches:
                 ...
        </code>

        Args:
          group: a parameter group, which is a list of parameters; should be
                one of self.param_groups.
          group_params_names: name for each parameter in group,
                which is List[str].
        """
        batches = defaultdict(
            list
        )  # `batches` maps from tuple (dtype_as_str,*shape) to list of nn.Parameter
        batches_names = defaultdict(
            list
        )  # `batches` maps from tuple (dtype_as_str,*shape) to list of str

        assert len(param_group) == len(group_params_names)
        for p, named_p in zip(param_group, group_params_names):
            key = (str(p.dtype), *p.shape)
            batches[key].append(p)
            batches_names[key].append(named_p)

        batches_names_keys = list(batches_names.keys())
        sorted_idx = sorted(
            range(len(batches_names)), key=lambda i: batches_names_keys[i]
        )
        batches_names = [batches_names[batches_names_keys[idx]] for idx in sorted_idx]
        batches = [batches[batches_names_keys[idx]] for idx in sorted_idx]

        stacked_params_dict = dict()

        # turn batches into a list, in deterministic order.
        # tuples will contain tuples of (stacked_param, state, stacked_params_names),
        # one for each batch in `batches`.
        tuples = []

        for batch, batch_names in zip(batches, batches_names):
            p = batch[0]
            # we arbitrarily store the state in the
            # state corresponding to the 1st parameter in the
            # group.  class Optimizer will take care of saving/loading state.
            state = self.state[p]
            p_stacked = torch.stack(batch)
            grad = torch.stack(
                [torch.zeros_like(p) if p.grad is None else p.grad for p in batch]
            )
            p_stacked.grad = grad
            stacked_params_dict[key] = p_stacked
            tuples.append((p_stacked, state, batch_names))

        yield tuples  # <-- calling code will do the actual optimization here!

        for (stacked_params, _state, _names), batch in zip(tuples, batches):
            for i, p in enumerate(batch):  # batch is list of Parameter
                p.copy_(stacked_params[i])


def basic_step(group, p, state, grad):
    # computes basic Adam update using beta2 (dividing by gradient stddev) only.  no
    # momentum yet.
    lr = group["lr"]
    if p.numel() == p.shape[0]:
        lr = lr * group["scalar_lr_scale"]
    beta2 = group["betas"][1]
    eps = group["eps"]
    # p shape: (batch_size,) or (batch_size, 1, [1,..])
    try:
        exp_avg_sq = state[
            "exp_avg_sq"
        ]  # shape: (batch_size,) or (batch_size, 1, [1,..])
    except KeyError:
        exp_avg_sq = torch.zeros(*p.shape, device=p.device, dtype=torch.float)
        state["exp_avg_sq"] = exp_avg_sq

    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # bias_correction2 is like in Adam.
    # slower update at the start will help stability anyway.
    bias_correction2 = 1 - beta2 ** (state["step"] + 1)
    if bias_correction2 < 0.99:
        # note: not in-place.
        exp_avg_sq = exp_avg_sq * (1.0 / bias_correction2)
    denom = exp_avg_sq.sqrt().add_(eps)

    return -lr * grad / denom


def scaling_step(group, p, state, grad):
    delta = basic_step(group, p, state, grad)
    if p.numel() == p.shape[0]:
        return delta
    # there is no scaling for scalar parameters.
    # (p.shape[0] is the batch of parameters.)

    step = state["step"]
    size_update_period = group["size_update_period"]

    try:
        param_rms = state["param_rms"]
        scale_grads = state["scale_grads"]
        scale_exp_avg_sq = state["scale_exp_avg_sq"]
    except KeyError:
        # we know p.ndim > 1 because we'd have returned above if not, so don't worry
        # about the speial case of dim=[] that pytorch treats inconsistently.
        param_rms = (p**2).mean(dim=list(range(1, p.ndim)), keepdim=True).sqrt()
        param_rms = param_rms.to(torch.float)
        scale_exp_avg_sq = torch.zeros_like(param_rms)
        scale_grads = torch.zeros(
            size_update_period,
            *param_rms.shape,
            dtype=torch.float,
            device=p.device,
        )
        state["param_rms"] = param_rms
        state["scale_grads"] = scale_grads
        state["scale_exp_avg_sq"] = scale_exp_avg_sq

    # on every step, update the gradient w.r.t. the scale of the parameter, we
    # store these as a batch and periodically update the size (for speed only, to
    # avoid too many operations).
    scale_grads[step % size_update_period] = (p * grad).sum(
        dim=list(range(1, p.ndim)), keepdim=True
    )

    # periodically recompute the value of param_rms.
    if step % size_update_period == size_update_period - 1:
        param_rms.copy_((p**2).mean(dim=list(range(1, p.ndim)), keepdim=True).sqrt())

    param_min_rms = group["param_min_rms"]

    # scale the step size by param_rms.  This is the most important "scaling" part of
    # ScaledAdam
    delta *= param_rms.clamp(min=param_min_rms)

    if step % size_update_period == size_update_period - 1 and step > 0:
        # This block updates the size of parameter by adding a step ("delta") value in
        # the direction of either shrinking or growing it.
        beta2 = group["betas"][1]
        size_lr = group["lr"] * group["scalar_lr_scale"]
        param_max_rms = group["param_max_rms"]
        eps = group["eps"]
        # correct beta2 for the size update period: we will have
        # faster decay at this level.
        beta2_corr = beta2**size_update_period
        scale_exp_avg_sq.mul_(beta2_corr).add_(
            (scale_grads**2).mean(dim=0),  # mean over dim `size_update_period`
            alpha=1 - beta2_corr,
        )  # shape is (batch_size, 1, 1, ...)

        # The 1st time we reach here is when size_step == 1.
        size_step = (step + 1) // size_update_period
        bias_correction2 = 1 - beta2_corr**size_step

        denom = scale_exp_avg_sq.sqrt() + eps

        scale_step = (
            -size_lr * (bias_correction2**0.5) * scale_grads.sum(dim=0) / denom
        )

        is_too_small = param_rms < param_min_rms

        # when the param gets too small, just don't shrink it any further.
        scale_step.masked_fill_(is_too_small, 0.0)

        # The following may help prevent instability: don't allow the scale step to be
        # too large in either direction.
        scale_step.clamp_(min=-0.1, max=0.1)

        # and ensure the parameter rms after update never exceeds param_max_rms.
        # We have to look at the trained model for parameters at or around the
        # param_max_rms, because sometimes they can indicate a problem with the
        # topology or settings.
        scale_step = torch.minimum(scale_step, (param_max_rms - param_rms) / param_rms)

        delta.add_(p * scale_step)

    return delta


def momentum_step(group, p, state, grad):
    delta = scaling_step(group, p, state, grad)
    beta1 = group["betas"][0]
    try:
        stored_delta = state["delta"]
    except KeyError:
        stored_delta = torch.zeros(*p.shape, device=p.device, dtype=torch.float)
        state["delta"] = stored_delta
    stored_delta.mul_(beta1)
    stored_delta.add_(delta, alpha=(1 - beta1))
    # we don't bother doing the "bias correction" part of Adam for beta1 because this is
    # just an edge effect that affects the first 10 or so batches; and the effect of not
    # doing it is just to do a slower update for the first few batches, which will help
    # stability.
    return stored_delta


class ScaledAdam(BatchedOptimizer):
    """
     Implements 'Scaled Adam', a variant of Adam where we scale each parameter's update
     proportional to the norm of that parameter; and also learn the scale of the
     parameter, in log space, subject to upper and lower limits (as if we had factored
     each parameter as param = underlying_param * log_scale.exp())


     Args:
          params: The parameters or param_groups to optimize (like other Optimizer
                    subclasses) Unlike common optimizers, which accept
                    model.parameters() or groups of parameters(), this optimizer
                    could accept model.named_parameters() or groups of
                    named_parameters(). See comments of function
                    _get_names_of_parameters for its 4 possible cases.
              lr:  The learning rate.  We will typically use a learning rate schedule
                    that starts at 0.03 and decreases over time, i.e. much higher
                    than other common optimizers.
     clipping_scale: (e.g. 2.0)
                   A scale for gradient-clipping: if specified, the normalized gradients
                   over the whole model will be clipped to have 2-norm equal to
                   `clipping_scale` times the median 2-norm over the most recent period
                   of `clipping_update_period` minibatches.  By "normalized gradients",
                   we mean after multiplying by the rms parameter value for this tensor
                   [for non-scalars]; this is appropriate because our update is scaled
                   by this quantity.
            betas: beta1,beta2 are momentum constants for regular momentum, and moving
                    sum-sq grad. Must satisfy 0 < beta <= beta2 < 1.
     scalar_lr_scale: A scaling factor on the learning rate, that we use to update the
                    scale of each parameter tensor and scalar parameters of the mode..
                    If each parameter were decomposed as p * p_scale.exp(),
                    where (p**2).mean().sqrt() == 1.0, scalar_lr_scale would be a the
                    scaling factor on the learning rate of p_scale.
              eps:  A general-purpose epsilon to prevent division by zero
    param_min_rms: Minimum root-mean-square value of parameter tensor, for purposes of
                   learning the scale on the parameters (we'll constrain the rms of
                   each non-scalar parameter tensor to be >= this value)
    param_max_rms: Maximum root-mean-square value of parameter tensor, for purposes of
                   learning the scale on the parameters (we'll constrain the rms of
                   each non-scalar parameter tensor to be <= this value)
       scalar_max: Maximum absolute value for scalar parameters (applicable if your
                   model has any parameters with numel() == 1).
    size_update_period: The periodicity, in steps, with which we update the size (scale)
                   of the parameter tensor.  This is provided to save a little time
                   in the update.
     clipping_update_period: if clipping_scale is specified, this is the period
    """

    def __init__(
        self,
        params,
        lr=3e-02,
        clipping_scale=None,
        betas=(0.9, 0.98),
        scalar_lr_scale=0.1,
        eps=1.0e-08,
        param_min_rms=1.0e-05,
        param_max_rms=3.0,
        scalar_max=10.0,
        size_update_period=4,
        clipping_update_period=100,
    ):

        defaults = dict(
            lr=lr,
            clipping_scale=clipping_scale,
            betas=betas,
            scalar_lr_scale=scalar_lr_scale,
            eps=eps,
            param_min_rms=param_min_rms,
            param_max_rms=param_max_rms,
            scalar_max=scalar_max,
            size_update_period=size_update_period,
            clipping_update_period=clipping_update_period,
        )

        # If params only contains parameters or group of parameters,
        # i.e when parameter names are not given,
        # this flag will be set to False in funciton _get_names_of_parameters.
        self.show_dominant_parameters = True
        param_groups, parameters_names = self._get_names_of_parameters(params)
        super(ScaledAdam, self).__init__(param_groups, defaults)
        assert len(self.param_groups) == len(parameters_names)
        self.parameters_names = parameters_names

    def _get_names_of_parameters(
        self, params_or_named_params
    ) -> Tuple[List[Dict], List[List[str]]]:
        """
        Args:
          params_or_named_params: according to the way ScaledAdam is initialized
            in train.py, this argument could be one of following 4 cases,
            case 1, a generator of parameter, e.g.:
              optimizer = ScaledAdam(model.parameters(), lr=params.base_lr,
                clipping_scale=3.0)

            case 2, a list of parameter groups with different config, e.g.:
              model_param_groups = [
                      {'params': model.encoder.parameters(), 'lr': 0.05},
                      {'params': model.decoder.parameters(), 'lr': 0.01},
                      {'params': model.joiner.parameters(), 'lr': 0.03},
                      ]
              optimizer = ScaledAdam(model_param_groups, lr=params.base_lr,
                clipping_scale=3.0)

            case 3, a generator of named_parameter, e.g.:
              optimizer = ScaledAdam(model.named_parameters(), lr=params.base_lr,
                clipping_scale=3.0)

            case 4, a list of named_parameter groups with different config, e.g.:
              model_named_param_groups = [
                      {'named_params': model.encoder.named_parameters(), 'lr': 0.05},
                      {'named_params': model.decoder.named_parameters(), 'lr': 0.01},
                      {'named_params': model.joiner.named_parameters(), 'lr': 0.03},
                      ]
              optimizer = ScaledAdam(model_named_param_groups, lr=params.base_lr,
                clipping_scale=3.0)

          For case 1 and case 2, input params is used to initialize the underlying
            torch.optimizer.
          For case 3 and case 4, firstly, names and params are extracted from input
            named_params, then, these extracted params are used to initialize the
            underlying torch.optimizer, and these extracted names are mainly used by
            function `_show_gradient_dominating_parameter`

        Returns:
          Returns a tuple containing 2 elements:
            - `param_groups` with type List[Dict], each Dict element is a parameter
                group. An example of `param_groups` could be:
              [
                  {'params': `one iterable of Parameter`, 'lr': 0.05},
                  {'params': `another iterable of Parameter`, 'lr': 0.08},
                  {'params': `a third iterable of Parameter`, 'lr': 0.1},
              ]
            - `param_gruops_names` with type List[List[str]],
               each `List[str]` is for a group['params'] in param_groups,
               and each `str` is the name of a parameter.
               A dummy name "foo" is related to each parameter,
               if input are params without names, i.e. case 1 or case 2.
        """
        # variable naming convention in this function:
        #   p is short for param.
        #   np is short for named_param.
        #   p_or_np is short for param_or_named_param.
        #   cur is short for current.
        #   group is a dict,
        #   e.g. {'params': iterable of parameter, 'lr': 0.05, other fields}.
        #   groups is a List[group]

        iterable_or_groups = list(params_or_named_params)
        if len(iterable_or_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        # The first value of returned tuple.  A list of dicts containing at
        # least 'params' as a key.
        param_groups = []

        # The second value of returned tuple,
        # a List[List[str]], each sub-List is for a group.
        param_groups_names = []

        if not isinstance(iterable_or_groups[0], dict):
            # case 1 or case 3,
            # the input is an iterable of parameter or named parameter.
            param_iterable_cur_group = []
            param_names_cur_group = []
            for p_or_np in iterable_or_groups:
                if isinstance(p_or_np, tuple):
                    # case 3
                    name, param = p_or_np
                else:
                    # case 1
                    assert isinstance(p_or_np, torch.Tensor)
                    param = p_or_np
                    # Assign a dummy name as a placeholder
                    name = "foo"
                    self.show_dominant_parameters = False
                param_iterable_cur_group.append(param)
                param_names_cur_group.append(name)
            param_groups.append({"params": param_iterable_cur_group})
            param_groups_names.append(param_names_cur_group)
        else:
            # case 2 or case 4
            # the input is groups of parameter or named parameter.
            for cur_group in iterable_or_groups:
                if "named_params" in cur_group:
                    name_list = [x[0] for x in cur_group["named_params"]]
                    p_list = [x[1] for x in cur_group["named_params"]]
                    del cur_group["named_params"]
                    cur_group["params"] = p_list
                else:
                    assert "params" in cur_group
                    name_list = ["foo" for _ in cur_group["params"]]
                param_groups.append(cur_group)
                param_groups_names.append(name_list)

        return param_groups, param_groups_names

    def __setstate__(self, state):
        super(ScaledAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, group_params_names in zip(self.param_groups, self.parameters_names):

            with self.batched_params(group["params"], group_params_names) as batches:

                # batches is list of pairs (stacked_param, state).  stacked_param is
                # like a regular parameter, and will have a .grad, but the 1st dim
                # corresponds to a stacking dim, it is not a real dim.

                if (
                    len(batches[0][1]) == 0
                ):  # if len(first state) == 0: not yet initialized
                    clipping_scale = 1
                else:
                    clipping_scale = self._get_clipping_scale(group, batches)

                for p, state, _ in batches:
                    # Perform optimization step.
                    # grad is not going to be None, we handled that when creating the
                    # batches.
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "ScaledAdam optimizer does not support sparse gradients"
                        )

                    try:
                        cur_step = state["step"]
                    except KeyError:
                        state["step"] = 0
                        cur_step = 0

                    grad = (
                        p.grad if clipping_scale == 1.0 else p.grad.mul_(clipping_scale)
                    )
                    p += momentum_step(group, p.detach(), state, grad)

                    if p.numel() == p.shape[0]:  # scalar parameter
                        scalar_max = group["scalar_max"]
                        p.clamp_(min=-scalar_max, max=scalar_max)

                    state["step"] = cur_step + 1

        return loss

    def _get_clipping_scale(
        self, group: dict, tuples: List[Tuple[Tensor, dict, List[str]]]
    ) -> float:
        """
        Returns a scalar factor <= 1.0 that dictates gradient clipping, i.e. we will
        scale the gradients by this amount before applying the rest of the update.

        Args:
           group: the parameter group, an item in self.param_groups
           tuples: a list of tuples of (param, state, param_names)
                where param is a batched set of parameters,
                with a .grad (1st dim is batch dim)
                and state is the state-dict where optimization parameters are kept.
                param_names is a List[str] while each str is name for a parameter
                in batched set of parameters "param".
        """
        assert len(tuples) >= 1
        clipping_scale = group["clipping_scale"]
        (first_p, first_state, _) = tuples[0]
        step = first_state["step"]
        if clipping_scale is None or step == 0:
            # no clipping.  return early on step == 0 because the other
            # parameters' state won't have been initialized yet.
            return 1.0
        clipping_update_period = group["clipping_update_period"]
        scalar_lr_scale = group["scalar_lr_scale"]

        tot_sumsq = torch.tensor(0.0, device=first_p.device)
        for p, state, param_names in tuples:
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError(
                    "ScaledAdam optimizer does not support sparse gradients"
                )
            if p.numel() == p.shape[0]:  # a batch of scalars
                tot_sumsq += (grad**2).sum() * (
                    scalar_lr_scale**2
                )  # sum() to change shape [1] to []
            else:
                tot_sumsq += ((grad * state["param_rms"]) ** 2).sum()

        tot_norm = tot_sumsq.sqrt()
        if "model_norms" not in first_state:
            first_state["model_norms"] = torch.zeros(
                clipping_update_period, device=p.device
            )
        first_state["model_norms"][step % clipping_update_period] = tot_norm

        irregular_estimate_steps = [
            i for i in [10, 20, 40] if i < clipping_update_period
        ]
        if step % clipping_update_period == 0 or step in irregular_estimate_steps:
            # Print some stats.
            # We don't reach here if step == 0 because we would have returned
            # above.
            sorted_norms = first_state["model_norms"].sort()[0].to("cpu")
            if step in irregular_estimate_steps:
                sorted_norms = sorted_norms[-step:]
            num_norms = sorted_norms.numel()
            quartiles = []
            for n in range(0, 5):
                index = min(num_norms - 1, (num_norms // 4) * n)
                quartiles.append(sorted_norms[index].item())

            median = quartiles[2]
            if median - median != 0:
                raise RuntimeError("Too many grads were not finite")
            threshold = clipping_scale * median
            if step in irregular_estimate_steps:
                # use larger thresholds on first few steps of estimating threshold,
                # as norm may be changing rapidly.
                threshold = threshold * 2.0
            first_state["model_norm_threshold"] = threshold
            percent_clipped = (
                first_state["num_clipped"] * 100.0 / num_norms
                if "num_clipped" in first_state
                else 0.0
            )
            first_state["num_clipped"] = 0
            quartiles = " ".join(["%.3e" % x for x in quartiles])
            logging.warning(
                f"Clipping_scale={clipping_scale}, grad-norm quartiles {quartiles}, "
                f"threshold={threshold:.3e}, percent-clipped={percent_clipped:.1f}"
            )

        try:
            model_norm_threshold = first_state["model_norm_threshold"]
        except KeyError:
            return 1.0  # threshold has not yet been set.

        ans = min(1.0, (model_norm_threshold / (tot_norm + 1.0e-20)).item())
        if ans != ans:  # e.g. ans is nan
            ans = 0.0
        if ans < 1.0:
            first_state["num_clipped"] += 1
        if ans < 0.5:
            logging.debug(
                f"Scaling gradients by {ans}, "
                f"model_norm_threshold={model_norm_threshold}"
            )
            if self.show_dominant_parameters:
                assert p.shape[0] == len(param_names)
                self._show_gradient_dominating_parameter(
                    tuples, tot_sumsq, group["scalar_lr_scale"]
                )
                self._show_param_with_unusual_grad(tuples)

        if ans == 0.0:
            for p, state, param_names in tuples:
                p.grad.zero_()  # get rid of infinity()

        return ans

    def _show_param_with_unusual_grad(
        self,
        tuples: List[Tuple[Tensor, dict, List[str]]],
    ):
        """
        Print information about parameter which has the largest ratio of
        grad-on-this-batch divided by normal grad size.
           tuples: a list of tuples of (param, state, param_names)
                where param is a batched set of parameters,
                with a .grad (1st dim is batch dim)
                and state is the state-dict where optimization parameters are kept.
                param_names is a List[str] while each str is name for a parameter
                in batched set of parameters "param".
        """
        # ratios_names is a list of 3-tuples: (grad_ratio, param_name, tensor)
        ratios_names = []
        for p, state, batch_param_names in tuples:
            dims = list(range(1, p.ndim))

            def mean(x):
                # workaround for bad interface of torch's "mean" for when dims is the
                # empty list.
                if len(dims) > 0:
                    return x.mean(dim=dims)
                else:
                    return x

            grad_ratio = (
                (mean(p.grad**2) / state["exp_avg_sq"].mean(dim=dims))
                .sqrt()
                .to("cpu")
            )

            ratios_names += zip(
                grad_ratio.tolist(), batch_param_names, p.grad.unbind(dim=0)
            )

        ratios_names = sorted(ratios_names, reverse=True)
        ratios_names = ratios_names[:10]
        ratios_names = [
            (ratio, name, largest_index(tensor))
            for (ratio, name, tensor) in ratios_names
        ]

        logging.debug(
            f"Parameters with most larger-than-usual grads, with ratios, "
            f"are: {ratios_names}"
        )

    def _show_gradient_dominating_parameter(
        self,
        tuples: List[Tuple[Tensor, dict, List[str]]],
        tot_sumsq: Tensor,
        scalar_lr_scale: float,
    ):
        """
        Show information of parameter which dominates tot_sumsq.

        Args:
           tuples: a list of tuples of (param, state, param_names)
                where param is a batched set of parameters,
                with a .grad (1st dim is batch dim)
                and state is the state-dict where optimization parameters are kept.
                param_names is a List[str] while each str is name for a parameter
                in batched set of parameters "param".
            tot_sumsq: sumsq of all parameters. Though it's could be calculated
                from tuples, we still pass it to save some time.
        """
        all_sumsq_orig = {}
        for p, state, batch_param_names in tuples:
            # p is a stacked batch parameters.
            batch_grad = p.grad
            if p.numel() == p.shape[0]:  # a batch of scalars
                # Dummy values used by following `zip` statement.
                batch_rms_orig = torch.full(
                    p.shape, scalar_lr_scale, device=batch_grad.device
                )
            else:
                batch_rms_orig = state["param_rms"]
            batch_sumsq_orig = (batch_grad * batch_rms_orig) ** 2
            if batch_grad.ndim > 1:
                # need to guard it with if-statement because sum() sums over
                # all dims if dim == ().
                batch_sumsq_orig = batch_sumsq_orig.sum(
                    dim=list(range(1, batch_grad.ndim))
                )
            for name, sumsq_orig, rms, grad in zip(
                batch_param_names, batch_sumsq_orig, batch_rms_orig, batch_grad
            ):

                proportion_orig = sumsq_orig / tot_sumsq
                all_sumsq_orig[name] = (proportion_orig, sumsq_orig, rms, grad)

        sorted_by_proportion = {
            k: v
            for k, v in sorted(
                all_sumsq_orig.items(),
                key=lambda item: item[1][0],
                reverse=True,
            )
        }
        dominant_param_name = next(iter(sorted_by_proportion))
        (
            dominant_proportion,
            dominant_sumsq,
            dominant_rms,
            dominant_grad,
        ) = sorted_by_proportion[dominant_param_name]
        logging.debug(
            f"Parameter dominating tot_sumsq {dominant_param_name}"
            f" with proportion {dominant_proportion:.2f},"
            f" where dominant_sumsq=(grad_sumsq*orig_rms_sq)"
            f"={dominant_sumsq:.3e},"
            f" grad_sumsq={(dominant_grad**2).sum():.3e},"
            f" orig_rms_sq={(dominant_rms**2).item():.3e}"
        )


def largest_index(x: Tensor):
    x = x.contiguous()
    argmax = x.abs().argmax().item()
    return [(argmax // x.stride(i)) % x.size(i) for i in range(x.ndim)]


def _test_scaled_adam(hidden_dim: int):
    import timeit

    from zipvoice.models.modules.scaling import ScaledLinear
    from zipvoice.utils.lr_scheduler import Eden

    E = 100
    B = 4
    T = 2
    logging.info("in test_eve_cain")
    # device = torch.device('cuda')
    device = torch.device("cpu")
    dtype = torch.float32

    fix_random_seed(42)
    # these input_magnitudes and output_magnitudes are to test that
    # Abel is working as we expect and is able to adjust scales of
    # different dims differently.
    input_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()
    output_magnitudes = (1.0 * torch.randn(E, dtype=dtype, device=device)).exp()

    fix_random_seed(42)
    Linear = ScaledLinear

    m = torch.nn.Sequential(
        Linear(E, hidden_dim),
        torch.nn.PReLU(),
        Linear(hidden_dim, hidden_dim),
        torch.nn.PReLU(),
        Linear(hidden_dim, E),
    ).to(device)

    train_pairs = [
        (
            100.0 * torch.randn(B, T, E, device=device, dtype=dtype) * input_magnitudes,
            torch.randn(B, T, E, device=device, dtype=dtype) * output_magnitudes,
        )
        for _ in range(20)
    ]
    optim = ScaledAdam(m.named_parameters(), lr=0.03, clipping_scale=2.0)
    scheduler = Eden(optim, lr_batches=200, lr_epochs=5, verbose=False)

    start = timeit.default_timer()
    avg_loss = 0.0
    for epoch in range(180):
        scheduler.step_epoch()
        # if epoch == 100 and iter in [2,3]:
        #    optim.reset_speedup()  # check it doesn't crash.

        # if epoch == 130:
        #    opts = diagnostics.TensorDiagnosticOptions(
        #        512
        #    )  # allow 4 megabytes per sub-module
        #    diagnostic = diagnostics.attach_diagnostics(m, opts)

        for n, (x, y) in enumerate(train_pairs):
            y_out = m(x)
            loss = ((y_out - y) ** 2).mean() * 100.0
            if epoch == 0 and n == 0:
                avg_loss = loss.item()
            else:
                avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
            if n == 0 and epoch % 5 == 0:
                # norm1 = '%.2e' % (m[0].weight**2).mean().sqrt().item()
                # norm1b = '%.2e' % (m[0].bias**2).mean().sqrt().item()
                # norm2 = '%.2e' % (m[2].weight**2).mean().sqrt().item()
                # norm2b = '%.2e' % (m[2].bias**2).mean().sqrt().item()
                # scale1 = '%.2e' % (m[0].weight_scale.exp().item())
                # scale1b = '%.2e' % (m[0].bias_scale.exp().item())
                # scale2 = '%.2e' % (m[2].weight_scale.exp().item())
                # scale2b = '%.2e' % (m[2].bias_scale.exp().item())
                lr = scheduler.get_last_lr()[0]
                logging.info(
                    f"Iter {iter}, epoch {epoch}, batch {n}, "
                    f"avg_loss {avg_loss:.4g}, lr={lr:.4e}"
                )  # , norms={norm1,norm1b,norm2,norm2b}")
                # scales={scale1,scale1b,scale2,scale2b}
            loss.log().backward()
            optim.step()
            optim.zero_grad()
            scheduler.step_batch()

        # diagnostic.print_diagnostics()

        stop = timeit.default_timer()
        logging.info(f"Iter={iter}, Time taken: {stop - start}")

        logging.info(f"last lr = {scheduler.get_last_lr()}")
        # logging.info("state dict = ", scheduler.state_dict())
        # logging.info("optim state_dict = ", optim.state_dict())
        logging.info(f"input_magnitudes = {input_magnitudes}")
        logging.info(f"output_magnitudes = {output_magnitudes}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.getLogger().setLevel(logging.INFO)
    import subprocess

    s = subprocess.check_output(
        "git status -uno .; git log -1; git diff HEAD .", shell=True
    )
    logging.info(s)
    import sys

    if len(sys.argv) > 1:
        hidden_dim = int(sys.argv[1])
    else:
        hidden_dim = 200

    _test_scaled_adam(hidden_dim)
