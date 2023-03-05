# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is from SimMIM.
# (https://github.com/microsoft/SimMIM)
# ------------------------------------------------------------------------------

import json
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import build_optimizer
from mmcv.runner import get_dist_info


def get_num_layer_for_swin(var_name, num_max_layer, layers_per_stage):
    var_name = var_name.replace('encoder', 'backbone') if var_name.startswith('encoder') else var_name
    if var_name in ("backbone.cls_token", "backbone.mask_token",
                    "backbone.pos_embed", "backbone.absolute_pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.layers"):
        if var_name.split('.')[3] == "blocks":
            stage_id = int(var_name.split('.')[2])
            layer_id = int(var_name.split('.')[4]) \
                       + sum(layers_per_stage[:stage_id])
            return layer_id + 1
        elif var_name.split('.')[3] == "downsample":
            stage_id = int(var_name.split('.')[2])
            layer_id = sum(layers_per_stage[:stage_id + 1])
            return layer_id
    else:
        return num_max_layer - 1

@OPTIMIZER_BUILDERS.register_module()
class LDMOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        no_decay_names = self.paramwise_cfg.get('no_decay_names', [])
        print("Build LDMOptimizerConstructor")
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('absolute_pos_embed'):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

                for nd_name in no_decay_names:
                    if nd_name in name:
                        group_name = "no_decay"
                        this_weight_decay = 0.
                        break

            if 'unet' in name or 'cond_stage_model' in name:
                layer_id = 0
            else:
                layer_id = 1
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                if layer_id == 0:
                    scale = 0.01
                else:
                    scale = 1.0
                # scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"],
                    "lr_scale": parameter_groups[key]["lr_scale"],
                    "lr": parameter_groups[key]["lr"],
                    "weight_decay": parameter_groups[key]["weight_decay"],
                }

        params.extend(parameter_groups.values())

def build_optimizers(model, cfgs):
    """Build multiple optimizers from configs.

    If `cfgs` contains several dicts for optimizers, then a dict for each
    constructed optimizers will be returned.
    If `cfgs` only contains one optimizer config, the constructed optimizer
    itself will be returned.

    For example,

    1) Multiple optimizer configs:

    .. code-block:: python

        optimizer_cfg = dict(
            model1=dict(type='SGD', lr=lr),
            model2=dict(type='SGD', lr=lr))

    The return dict is
    ``dict('model1': torch.optim.Optimizer, 'model2': torch.optim.Optimizer)``

    2) Single optimizer config:

    .. code-block:: python

        optimizer_cfg = dict(type='SGD', lr=lr)

    The return is ``torch.optim.Optimizer``.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        cfgs (dict): The config dict of the optimizer.

    Returns:
        dict[:obj:`torch.optim.Optimizer`] | :obj:`torch.optim.Optimizer`:
            The initialized optimizers.
    """
    optimizers = {}
    if hasattr(model, 'module'):
        model = model.module
    # determine whether 'cfgs' has several dicts for optimizers
    if all(isinstance(v, dict) for v in cfgs.values()):
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            module = getattr(model, key)
            optimizers[key] = build_optimizer(module, cfg_)
        return optimizers

    return build_optimizer(model, cfgs)