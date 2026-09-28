"""Expand the guard actor to ball-aware swing timing and optional arm corrections."""

import torch
from rsl_rl.models import RNNModel


class IncomingSwingActor(RNNModel):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        initial_weights,
        distribution_cfg=None,
        tempo_init_std=None,
    ):
        dimensions = (obs["policy"].shape[-1], output_dim)
        if obs_groups[obs_set] != ["policy"] or dimensions not in (
            (213, 13), (227, 27), (227, 15), (241, 7)
        ):
            raise ValueError(
                "incoming swing requires 213/13, 227/27, 227/15 or 241/7 observations/actions"
            )
        if output_dim in (7, 15) and initial_weights is not None:
            raise ValueError(
                "the strike actor starts fresh; guard weights belong to the controller"
            )
        if tempo_init_std is not None and (
            output_dim not in (7, 15)
            or not 0 < tempo_init_std < float("inf")
            or distribution_cfg is None
            or distribution_cfg.get("std_type") != "log"
        ):
            raise ValueError(
                "tempo noise requires a positive finite std and a fresh log-Gaussian strike actor"
            )
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=[32],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=distribution_cfg,
            rnn_type="lstm",
            rnn_hidden_dim=64,
            rnn_num_layers=1,
        )
        if output_dim in (7, 15):
            torch.nn.init.zeros_(self.mlp[2].weight)
            torch.nn.init.zeros_(self.mlp[2].bias)
            if tempo_init_std is not None:
                with torch.no_grad():
                    self.distribution.log_std_param[0] = torch.log(torch.tensor(tempo_init_std))
            return
        saved = torch.load(initial_weights, map_location="cpu", weights_only=True)
        expanded = self.state_dict()
        for key, value in saved.items():
            target = expanded[key]
            if target.shape == value.shape:
                target.copy_(value)
            elif key == "rnn.rnn.weight_ih_l0":
                target.zero_()
                target[:, :203] = value
            elif key in ("mlp.2.weight", "mlp.2.bias"):
                target.zero_()
                target[:12] = value
            elif key == "distribution.log_std_param":
                target[:12] = value
            else:
                raise ValueError(f"unsupported guard parameter expansion: {key}")
        self.load_state_dict(expanded, strict=True)
        self.reset()
