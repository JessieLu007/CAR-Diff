# scripts/exp_config_presets.py
from configs.experiment_configs import make_default_experiment_config

def build_exp_cfg(random_seed: int):
    base_cfg = make_default_experiment_config()
    return base_cfg.with_overrides(
        data={"num_transform_method": "quantile", "cat_encoding_method": "ordinal"},
        pipeline={"random_seed": random_seed},
        model={
            "dim_hidden": 512, "num_timesteps": 200, "prediction_type": "eps",
            "focal_mode": "off", "use_marginal_align": False, "logit_adj_tau": 0,
            "gamma_base": 1, "focal_alpha_scale": 1.1, "gma_loss_weight": 0.00,
            "corr_weight": 0.05, "corr_t_ratio": 0.5, "posterior_switch_ratio": 1.0,
            "cat_temperature_cap": 1, "x0_clamp_min": -4, "x0_clamp_max": 4,
        },
        training={
            "task": "full_data", "objective": "fidelity", "batch_size": 512,
            "patience": 150, "epochs": 1000, "lr": 1e-3, "uncond_p": 0.1,
        },
        sampling={"temperature_fidelity": 1, "num_samples": None},
    )