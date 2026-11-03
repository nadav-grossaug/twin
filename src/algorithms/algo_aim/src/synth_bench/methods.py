"""Registry of the benchmarked Synthcity methods and their quick-budget params."""

from dataclasses import dataclass, field

from synth_bench.datasets import DatasetSpec

DP_EPSILON = 1.0  # common privacy budget for every DP method (a "moderate" budget)


@dataclass(frozen=True)
class MethodSpec:
    """A Synthcity plugin with its benchmark params and the reason for each choice."""

    name: str
    family: str
    dp: bool
    params: dict = field(default_factory=dict)
    rationale: dict[str, str] = field(default_factory=dict)


METHODS: dict[str, MethodSpec] = {
    m.name: m
    for m in [
        MethodSpec(
            name="marginal_distributions",
            family="baseline (independent marginals)",
            dp=False,
            rationale={"(none)": "No hyperparameters; samples every column independently."},
        ),
        MethodSpec(
            name="bayesian_network",
            family="bayesian network",
            dp=False,
            params={"struct_learning_search_method": "tree_search", "struct_learning_score": "k2"},
            rationale={
                "struct_learning_search_method": "Default; Chow-Liu tree search is fast and stable.",
                "struct_learning_score": "Default K2 score.",
            },
        ),
        MethodSpec(
            name="ctgan",
            family="GAN",
            dp=False,
            params={"n_iter": 300, "batch_size": 500},
            rationale={
                "n_iter": "Epochs; cut from 2000 to fit the CPU budget.",
                "batch_size": "Larger than the default 200 for fewer, cheaper steps per epoch.",
            },
        ),
        MethodSpec(
            name="tvae",
            family="VAE",
            dp=False,
            params={"n_iter": 300, "batch_size": 500},
            rationale={
                "n_iter": "Epochs; cut from 1000 to fit the CPU budget.",
                "batch_size": "Larger than the default 200 for fewer, cheaper steps per epoch.",
            },
        ),
        MethodSpec(
            name="nflow",
            family="normalizing flow",
            dp=False,
            params={"n_iter": 300, "batch_size": 500},
            rationale={
                "n_iter": "Epochs; cut from 1000 to fit the CPU budget.",
                "batch_size": "Larger than the default 200 for fewer, cheaper steps per epoch.",
            },
        ),
        MethodSpec(
            name="arf",
            family="tree-based (adversarial random forest)",
            dp=False,
            params={"num_trees": 30, "max_iters": 10, "verbose": False},
            rationale={
                "num_trees": "Default; ARF is cheap so no reduction is needed.",
                "max_iters": "Default adversarial rounds (early stopping is on).",
                "verbose": "Silence per-iteration prints.",
            },
        ),
        MethodSpec(
            name="ddpm",
            family="diffusion",
            dp=False,
            params={"n_iter": 1000, "batch_size": 1024, "num_timesteps": 1000},
            rationale={
                "n_iter": "Epochs; the TabDDPM MLP is small, the default is affordable.",
                "batch_size": "Default.",
                "num_timesteps": "Default diffusion length.",
                "is_classification": "Set from the dataset task: conditions on the target.",
            },
        ),
        MethodSpec(
            name="adsgan",
            family="GAN with identifiability penalty",
            dp=False,
            params={"n_iter": 300, "batch_size": 500},
            rationale={
                "n_iter": "Epochs; cut from 10000 (same budget as ctgan).",
                "batch_size": "Same as ctgan for comparability.",
            },
        ),
        MethodSpec(
            name="privbayes",
            family="DP bayesian network",
            dp=True,
            params={"epsilon": DP_EPSILON},
            rationale={"epsilon": "Common DP budget."},
        ),
        MethodSpec(
            name="dpgan",
            family="DP GAN (DP-SGD)",
            dp=True,
            params={"n_iter": 300, "batch_size": 500, "epsilon": DP_EPSILON},
            rationale={
                "n_iter": "Epochs; same budget as ctgan.",
                "batch_size": "Same as ctgan for comparability.",
                "epsilon": "Common DP budget; delta defaults to 1/n.",
            },
        ),
        MethodSpec(
            name="pategan",
            family="DP GAN (PATE)",
            dp=True,
            params={"n_iter": 200, "n_teachers": 10, "epsilon": DP_EPSILON},
            rationale={
                "n_iter": "Default outer iterations.",
                "n_teachers": "Default teacher ensemble size.",
                "epsilon": "Common DP budget; delta defaults to 1/n.",
            },
        ),
        MethodSpec(
            name="aim",
            family="DP marginal-based (AIM)",
            dp=True,
            params={"epsilon": DP_EPSILON, "delta": 1e-9, "degree": 2},
            rationale={
                "epsilon": "Common DP budget.",
                "delta": "Default.",
                "degree": "Default: 2-way marginals in the workload.",
            },
        ),
    ]
}


def list_methods() -> list[str]:
    """Names of all registered methods, in benchmark order."""
    return list(METHODS)


def get_method(name: str) -> MethodSpec:
    """Return the method spec registered under ``name``."""
    if name not in METHODS:
        raise KeyError(f"Unknown method {name!r}; available: {list_methods()}")
    return METHODS[name]


def get_params(method: str, dataset: DatasetSpec) -> dict:
    """Plugin params for ``method`` on ``dataset``: defaults, task-derived, then overrides."""
    params = dict(get_method(method).params)
    if method == "ddpm":
        params["is_classification"] = dataset.task_type == "classification"
    params.update(dataset.method_overrides.get(method, {}))
    return params
