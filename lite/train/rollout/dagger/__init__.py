"""DAgger rollout public API.

Slime resolves ``lite.train.rollout.dagger.{generate,
convert_samples_to_train_data,generate_rollout}`` by dotted string. Keep this
package facade aligned with ``scripts/train/run_dagger.sh``.

Exports are lazy so ``import lite.train.rollout.dagger.teacher`` does not load
the full rollout engine and its training-container-only dependencies.
"""

__all__ = ["generate", "convert_samples_to_train_data", "generate_rollout"]


def __getattr__(name: str):
    if name in __all__:
        from importlib import import_module

        rollout = import_module(".rollout", __name__)
        return getattr(rollout, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
