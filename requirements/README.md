# Dependency policy

Foundation dependencies are declared and constrained in `pyproject.toml`.
There are no separate runtime requirements files yet.

The Qwen, PyTorch/CUDA, audio, evaluation, and Azure stacks will be added only
after their current official compatibility is qualified. That milestone must
record exact package and model revisions and explain every heavy dependency.
Application and training code must never install or upgrade packages.
