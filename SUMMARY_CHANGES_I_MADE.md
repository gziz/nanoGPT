# SUMMARY OF CHANGES I MADE

1. **`064224d` — First commit (notes)**: Added `dictionary.md` with tensor dimension conventions (B, T, C, etc.), a `following_video.ipynb` notebook exploring bag-of-words averaging with PyTorch, a custom training script `train_my.py`, and added docstrings/comments to `model.py` (MLP class and CausalSelfAttention).

2. **`8a24476` — Move config outside of train.py**: Major project restructuring — moved all source files into `src/` and notebooks into `notebooks/`. Extracted training configuration from `train_my.py` into a standalone `src/config.py`. The original `train.py` was preserved as `src/_train.py` and `train_my.py` became the new `src/train.py` (simplified).

3. **`3691d09` — Move config fixes**: Expanded `src/config.py` with more configuration options, fixed imports and references in `src/train.py` to work with the extracted config, and added a `makefile` with training/sampling commands.

4. **`f80eee3` — Modularize sample**: Refactored `src/sample.py` into a cleaner modular structure, created `src/sample_config.py` for sampling configuration, renamed `config.py` → `train_config.py`, reorganized `src/train.py` with better structure, and added `src/utils.py`.

5. **`b04f5a7` — Various changes**: Deleted the legacy `src/_train.py` (338 lines removed), cleaned up `model.py`, `sample.py`, `train.py`, and `train_config.py`, and added type hints to `configurator.py`.

6. **`d12052b` — Nit lint**: Minor one-line formatting fix in `src/model.py`.

**Overall arc**: The commits progressively restructure the nanoGPT codebase — organizing files into `src/` and `notebooks/`, extracting config into dedicated modules, modularizing the sampling script, cleaning up legacy code, and finishing with a small lint fix.
