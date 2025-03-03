uv run scripts/compute_norm_stats.py --config-name tiny_pi0_fast_libero

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py tiny_pi0_fast_libero --exp-name=XXX --overwrite