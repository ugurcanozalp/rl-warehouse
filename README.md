# rl-warehouse 🎮
This repository is my reinforcement learning framework based on PyTorch and TensorboardX, for fast prototyping reinforcement learning algorithms. It includes several rl algortihms, both developed myself and the others, in a clean and simple way. 

## Scripts
You can find scripts in scripts folder. These scripts can be manipulated as much as required, and can be converted to a jupyter notebook etc. Example usage for STAC training: 

```bash
python -m scripts.stac --env_name Hopper-v4 --autotune --target_entropy -1 --beta 0.75
```

## Algorithms implemented
- [x] Deep Bayesian Actor Critic (not published yet)
- [x] [Soft Actor Critic](https://arxiv.org/abs/1812.05905v2)
- [x] [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
