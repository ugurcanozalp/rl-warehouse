# rl-warehouse 🎮
This repository is my reinforcement learning framework based on PyTorch and TensorboardX, for fast prototyping single-agent reinforcement learning algorithms. For now, only Gymnasium environments are supported. This repository includes several rl algortihms, both developed myself and the others, in a clean and simple way. 

## Scripts
You can find scripts in scripts folder. These scripts can be manipulated as much as required, and can be converted to a jupyter notebook etc. Example usage for `STAC` training: 

```bash
python -m scripts.stac --env_name Hopper-v4 --autotune --target_entropy -1 --beta 0.75
```

## Off-policy Algorithms implemented
- [x] Stochastic Actor Critic (not published yet)
- [x] [Dropout Q Functions](https://arxiv.org/pdf/2110.02034)
- [x] [Soft Actor Critic](https://arxiv.org/abs/1812.05905v2)
- [x] [Tactical Optimism Pessimism for DRL](https://arxiv.org/abs/2102.03765)
- [x] [Tactical Optimism Pessimism for DRL (SAC variant)]()
- [x] [Truncated Quantile Critics](https://arxiv.org/abs/2005.04269)
- [x] [Randomized Ensembled Double Q-Learning](https://arxiv.org/abs/2101.05982)

## On-policy Algorithms implemented
- [x] [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

{"step": \d*, "eval_beta": 0.125}\n