# EMU: Efficient Episodic Memory Utilization of Cooperative Multi-agent Reinforcement Learning

# Note
This codebase accompanies the paper submission "**Efficient Episodic Memory Utilization of Cooperative Multi-agent Reinforcement Learning (EMU)**" and is based on [GRF](https://github.com/google-research/football), [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced.
The paper is now available in [OpenReview](https://openreview.net/forum?id=LjivA1SLZ6) and [arXiv](https://arxiv.org/abs/2403.01112).

PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning and our code includes implementations of the following algorithms:
- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/pdf/2008.01062)
- [**EMC**: EMC: Episodic Multi-agent Reinforcement Learning with Curiosity-driven Exploration](https://arxiv.org/abs/2111.11032)
- [**CDS**: CDS: Celebrating Diversity in Shared Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2106.02195)

# Run an experiment
To train EMU(QPLEX) on SC2 setting tasks, run the following command:
```
python3 src/main.py --config=EMU_sc2 --env-config=sc2 with env_args.map_name=5m_vs_6m
```
For EMU(CDS), please change config file to EMU_sc2_cds.

To train EMU(QPLEX) on SC2 setting tasks, run the following command:
```
python3 src/main.py --config=EMU_grf --env-config=academy_3_vs_1_with_keeper
```
For EMU(CDS), please change config file to EMU_grf_cds.

# Publication
If you find this repository useful, please cite our paper:
```
@inproceedings{na2023efficient,
  title={Efficient Episodic Memory Utilization of Cooperative Multi-Agent Reinforcement Learning},
  author={Na, Hyungho and Seo, Yunkyeong and Moon, Il-chul},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
