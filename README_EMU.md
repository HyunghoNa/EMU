# Efficient episodic Memory Utilization for Multi-agent Reinforcement Learning

## Note
This codebase is based on [PyMARL](https://github.com/oxwhirl/pymarl), [SMAC](https://github.com/oxwhirl/smac), [QPLEX](https://github.com/wjh720/QPLEX), [EMC](https://github.com/kikojay/EMC), and [CDS](https://github.com/lich14/CDS) codebases which are open-sourced. 
We use the modified version of starcraft.py presented in [RODE](https://github.com/TonghanWang/RODE) to make it be compatible with other baselines such as QPLEX and CDS.

The implementation of the following methods can also be found in this codebase, which are finished by the authors of following papers:

- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/pdf/2008.01062)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement learning](https://arxiv.org/abs/1905.05408)
- [**Qatten**: Qatten: A General Framework for Cooperative Multiagent Reinforcement Learning](https://arxiv.org/abs/2002.03939)
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**EMC**: EMC: Episodic Multi-agent Reinforcement Learning with Curiosity-driven Exploration](https://arxiv.org/abs/2111.11032)
- [**CDS**: CDS: Celebrating Diversity in Shared Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2106.02195)

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over. We use a modified version of SMAC since we test 17 maps, which is illustrated in the folder of `EMC_smac_env`.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 
We evaluate our method in two environments: 17 maps in SMAC("sc2"). We use the default settings in SMAC, and the **results in our paper use Version SC2.4.10.
|    Task config   |  Algorithm config|
|        ----      |       ----       |
|        sc2       |       EMU_sc2    |

For curiosity and diversity incentive based on EMC and CDS, we use task-dependent configuration as their original paper suggest.

We provide a way to run experiments.

#### Use command line.
To train EMU on SC2 setting tasks, run the following command:
```shell
python3 src/main.py --config=EMU_sc2_hard --env-config=sc2 with env_args.map_name=5m_vs_6m
```
The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`


## Publication

If you find this repository useful, please cite our paper:
```
@inproceedings{na2024efficient,
  title={Efficient Episodic Memory Utilization of Cooperative Multi-Agent Reinforcement Learning},
  author={Na, Hyungho and Seo, Yunkyeong and Moon, Il-chul},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```