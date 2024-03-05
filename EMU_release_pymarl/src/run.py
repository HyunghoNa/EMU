import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.episode_buffer import Prioritized_ReplayBuffer
from components.transforms import OneHot
from utils.torch_utils import to_cuda
from modules.agents.LRN_KNN import LRU_KNN
from modules.agents.LRN_KNN_STATE import LRU_KNN_STATE
from components.episodic_memory_buffer import Episodic_memory_buffer

import numpy as np
import copy as cp
import random
import time

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    set_device = os.getenv('SET_DEVICE')
    if args.use_cuda and set_device != '-1':
        if set_device is None:
            args.device = "cuda"
        else:
            args.device = f"cuda:{set_device}"
    else:
        args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env,
                                     args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
        tb_info_get = os.path.join("results", "tb_logs", args.env, args.env_args['map_name'], "{}").format(unique_token)
        _log.info("saving tb_logs to " + tb_info_get)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def save_one_buffer(args, save_buffer, env_name, from_start=False):
    x_env_name = env_name
    if from_start:
        x_env_name += '_from_start/'
    path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.save_buffer_id) + '/'
    if os.path.exists(path_name):
        random_name = '../../buffer/' + x_env_name + '/buffer_' + str(random.randint(10, 1000)) + '/'
        os.rename(path_name, random_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    save_buffer.save(path_name)


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]
    args.obs_shape = env_info["obs_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "flag_win": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/' + args.env_args['map_name']

    if args.is_prioritized_buffer:
        buffer = Prioritized_ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                          args.prioritized_buffer_alpha,
                                          preprocess=preprocess,
                                          device="cpu" if args.buffer_cpu_only else args.device)
    else:
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              args.burn_in_period,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_save_buffer:
        save_buffer = ReplayBuffer(scheme, groups, args.save_buffer_size, env_info["episode_limit"] + 1,
                                   args.burn_in_period,
                                   preprocess=preprocess,
                                   device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_batch_rl:
        assert (args.is_save_buffer == False)
        x_env_name = env_name
        if args.is_from_start:
            x_env_name += '_from_start/'
        path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.load_buffer_id) + '/'
        assert (os.path.exists(path_name) == True)
        buffer.load(path_name)

    if getattr(args, "use_emdqn", False):
        ec_buffer=Episodic_memory_buffer(args,scheme)
    else:
        ec_buffer=None 

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    if args.runner != 'offpolicy':
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    if args.learner=="fast_QLearner" or args.learner=="qplex_curiosity_vdn_learner" or args.learner=="max_q_learner":
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, groups=groups)
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.runner == 'offpolicy':
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, test_mac=learner.extrinsic_mac)

    if hasattr(args, "save_buffer") and args.save_buffer:
        learner.buffer = buffer

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    embedder_update_time = 0
    ec_buffer_stats_update_time=0   

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        if not args.is_batch_rl:
            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            if getattr(args, "use_emdqn", False):
                
                if args.use_AEM == True:
                    #.. periodically update buffer statistics 
                    if (runner.t_env - ec_buffer_stats_update_time >= args.ec_buffer_stats_update_interval) and (runner.t_env >= args.t_EC_update):
                        ec_buffer.update_ec_buffer_stats()
                        ec_buffer_stats_update_time = runner.t_env
                    ec_buffer.update_ec_modified(episode_batch)
                else:
                    ec_buffer.update_ec_original(episode_batch)

            buffer.insert_episode_batch(episode_batch)

            if args.is_save_buffer:
                save_buffer.insert_episode_batch(episode_batch)
                if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                    save_buffer.is_from_start = False
                    save_one_buffer(args, save_buffer, env_name, from_start=True)
                    break
                if save_buffer.buffer_index % args.save_buffer_interval == 0 and os.name != 'nt':
                    print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

        for _ in range(args.num_circle):
            if buffer.can_sample(args.batch_size):
                if args.is_prioritized_buffer:
                    sample_indices, episode_sample = buffer.sample(args.batch_size)
                else:
                    episode_sample = buffer.sample(args.batch_size)

                if args.is_batch_rl:
                    runner.t_env += int(th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size
                                    
                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                if args.is_prioritized_buffer:
                    if getattr(args, "use_emdqn", False):
                        td_error = learner.train(episode_sample, runner.t_env, episode,ec_buffer=ec_buffer)
                    else:
                        td_error = learner.train(episode_sample, runner.t_env, episode)
                        buffer.update_priority(sample_indices, td_error)
                else:
                    if getattr(args, "use_emdqn", False):
                        td_error = learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                    else:
                        learner.train(episode_sample, runner.t_env, episode)

        # update state_embedding and prediction_network once in a while -------------------------
        if (args.use_emdqn==True) and args.use_AEM and ( args.memory_emb_type == 2 or args.memory_emb_type == 3 ) and (runner.t_env - embedder_update_time >= args.ec_buffer_embedder_update_interval):
            embedder_update_time = runner.t_env
            emb_start_time = time.time()
            ec_buffer.train_embedder()
            ec_buffer.update_embedding()
            emb_end_time = time.time()
            total_time = emb_end_time - emb_start_time
            if os.name != 'nt':
                print("Processing time for memory embedding:", total_time )
            if args.additional_update == True and ec_buffer.ec_buffer.build_tree == True:
                # re-update can fixate on current replay memory & can take long time
                if buffer.can_sample(args.buffer_size_update):
                    add_train_start_time = time.time()
                    all_episode_sample = buffer.sample(args.buffer_size_update)
                    ec_buffer.update_ec_modified(all_episode_sample)
                    add_train_end_time = time.time()
                    total_time = add_train_end_time - add_train_start_time
                    if os.name != 'nt':
                        print("Processing time for additional memory update:", total_time )
        #----------------------------------------------------------------------------------------

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0 :

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            logger.log_stat("num_circle", args.num_circle, runner.t_env)

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                episode_sample = runner.run(test_mode=True)
                if args.mac == "offline_mac":
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    learner.train(episode_sample, runner.t_env, episode, show_v=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_folder = args.config_name + '_' + args.env_args['map_name']
            save_path = os.path.join(args.local_results_path, "models", save_folder, args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)

            logger.console_logger.info("Saving models to {}".format(save_path))           
            learner.save_models(save_path, ec_buffer)

        episode += args.batch_size_run * args.num_circle

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    if args.is_save_buffer and save_buffer.is_from_start:
        save_buffer.is_from_start = False
        save_one_buffer(args, save_buffer, env_name, from_start=True)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
