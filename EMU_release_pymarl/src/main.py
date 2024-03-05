from unittest.util import _count_diff_hashable
import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

#ex = Experiment("pymarl")
ex = Experiment("pymarl", save_git_info=False) 

ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config_env(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def _get_config_alg(params, arg_name, subfolder, map_name):
    config_name = None
    
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name_default = _v.split("=")[1]
            del params[_i]
            break

    # use task dependent configuration
    if map_name=="3s5z_vs_3s6z":        
        if 'cds' in config_name_default: 
            config_name="EMU_sc2_hard_cds_3s5z_vs_3s6z"
        else:
            config_name="EMU_sc2_hard_3s5z_vs_3s6z"

    elif map_name=="6h_vs_8z":
        if 'cds' in config_name_default: 
            config_name="EMU_sc2_hard_cds_6h_vs_8z"
        else:
            config_name="EMU_sc2_hard_6h_vs_8z"

    elif map_name=="corridor":
        if 'cds' in config_name_default: 
            config_name="EMU_sc2_hard_cds_corridor"
        else:
            config_name="EMU_sc2_hard_corridor"

    elif map_name=="MMM2":
        if 'cds' in config_name_default: 
            config_name="EMU_sc2_hard_cds_MMM2"
        else:
            config_name="EMU_sc2_hard_MMM2"    
    else:
        
        if "academy" in map_name:
            if 'cds' in config_name_default: 
                config_name="EMU_grf_cds"
            else:
                config_name="EMU_grf"
        else:
            if 'cds' in config_name_default: 
                config_name="EMU_sc2_cds"
            else:
                config_name="EMU_sc2"
            config_name = config_name_default
                
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        #return config_dict
        return config_dict, config_name


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r", encoding="utf8", errors="ignore") as f:
        try:
            config_dict = yaml.safe_load(f)
            
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config= _get_config_env(params, "--env-config", "envs")
    config_dict = recursive_dict_update(config_dict, env_config)
    
    if "academy" in env_config['env']:
        map_name=env_config['env']
    else:
        map_name=env_config['env_args']['map_name']
    
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "env_args.map_name":
            map_name = _v.split("=")[1]
        
    print("Map_name    >>>>> ",map_name)
    alg_config, config_name = _get_config_alg(params, "--config", "algs", map_name)
    
    print("Config_file >>>>> ",config_name)
    config_dict = recursive_dict_update(config_dict, alg_config)
    
    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")

    if config_dict['config_name'] == '':
        cur_config_name = config_dict['agent']
    else:
        cur_config_name = config_dict['config_name']
    if config_dict['env_args']['map_name'] == '':
        save_folder = cur_config_name 
    else:
        save_folder = cur_config_name + '_' + config_dict['env_args']['map_name']

    save_folder   = cur_config_name
    file_obs_path = os.path.join(file_obs_path, save_folder )
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)