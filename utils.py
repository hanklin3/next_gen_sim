import configparser
from sumolib import checkBinary
import os
import sys

from trajectory_pool import TrajectoryPool
from vehicle import Vehicle


def time_buff_to_traj_pool(TIME_BUFF):
    traj_pool = TrajectoryPool()
    for i in range(len(TIME_BUFF)):
        traj_pool.update(TIME_BUFF[i])
    return traj_pool


def to_vehicle(x, y, angle_deg, id, speed, road_id, lane_id, lane_index):

    v = Vehicle()
    v.location.x = x
    v.location.y = y
    v.id = id
    # sumo: north, clockwise
    # yours: east, counterclockwise
    v.speed_heading = (-angle_deg + 90 ) % 360
    v.speed = speed
    v.road_id = road_id
    v.lane_id = lane_id
    v.lane_index = lane_index

    factor = 1
    v.size.length, v.size.width = 3.6*factor, 1.8*factor
    v.safe_size.length, v.safe_size.width = 3.8*factor, 2.0*factor
    v.update_poly_box_and_realworld_4_vertices()
    v.update_safe_poly_box()
    return v


def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {'gui': content['simulation'].getboolean('gui'),
              'total_episodes': content['simulation'].getint('total_episodes'),
              'max_steps': content['simulation'].getint('max_steps'),
              'n_cars_generated': content['simulation'].getint('n_cars_generated'),
              'green_duration': content['simulation'].getint('green_duration'),
              'yellow_duration': content['simulation'].getint('yellow_duration'),
              'batch_size': content['simulation'].getint('batch_size'),
              'learning_rate': content['simulation'].getfloat('learning_rate'),
              'num_lanes': content['simulation'].getfloat('num_lanes'),
              'lane_length': content['simulation'].getfloat('lane_length'),
              'speed_limit': content['simulation'].getfloat('speed_limit'),
              'left_turn': content['simulation'].getfloat('left_turn'),
              'width_layers': content['model'].getint('width_layers'),
              'training_epochs': content['model'].getint('training_epochs'),
              'memory_size_min': content['memory'].getint('memory_size_min'),
              'memory_size_max': content['memory'].getint('memory_size_max'),
              'num_states': content['agent'].getint('num_states'),
              'num_actions': content['agent'].getint('num_actions'), 'gamma': content['agent'].getfloat('gamma'),
              'path_name': content['dir']['path_name'],
              'sumocfg_file_name': content['dir']['sumocfg_file_name'],
              'wandb': content['dir']['wandb']}
    return config


def import_test_configuration(config_file):
    """
    Read the config file regarding the testing and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {'gui': content['simulation'].getboolean('gui'), 'max_steps': content['simulation'].getint('max_steps'),
              'n_cars_generated': content['simulation'].getint('n_cars_generated'),
              'episode_seed': content['simulation'].getint('episode_seed'),
              'green_duration': content['simulation'].getint('green_duration'),
              'yellow_duration': content['simulation'].getint('yellow_duration'),
              'num_lanes': content['simulation'].getfloat('num_lanes'),
              'lane_length': content['simulation'].getfloat('lane_length'),
              'speed_limit': content['simulation'].getfloat('speed_limit'),
              'left_turn': content['simulation'].getfloat('left_turn'),
              'num_states': content['agent'].getint('num_states'),
              'num_actions': content['agent'].getint('num_actions'),
              'width_layers': content['agent'].getint('width_layers'),
              'model_to_test': content['agent'].getint('model_to_test'),
              'sumocfg_file_name': content['dir']['sumocfg_file_name'],
              'models_path_name': content['dir']['models_path_name'],
              'model_num': content['dir']['model_num']}
    return config

def import_transfer_configuration(config_file):
    """
    Read the config file regarding the testing and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {'gui': content['simulation'].getboolean('gui'), 'max_steps': content['simulation'].getint('max_steps'),
              'n_cars_generated': content['simulation'].getint('n_cars_generated'),
              'episode_seed': content['simulation'].getint('episode_seed'),
              'green_duration': content['simulation'].getint('green_duration'),
              'yellow_duration': content['simulation'].getint('yellow_duration'),
              'num_lanes': content['simulation'].getfloat('num_lanes'),
              'lane_length': content['simulation'].getfloat('lane_length'),
              'speed_limit': content['simulation'].getfloat('speed_limit'),
              'left_turn': content['simulation'].getfloat('left_turn'),
              'num_states': content['agent'].getint('num_states'),
              'num_actions': content['agent'].getint('num_actions'),
              'width_layers': content['agent'].getint('width_layers'),
              'model_to_test': content['agent'].getint('model_to_test'),
              'num_episodes': content['agent'].getint('num_episodes'),
              'sumocfg_file_name': content['dir']['sumocfg_file_name'],
              'source_path_name': content['dir']['source_path_name'],
              'model_num': content['dir']['model_num']}
    return config


def set_sumo(gui, sumocfg_file_path, max_steps, step_length=1.0):
    """
    Configure various parameters of SUMO
    """
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if not gui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", sumocfg_file_path,
                "--no-step-log", "true", "--waiting-time-memory", str(max_steps), "--step-length", str(step_length)]

    return sumo_cmd

def set_sumo_transfer(gui, dir_name, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if not gui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join(f'results/{dir_name}', sumocfg_file_name),
                "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd


def set_train_path(path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        model_dir_content = [name for name in dir_content if name.startswith('model_')]
        previous_versions = [int(name.split("_")[1]) for name in model_dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 


def set_test_path(models_path_name):
    """
    Returns a model path
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name + '/')

    if os.path.isdir(model_folder_path):
        return model_folder_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')

def set_transfer_path(source_path_name, model_num):
    """
    Returns a model path
    """
    model_folder_path = os.path.join(os.getcwd(), 'results/'+source_path_name+'model_'+str(model_num)+'/')
    
    if os.path.isdir(model_folder_path):
        return model_folder_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')
