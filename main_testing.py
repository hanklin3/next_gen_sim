# %%
import os
import sys
import yaml

from utils import set_sumo


if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
sumo_cmd = ['sumo', '-c', 'data/sumo/ring/circles.sumocfg']

path = 'configs/ring_inference.yml'

with open(path) as file:
    try:
        configs = yaml.safe_load(file)
        print(f"Loading config file: {path}")
    except yaml.YAMLError as exception:
        print(exception)
            
sumo_cmd = set_sumo(configs['gui'], 
                    configs['sumocfg_file_name'], configs['max_steps'])
print('sumo_cmd', sumo_cmd)
# %%
import traci

traci.start(sumo_cmd)
# %%
step = 0
while step < 1000:
    print(step)

    traci.simulationStep()
    
    step += 1

    car_list = traci.vehicle.getIDList()
    print('car_list', car_list)
    for car_id in car_list:
        x,y = traci.vehicle.getPosition(car_id)
        angle_deg = traci.vehicle.getAngle(car_id)
        # speed = traci.getSpeed(car_id)
        print(car_id, '(', x, y, ')', angle_deg)


    
    

traci.close()
# %%
