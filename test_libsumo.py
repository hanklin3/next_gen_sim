import numpy as np
import os
import sys

if os.environ['LIBSUMO'] == "1":
    # sys.path.append(os.path.join(os.environ['W'], 'sumo-1.12.0', 'tools'))
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import libsumo as traci
    print('Using libsumo')
else:
    import traci
    print('Traci')

def traci_get_vehicle_data():
    car_list = traci.vehicle.getIDList()
    vehicle_list = []
    
    for car_id in car_list:
        x,y = traci.vehicle.getPosition(car_id)
        # Returns the angle in degrees of the named vehicle within the last step.
        deg2rad = np.pi/180.0
        angle_deg = traci.vehicle.getAngle(car_id) * deg2rad
        speed = traci.vehicle.getSpeed(car_id)
        # speed = traci.vehicle.getLateralSpeed(car_id)
        acceleration = traci.vehicle.getAcceleration(car_id)
        road_id = traci.vehicle.getRoadID(car_id)
        lane_id = traci.vehicle.getLaneID(car_id)
        lane_index = traci.vehicle.getLaneIndex(car_id)
        # speed = traci.getSpeed(car_id)
        # print(car_id, '(', x, y, ')', angle_deg, road_id, lane_index)
        # print('road_id, lane_id', road_id, lane_index, type(road_id), type(lane_index))

        # lat, lon, cos_heading, sin_heading = converter(x, y, angle_deg)
        # print('lat, lon, cos_heading, sin_heading', 
        #       lat, lon, cos_heading, sin_heading)
        vehicle_list.append([x, y, angle_deg, car_id, speed, road_id, 
                                        lane_id, lane_index, acceleration])
    return vehicle_list


def traci_set_vehicle_state():
    for vid in range(10):
        speed = 3.0
        traci.vehicle.setSpeed(str(int(vid)), speed)

# sumo_cmd = ['/home/gridsan/tools/groups/wulab/sumo-1.12.0/bin/sumo', '-c', './data/sumo/ring_18cars_0.1acc_1.5dec/circles.sumocfg', '--no-step-log', 'true', '--step-length', '0.4']
sumo_cmd = ['sumo', '-c', './data/sumo/ring_18cars_0.1acc_1.5dec/circles.sumocfg', '--no-step-log', 'true', '--step-length', '0.4']

traci.start(sumo_cmd)

step = 0
step_max = 1000
while step < step_max:
    print(step)

    traci.simulationStep()

    step += 1

    vehicle_list = traci_get_vehicle_data()
    # print(vehicle_list)
print('Success!!')
traci.close()
    