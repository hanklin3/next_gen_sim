import numpy as np
from trajectory_pool import TrajectoryPool
from vehicle import Vehicle
import traci
import traci.constants as tc

def time_buff_to_traj_pool(TIME_BUFF):
    traj_pool = TrajectoryPool()
    for i in range(len(TIME_BUFF)):
        traj_pool.update(TIME_BUFF[i])
    return traj_pool


def to_vehicle(x, y, angle_deg, id, speed, road_id, lane_id, lane_index, acceleration):

    v = Vehicle()
    v.location.x = x
    v.location.y = y
    v.id = id
    # sumo: north, clockwise
    # NeuralNDE: east, counterclockwise
    v.speed_heading_deg = angle_deg #sumo -> NeuralNDE: (-angle_deg + 90 ) % 360
    # v.speed_heading_deg = (-angle_deg + 90 ) % 360 #NeuralNDE -> sumo: (-angle_deg + 90 ) % 360
    v.speed = speed
    v.road_id = road_id
    v.lane_id = lane_id
    v.lane_index = lane_index
    v.acceleration = acceleration

    factor = 1
    v.size.length, v.size.width = 3.6*factor, 1.8*factor
    v.safe_size.length, v.safe_size.width = 3.8*factor, 2.0*factor
    # v.update_poly_box_and_realworld_4_vertices()
    # v.update_safe_poly_box()
    return v

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
        vehicle_list.append(to_vehicle(x, y, angle_deg, car_id, speed, road_id, 
                                        lane_id, lane_index, acceleration))
    return vehicle_list


def traci_set_vehicle_state(model_output, buff_vid,
                            pred_lat, pred_lon, 
                            pred_cos_heading, pred_sin_heading,
                            pred_speed, pred_acceleration, sim_resol=0.4):
    for row_idx, row in enumerate(buff_vid):
        # print('row_idx, row', row_idx, row)
        vid = row[0]
        if np.isnan(vid):
            continue

        rad2deg = 180.0 / np.pi
        sin_heading = pred_sin_heading[row_idx][0]
        cos_heading = pred_cos_heading[row_idx][0]
        angle_deg = np.arctan2(sin_heading, cos_heading) * rad2deg
        angle_deg = tc.INVALID_DOUBLE_VALUE if np.isnan(angle_deg) else angle_deg
        print('angle_deg', angle_deg)
        # lane_index = int(buff_lane_index[row_idx][0])
        # print('lane_index', lane_index)
        
        if model_output == 'position_dxdy':
            dx = np.diff(pred_lat[row_idx,:])
            dy = np.diff(pred_lon[row_idx,:])
            speed = np.sqrt(dx**2 + dy**2) / sim_resol
            # speed = max(dx / configs['sim_resol'], dy / configs['sim_resol'])
            # print('dx', dx.shape, dx) # (4,)
            # print('speed', speed)
            
            # assert speed[0] > 0, (speed, pred_speed[row_idx,:], pred_speed[row_idx,:])
            print('position_dxdy', str(int(vid)), speed[0])
            traci.vehicle.setSpeed(str(int(vid)), speed[0])
            # traci.setPreviousSpeed(str(int(vid)), speed[0])
        elif model_output == 'position_xy':
            # If keepRoute is set to 1, the closest position
            # within the existing route is taken. If keepRoute is set to 0, the vehicle may move to
            # any edge in the network but its route then only consists of that edge.
            # If keepRoute is set to 2 the vehicle has all the freedom of keepRoute=0
            # but in addition to that may even move outside the road network.
            keeproute = 2 # which will map the vehicle to the exact x and y positions
            traci.vehicle.moveToXY(
                str(int(vid)),
                edgeID="",
                # laneIndex=-1, #lane_index,
                lane=-1,
                x=pred_lat[row_idx,0], #front_bumper_xy_sumo[0],
                y=pred_lon[row_idx,0], #front_bumper_xy_sumo[1],
                # angle=tc.INVALID_DOUBLE_VALUE, #(-angle_deg + 90 ) % 360,
                angle=angle_deg,
                keepRoute=keeproute,
            )
        elif model_output == 'speed':
            ####################Speed
            # print('pred_speed', pred_speed[row_idx,0])
            traci.vehicle.setSpeed(str(int(vid)), pred_speed[row_idx,0])
        elif model_output == 'no_set':
            pass
        elif model_output == 'acceleration':
            ####################Acce
            # print('pred_acceleration', pred_acceleration[row_idx,0])
            # traci.vehicle.setAcceleration(str(int(vid)), pred_acceleration[row_idx,0], 0.4)
            traci.vehicle.setAccel(str(int(vid)), pred_acceleration[row_idx,0])
            #####################
            # dx = np.diff(buff_lat[row_idx,:], n=2)
            # dy = np.diff(buff_lon[row_idx,:], n=2)
            # acceleration = np.sqrt(dx**2 + dy**2) / configs['sim_resol']
            
            # print('acceleration', acceleration)
            
            # traci.vehicle.setAcceleration(str(int(vid)), acceleration[0], 0.1)
        else:
            assert False, "Unsupported model output type"