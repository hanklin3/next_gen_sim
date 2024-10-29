import os
import numpy as np
import sys
# from trajectory_pool import TrajectoryPool
from vehicle import Vehicle

is_libsumo = False
if os.environ['LIBSUMO'] == "1":
    # sys.path.append(os.path.join(os.environ['W'], 'sumo-1.12.0', 'tools'))
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import libsumo as traci
    print('Using libsumo')
    is_libsumo = True
else:
    import traci
    print('Traci')
    
import traci.constants as tc


def to_vehicle(x, y, angle_deg, id, speed, road_id, lane_id, lane_index, acceleration):

    v = Vehicle()
    v.location.x = x
    v.location.y = y
    v.id = id
    # sumo: north, clockwise
    # NeuralNDE: east, counterclockwise
    # v.speed_heading_deg = angle_deg #sumo -> NeuralNDE: (-angle_deg + 90 ) % 360
    v.speed_heading_deg = (-angle_deg + 90 ) % 360 #NeuralNDE -> sumo: (-angle_deg + 90 ) % 360
    v.speed = speed
    v.road_id = road_id
    v.lane_id = lane_id
    v.lane_index = lane_index
    v.acceleration = acceleration

    factor = 1
    v.size.length, v.size.width = 3.6*factor, 1.8*factor
    v.safe_size.length, v.safe_size.width = 3.8*factor, 2.0*factor
    v.update_poly_box_and_realworld_4_vertices()
    v.update_safe_poly_box()
    
    return v

def traci_get_vehicle_data():
    car_list = traci.vehicle.getIDList()
    vehicle_list = []
    
    for car_id in car_list:
        x,y = traci.vehicle.getPosition(car_id)
        # Returns the angle in degrees of the named vehicle within the last step.
        angle_deg = traci.vehicle.getAngle(car_id)
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


def cossin2deg(sin_heading, cos_heading):
    rad2deg = 180.0 / np.pi
    angle_deg = np.arctan2(sin_heading, cos_heading) * rad2deg
    angle_deg = 360 + angle_deg if angle_deg < 0 else angle_deg
    # angle_deg = angle_deg % 360
    return angle_deg


def traci_set_vehicle_state(model_output, buff_vid,
                            pred_lat, pred_lon, 
                            pred_cos_heading, pred_sin_heading,
                            pred_speed, pred_acceleration, 
                            buff_lat, buff_lon, buff_cos_heading, buff_sin_heading,
                            sim_resol=0.4):
    for row_idx, row in enumerate(buff_vid):
        # print('row_idx, row', row_idx, row)
        vid = row[0]
        if np.isnan(vid):
            continue

        next_idx = 0
        sin_heading = pred_sin_heading[row_idx][next_idx]
        cos_heading = pred_cos_heading[row_idx][next_idx]
        # sin_heading = buff_sin_heading[row_idx][-1]
        # cos_heading = buff_cos_heading[row_idx][-1]
        angle_deg = cossin2deg(sin_heading, cos_heading)
        angle_deg = tc.INVALID_DOUBLE_VALUE if np.isnan(angle_deg) else angle_deg
        angle_deg = (90-angle_deg + 90 ) % 360 # NeuralNDE -> sumo: (-angle_deg + 90 ) % 360
        print('angle_deg', angle_deg)
        # lane_index = int(buff_lane_index[row_idx][0])
        # print('lane_index', lane_index)

        current_idx = 4
        x_current = buff_lat[row_idx, current_idx]
        y_current = buff_lon[row_idx, current_idx]
        x_next = pred_lat[row_idx, next_idx]
        y_next = pred_lon[row_idx, next_idx]
        # angle_deg = cossin2deg(x_next - x_current, y_next - y_current)
        # why is it not atan2(y2-y1, x2-x1)?
        # angle_deg = cossin2deg(y_next - y_current, x_next - x_current)

        speedMode = 0 # no check
        traci.vehicle.setSpeedMode(str(int(vid)), speedMode)
        
        if model_output == 'position_dxdy':
            cur_pred_lat = np.concatenate([buff_lat[row_idx, current_idx:current_idx+1],  pred_lat[row_idx,:]])
            cur_pred_lon = np.concatenate([buff_lon[row_idx ,current_idx:current_idx+1],  pred_lon[row_idx,:]])
            dx = np.diff(cur_pred_lat[:])
            dy = np.diff(cur_pred_lon[:])
            speed = np.sqrt(dx**2 + dy**2) / sim_resol
            # speed = max(dx / configs['sim_resol'], dy / configs['sim_resol'])
            # print('dx', dx.shape, dx) # (4,)
            # print('speed', speed)

            # assert speed[0] > 0, (speed, pred_speed[row_idx,:], pred_speed[row_idx,:])
            # assert not np.isnan(float(speed[0])), float(speed[0])
            print('position_dxdy', str(int(vid)), float(speed[0]))

            traci.vehicle.setSpeed(str(int(vid)), float(speed[0]))
            # traci.setPreviousSpeed(str(int(vid)), speed[0])
        elif model_output == 'position_xy':
            # If keepRoute is set to 1, the closest position
            # within the existing route is taken. If keepRoute is set to 0, the vehicle may move to
            # any edge in the network but its route then only consists of that edge.
            # If keepRoute is set to 2 the vehicle has all the freedom of keepRoute=0
            # but in addition to that may even move outside the road network.
            keeproute = 2 # which will map the vehicle to the exact x and y positions

            x=float(pred_lat[row_idx,next_idx]) #front_bumper_xy_sumo[0],
            y=float(pred_lon[row_idx,next_idx]) #front_bumper_xy_sumo[1],
            angle=float(angle_deg)
            # angle=float((-angle_deg + 90 ) % 360)
            # angle=tc.INVALID_DOUBLE_VALUE #(-angle_deg + 90 ) % 360,
            print('moveToXY', vid, x, y, angle)
            if not is_libsumo:
                traci.vehicle.moveToXY(
                    str(int(vid)),
                    edgeID="", lane=-1, x=x, y=y, angle=angle, keepRoute=keeproute,
                )
            else:
                traci.vehicle.moveToXY(
                    str(int(vid)),
                    edgeID="", laneIndex=-1, x=x,  y=y, angle=angle, keepRoute=keeproute,
                )
        elif model_output == 'speed':
            ####################Speed
            # print('pred_speed', pred_speed[row_idx,0])
            traci.vehicle.setSpeed(str(int(vid)), float(pred_speed[row_idx,0]))
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