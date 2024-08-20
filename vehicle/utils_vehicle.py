from trajectory_pool import TrajectoryPool
from vehicle import Vehicle
import traci

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
    # yours: east, counterclockwise
    v.speed_heading = (-angle_deg + 90 ) % 360
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