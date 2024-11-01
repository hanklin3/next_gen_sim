import numpy as np
import copy
import matplotlib.pyplot as plt

from vehicle.utils_vehicle import cossin2deg

def time_buff_to_traj_pool(TIME_BUFF):
    traj_pool = TrajectoryPool()
    for i in range(len(TIME_BUFF)):
        traj_pool.update(TIME_BUFF[i])
    return traj_pool

class TrajectoryPool(object):
    """
    A tool for managing trajectories (longitudinal data).
    """

    def __init__(self, max_missing_age=5, road_matcher=None, ROI_matcher=None):
        self.pool = {}  # vehicle pool
        self.max_missing_age = max_missing_age  # vehicles with > max_missing_age will be removed
        self.t_latest = 0  # time stamp of latest frame

        self.road_matcher = road_matcher  # drivable map matcher, used to get pixel points and map to drivable map.
        self.ROI_matcher = ROI_matcher  # used to match a position to ROI.

    def update(self, vehicle_list, ROI_matching=False):

        self.t_latest += 1

        # all vehicles missing days +1
        for vid, value in self.pool.items():
            self.pool[vid]['missing_days'] += 1

        # update trajectory pool
        for i in range(len(vehicle_list)):
            v = vehicle_list[i]
            if v.location.x == None or v.location.y == None:
                continue

            if ROI_matching:
                pxl_pt = self.road_matcher._world2pxl([v.location.x, v.location.y])
                pxl_pt[1] = np.clip(pxl_pt[1], a_min=0, a_max=self.road_matcher.road_map.shape[0]-1)
                pxl_pt[0] = np.clip(pxl_pt[0], a_min=0, a_max=self.road_matcher.road_map.shape[1]-1)
                v.region_position = self.ROI_matcher.region_position_matching(pxl_pt)
                v.yielding_area = self.ROI_matcher.yielding_area_matching(pxl_pt)
                v.at_circle_lane = self.ROI_matcher.at_circle_lane_matching(pxl_pt)

            if v.id not in self.pool.keys():
                self.pool[v.id] = {'vid': [str(v.id)], 'update': False, 'vehicle': [v], 'dt': [1], 't': [self.t_latest], 'missing_days': 0,
                                   'x': [v.location.x], 'y': [v.location.y], 'heading': [v.speed_heading_deg],
                                   'speed': [v.speed],
                                   'region_position': [v.region_position if ROI_matching else None],
                                   'yielding_area': [v.yielding_area if ROI_matching else None], 'at_circle_lane': [v.at_circle_lane if ROI_matching else None]}
            else:
                self.pool[v.id]['vehicle'].append(v)
                self.pool[v.id]['update'] = True  # means this vehicle's just updated
                self.pool[v.id]['dt'].append(copy.deepcopy(self.pool[v.id]['missing_days']))  # dt since last time saw it
                self.pool[v.id]['t'].append(self.t_latest)  # time stamp
                self.pool[v.id]['missing_days'] = 0  # just saw it, so clear missing days
                self.pool[v.id]['vid'].append(str(v.id))
                self.pool[v.id]['x'].append(v.location.x)
                self.pool[v.id]['y'].append(v.location.y)
                self.pool[v.id]['heading'].append(v.speed_heading_deg)
                self.pool[v.id]['region_position'].append(v.region_position if ROI_matching else None)
                self.pool[v.id]['yielding_area'].append(v.yielding_area if ROI_matching else None)
                self.pool[v.id]['at_circle_lane'].append(v.at_circle_lane if ROI_matching else None)
                self.pool[v.id]['speed'].append(v.speed)

        # remove dead traj id (missing for a long time)
        for vid, value in self.pool.copy().items():
            if self.pool[vid]['missing_days'] > self.max_missing_age:
                del self.pool[vid]

    def vehicle_id_list(self):
        return list(self.pool.keys())

    def flatten_trajectory(self, max_num_vehicles, time_length, output_vid=False):

        # create lat and lon and heading buffer
        veh_num = len(list(self.pool.keys()))
        buff_lat = np.empty([veh_num, time_length])
        buff_lat[:] = np.nan
        buff_lon = np.empty([veh_num, time_length])
        buff_lon[:] = np.nan
        buff_cos_heading = np.empty([veh_num, time_length])
        buff_cos_heading[:] = np.nan
        buff_sin_heading = np.empty([veh_num, time_length])
        buff_sin_heading[:] = np.nan
        buff_heading_deg = np.empty([veh_num, time_length])
        buff_heading_deg[:] = np.nan
        buff_vid = np.empty([veh_num, time_length])
        buff_vid[:] = np.nan
        buff_speed = np.empty([veh_num, time_length])
        buff_speed[:] = np.nan
        buff_acc = np.empty([veh_num, time_length])
        buff_acc[:] = np.nan
        buff_road_id = np.empty([veh_num, time_length], dtype=np.dtype('a16'))
        buff_road_id[:] = np.nan
        buff_lane_id = np.empty([veh_num, time_length], dtype=np.dtype('a16'))
        buff_lane_id[:] = np.nan
        buff_lane_index = np.empty([veh_num, time_length])
        buff_lane_index[:] = np.nan
        buff_time = np.empty([veh_num, time_length])
        buff_time[:] = np.nan

        # fill-in lon and lat and heading buffer
        i = 0
        for _, traj in self.pool.items():
            ts = traj['t']
            vs = traj['vehicle']
            for j in range(len(ts)):
                lat, lon = vs[j].location.x, vs[j].location.y
                heading = vs[j].speed_heading_deg  # Convert degrees to radians
                if lat is None:
                    continue
                t = self.t_latest - ts[j]
                if t >= time_length:
                    continue
                buff_lat[i, t] = lat
                buff_lon[i, t] = lon
                buff_cos_heading[i, t] = np.cos(np.radians(heading))
                buff_sin_heading[i, t] = np.sin(np.radians(heading))
                # buff_cos_heading[i, t] = heading
                # buff_sin_heading[i, t] = heading
                angle_deg = cossin2deg(buff_sin_heading[i, t], buff_cos_heading[i, t])
                assert np.allclose(angle_deg, heading), (angle_deg, heading)
                buff_heading_deg[i, t] = angle_deg
                buff_speed[i, t] = vs[j].speed
                buff_acc[i, t] = vs[j].acceleration
                buff_vid[i, t] = vs[j].id
                buff_road_id[i, t] = vs[j].road_id
                buff_lane_id[i, t] = vs[j].lane_id
                buff_lane_index[i, t] = vs[j].lane_index
                buff_time[i, t] = ts[j]
                # print('t', t, self.t_latest,  ts[j])
                # t 4 5 1
                # t 3 5 2
                # t 2 5 3
                # t 1 5 4
                # t 0 5 5
            i += 1

        # # fill-in id buffer
        # i = 0
        # for _, traj in self.pool.items():
        #     vs = traj['vehicle']
        #     buff_vid[i, :] = vs[-1].id
        #     i += 1

        # print('tp buff_time before', buff_time[0]) #  [5. 4. 3. 2. 1.]

        buff_lat = buff_lat[:, ::-1]
        buff_lon = buff_lon[:, ::-1]
        buff_cos_heading = buff_cos_heading[:, ::-1]
        buff_sin_heading = buff_sin_heading[:, ::-1]
        buff_heading_deg = buff_heading_deg[:, ::-1]
        buff_speed = buff_speed[:, ::-1]
        buff_acc = buff_acc[:, ::-1]
        buff_vid = buff_vid[:, ::-1]
        buff_road_id = buff_road_id[:, ::-1]
        buff_lane_id = buff_lane_id[:, ::-1]
        buff_lane_index = buff_lane_index[:, ::-1]
        buff_time = buff_time[:, ::-1]

        # print('tp buff_time after', buff_time[0]) # [1. 2. 3. 4. 5.]

        # pad or crop to m x max_num_vehicles
        buff_lat = self._fixed_num_vehicles(buff_lat, max_num_vehicles)
        buff_lon = self._fixed_num_vehicles(buff_lon, max_num_vehicles)
        buff_cos_heading = self._fixed_num_vehicles(buff_cos_heading, max_num_vehicles)
        buff_sin_heading = self._fixed_num_vehicles(buff_sin_heading, max_num_vehicles)
        buff_heading_deg = self._fixed_num_vehicles(buff_heading_deg, max_num_vehicles)
        buff_vid = self._fixed_num_vehicles(buff_vid, max_num_vehicles)
        buff_speed = self._fixed_num_vehicles(buff_speed, max_num_vehicles)
        buff_acc = self._fixed_num_vehicles(buff_acc, max_num_vehicles)
        buff_road_id = self._fixed_num_vehicles(buff_road_id, max_num_vehicles)
        buff_lane_id = self._fixed_num_vehicles(buff_lane_id, max_num_vehicles)
        buff_lane_index = self._fixed_num_vehicles(buff_lane_index, max_num_vehicles)

        if output_vid:
            return buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, \
                buff_vid, buff_speed, buff_acc, buff_road_id, buff_lane_id, buff_lane_index, buff_time
        else:
            return buff_lat, buff_lon, buff_cos_heading, buff_sin_heading #, buff_heading_deg

    @staticmethod
    def _fixed_num_vehicles(x, max_num_vehicles):

        m_veh, l = x.shape
        if m_veh >= max_num_vehicles:
            x_ = x[0:max_num_vehicles, :]
        else:
            x_ = np.empty([max_num_vehicles, l], dtype=x.dtype)
            x_[:] = np.nan
            x_[0:m_veh, :] = x

        return x_

