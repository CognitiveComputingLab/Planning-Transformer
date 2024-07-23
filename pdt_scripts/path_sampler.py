import numpy as np
from enum import Enum
from shapely.geometry import LineString


class SamplingMethod(Enum):
    FIXED_TIME = 1
    FIXED_DISTANCE = 2
    LOG_TIME = 3
    LOG_DISTANCE = 4


class PathSampler:
    def __init__(self, method=SamplingMethod.FIXED_TIME):
        self.set_method(method)

    def set_method(self, method):
        if isinstance(method, int):
            method = SamplingMethod(method)
        if not isinstance(method, SamplingMethod):
            raise ValueError(f"Invalid sampling method: {method}. Expected a member of SamplingMethod Enum.")
        self.method = method

    def sample(self, path, num_points):
        path = np.array(path)

        if self.method == SamplingMethod.FIXED_TIME:
            return self._fixed_time(path, num_points)
        elif self.method == SamplingMethod.LOG_TIME:
            return self._log_time(path, num_points)
        elif self.method == SamplingMethod.FIXED_DISTANCE:
            return self._fixed_distance(path, num_points)
        elif self.method == SamplingMethod.LOG_DISTANCE:
            return self._log_distance(path, num_points)
        else:
            raise ValueError(f"Invalid sampling method: {self.method}, {SamplingMethod.LOG_TIME}")

    def _fixed_time(self, path, num_points):
        return path[np.linspace(0, len(path) - 1, num_points, dtype=int)]

    def _log_time(self, path, num_points):
        indices = np.logspace(0, np.log10(len(path)), num_points, dtype=int) - 1

        # make unique by shifting indices when consecutive items are equal
        # for i in range(1, len(indices)):
        #     indices[i] = max(indices[i], indices[i-1] + 1)

        return path[indices]

    def _fixed_distance(self, path, num_points):
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum_dists = np.insert(np.cumsum(dists), 0, 0)
        target_dists = np.linspace(0, cum_dists[-1], num_points)
        target_indices = np.searchsorted(cum_dists, target_dists, side='right') - 1
        target_indices[0], target_indices[-1] = 0, len(path) - 1
        return path[target_indices]

    def _log_distance(self, path, num_points):
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum_dists = np.insert(np.cumsum(dists), 0, 0)
        log_scale = np.logspace(0, 1, num_points, base=10) - 1
        max_log = log_scale[-1]
        scaled_log_dists = log_scale / max_log * cum_dists[-1]
        target_indices = np.searchsorted(cum_dists, scaled_log_dists, side='right') - 1
        return path[target_indices]

    # def _simplify_path_to_target_points(self, path, num_points, tolerance_increment=0.2, debug=False):
    #     if len(path) < 2 or num_points >= len(path):
    #         return path
    #
    #     # intial coarse simplify
    #     # path = simplify_path_to_target_points_fast(path, target_num_points*5)
    #     tolerance = 1
    #
    #     line = LineString(path)
    #
    #     simplified_line = line.simplify(tolerance)
    #     simplified_points = list(simplified_line.coords)
    #
    #     while len(simplified_points) != num_points:
    #         if debug: print(tolerance, len(simplified_points))
    #         if len(simplified_points) > num_points:
    #             tolerance += tolerance_increment
    #         else:
    #             tolerance -= tolerance_increment
    #             tolerance_increment /= 2
    #
    #         simplified_line = line.simplify(tolerance)
    #         simplified_points = list(simplified_line.coords)
    #
    #         if tolerance <= 0 or tolerance_increment < 1e-5:
    #             break
    #
    #     # Directly adjust the number of points to match the target
    #     adjust_path_points(simplified_points, num_points)
    #
    #     return simplified_points


def adjust_path_points(simplified_points, target_points):
    while len(simplified_points) < target_points:
        # Find indices of the furthest consecutive points
        furthest_idx = max(range(len(simplified_points) - 1),
                           key=lambda i: np.linalg.norm(
                               np.array(simplified_points[i]) - np.array(simplified_points[i + 1])))
        # Insert a midpoint
        midpoint = tuple(np.mean([simplified_points[furthest_idx], simplified_points[furthest_idx + 1]], axis=0))
        simplified_points.insert(furthest_idx + 1, midpoint)

    while len(simplified_points) > target_points:
        # Find indices of the closest consecutive points
        closest_idx = min(range(len(simplified_points) - 1),
                          key=lambda i: np.linalg.norm(
                              np.array(simplified_points[i]) - np.array(simplified_points[i + 1])))
        # Remove one of the closest points
        del simplified_points[closest_idx]
