import numpy as np
from enum import Enum

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
        indices[indices < 0] = 0  # Ensure no negative indices

        for i in range(1, len(indices)):
            while indices[i] <= indices[i - 1]:
                indices[i:] = indices[i:] + 1
                indices[indices >= len(path)] = len(path) - 1  # Cap indices at the maximum valid index

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