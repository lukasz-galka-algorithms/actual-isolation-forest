import math
import random
import time

import numpy as np
from numba import njit

@njit
def _path_len_normalized_extended(left, right, points, normals, leaf_sz, n_nodes, average_node_depth, X):
    m = X.shape[0]
    out = np.empty(m, dtype=np.float64)
    d = X.shape[1]

    for i in range(m):
        node, depth = 0, 0
        while node < n_nodes and left[node] != -1:
            depth += 1
            proj = 0.0
            for j in range(d):
                proj += (X[i, j] - points[node, j]) * normals[node, j]
            node = left[node] if proj <= 0.0 else right[node]
        out[i] = (depth + leaf_sz[node] - 1) / average_node_depth
    return out

@njit
def is_all_the_same(data):
    n, d = data.shape
    for i in range(1, n):
        for j in range(d):
            if data[i, j] != data[0, j]:
                return False
    return True

class ExtendedITree:
    def __init__(self, extension_level=0):
        self.extension_level = extension_level

    def fit(self, X, Y=None):
        n, self.d = X.shape
        self.ext_level_computed = self.extension_level if self.extension_level < X.shape[1] else X.shape[1] - 1
        max_nodes = 2 * n
        self.node_depth_sum = 0.0

        self.left = np.full(max_nodes, -1, dtype=np.int64)
        self.right = np.full_like(self.left, -1)
        self.points = np.zeros((max_nodes, self.d), dtype=np.float64)
        self.normals = np.zeros_like(self.points)
        self.leaf_sz = np.zeros(max_nodes, dtype=np.int64)

        self._node_cnt = 0
        self._grow(X, 0)
        self.average_node_depth = self.node_depth_sum / n

    def _ensure_capacity(self):
        if self._node_cnt < self.left.size:
            return

        new_sz = self.left.size * 2

        self.left = np.resize(self.left, new_sz)
        self.right = np.resize(self.right, new_sz)
        self.points = np.resize(self.points, (new_sz, self.d))
        self.normals = np.resize(self.normals, (new_sz, self.d))
        self.leaf_sz = np.resize(self.leaf_sz, new_sz)

        self.left[self._node_cnt:new_sz] = -1
        self.right[self._node_cnt:new_sz] = -1
        self.leaf_sz[self._node_cnt:new_sz] = 0

    def _alloc_node(self):
        self._ensure_capacity()
        idx = self._node_cnt
        self._node_cnt += 1
        return idx

    def _grow(self, data, depth):
        idx = self._alloc_node()
        data_len = data.shape[0]
        data_dim = data.shape[1]

        if data_len <= 1:
            self.leaf_sz[idx] = data_len
            self.node_depth_sum += depth
            return idx

        if is_all_the_same(data):
            self.leaf_sz[idx] = data_len
            self.node_depth_sum += data_len * depth + data_len - 1 + (data_len * (data_len - 1)) / 2.0
            return idx

        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        self.points[idx] = np.random.uniform(mins, maxs)

        normal_vector = np.random.normal(0, 1, size=data_dim)
        normal_vector[(mins == maxs)] = 0.0
        variable_dims = [i for i in range(data_dim) if maxs[i] != mins[i]]
        np.random.shuffle(variable_dims)
        to_zero = len(variable_dims) - self.ext_level_computed - 1
        to_zero = max(0, to_zero)
        for i in range(to_zero):
            normal_vector[variable_dims[i]] = 0.0
        self.normals[idx] = normal_vector

        diffs = data - self.points[idx]
        projections = np.dot(diffs, self.normals[idx])
        mask = projections <= 0.0

        li = self._grow(data[mask], depth + 1)
        ri = self._grow(data[~mask], depth + 1)

        self.left[idx] = li
        self.right[idx] = ri
        return idx

    def get_path_length_normalized(self, X):
        return _path_len_normalized_extended(
            self.left, self.right,
            self.points, self.normals,
            self.leaf_sz, self._node_cnt,
            self.average_node_depth,
            X
        )

class ActualExtendedIsolationForest:

    def __init__(self, trees_number=100, samples_per_tree=256, extension_level=0, seed=None, **kwargs):
        self.uses_gpu = False
        self.trees_number = trees_number
        self.samples_per_tree = samples_per_tree
        self.extension_level = extension_level
        self.seed = seed
        self.forest = []
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            np.random.seed(random.randint(0, 2 ** 32 - 1))

    def fit(self, X, Y=None):
        cpu_phase_1 = time.perf_counter_ns()
        self.train_gpu_time = None

        self.forest = []
        for _ in range(self.trees_number):
            indices = np.random.choice(X.shape[0], self.samples_per_tree, replace=True)
            tree = ExtendedITree(extension_level=self.extension_level)
            tree.fit(X[indices])
            self.forest.append(tree)

        cpu_phase_2 = time.perf_counter_ns()
        self.train_cpu_time = cpu_phase_2 - cpu_phase_1
        self.train_all_time = self.train_cpu_time

    def decision_function(self, X, batch_size=2048):
        cpu_phase_1 = time.perf_counter_ns()
        self.eval_gpu_time = None

        n_samples = X.shape[0]
        scores = np.empty(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]

            path_lengths_sum_batch = np.zeros(X_batch.shape[0])

            for i, tree in enumerate(self.forest):
                path_lengths_batch = tree.get_path_length_normalized(X_batch)
                score_path_batch = np.exp2(-path_lengths_batch)
                path_lengths_sum_batch += score_path_batch

            scores[start:end] = path_lengths_sum_batch / self.trees_number

        cpu_phase_2 = time.perf_counter_ns()
        self.eval_cpu_time = cpu_phase_2 - cpu_phase_1
        self.eval_all_time = self.eval_cpu_time

        return scores

    def predict(self, X, threshold=0.5):
        scores = self.decision_function(X)
        return np.where(scores < threshold, 0, 1)
