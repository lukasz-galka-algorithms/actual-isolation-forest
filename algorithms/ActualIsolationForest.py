import math
import random
import time

import numpy as np
from numba import njit

@njit
def _path_len_normalized_iso(left, right, feat, val, leaf_sz, n_nodes, average_node_depth, X):
    m  = X.shape[0]
    out = np.empty(m, dtype=np.float64)

    for i in range(m):
        node, depth = 0, 0
        while node < n_nodes and left[node] != -1:
            depth += 1
            node = left[node] if X[i, feat[node]] < val[node] else right[node]
        out[i] = (depth + leaf_sz[node] - 1) / average_node_depth
    return out

class ITree:
    def __init__(self):
        pass

    def fit(self, X, Y=None):
        n, d = X.shape
        max_nodes = 2 * n
        self.node_depth_sum = 0.0

        self.left = np.full(max_nodes, -1, dtype=np.int64)
        self.right = np.full_like(self.left, -1)
        self.split_feat = np.zeros(max_nodes, dtype=np.int64)
        self.split_val = np.zeros(max_nodes, dtype=np.float64)
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
        self.split_feat = np.resize(self.split_feat, new_sz)
        self.split_val = np.resize(self.split_val, new_sz)
        self.leaf_sz = np.resize(self.leaf_sz, new_sz)

        self.left[self._node_cnt:new_sz] = -1
        self.right[self._node_cnt:new_sz] = -1
        self.leaf_sz[self._node_cnt:new_sz] = 0

    def _alloc(self):
        self._ensure_capacity()
        idx = self._node_cnt
        self._node_cnt += 1
        return idx

    def _grow(self, data, depth):
        idx = self._alloc()
        data_len = data.shape[0]
        data_dim = data.shape[1]

        if data_len <= 1:
            self.leaf_sz[idx] = data_len
            self.node_depth_sum += depth
            return idx

        feat = np.random.randint(data_dim)
        thr = None
        for _ in range(data_dim):
            feat = (feat + 1) % data_dim
            xmin, xmax = data[:, feat].min(), data[:, feat].max()
            if xmin != xmax:
                thr = np.random.uniform(xmin, xmax)
                break

        if thr == None:
            self.leaf_sz[idx] = data_len
            self.node_depth_sum += data_len * depth + data_len - 1 + (data_len * (data_len - 1)) / 2.0
            return idx

        mask = data[:, feat] < thr
        li = self._grow(data[mask], depth + 1)
        ri = self._grow(data[~mask], depth + 1)

        self.left[idx] = li
        self.right[idx] = ri
        self.split_feat[idx] = feat
        self.split_val[idx] = thr
        return idx

    def get_path_length_normalized(self, X):
        return _path_len_normalized_iso(
            self.left, self.right,
            self.split_feat, self.split_val,
            self.leaf_sz, self._node_cnt,
            self.average_node_depth,
            X
        )

class ActualIsolationForest:

    def __init__(self, trees_number = 100, samples_per_tree = 256, seed=None, **kwargs):
        self.uses_gpu = False
        self.trees_number = trees_number
        self.samples_per_tree = samples_per_tree
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
            iTree = ITree()
            iTree.fit(X[indices])
            self.forest.append(iTree)

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