import random

from algorithms.RTSExtendedIsolationForest import RTSExtendedIsolationForest
from algorithms.RTSIsolationForest import RTSIsolationForest
from algorithms.RTSProximityIsolationForest import RTSProximityIsolationForest


class AlgorithmFactory:
    @staticmethod
    def create(algorithm):
        name, short_name, params = algorithm

        if name == "RTSIsolationForest":
            return RTSIsolationForest(**params)
        elif name == "RTSExtendedIsolationForest":
            return RTSExtendedIsolationForest(**params)
        elif name == "RTSProximityIsolationForest":
            return RTSProximityIsolationForest(**params)
        else:
            raise ValueError(f"Unknown algorithm: {name}")