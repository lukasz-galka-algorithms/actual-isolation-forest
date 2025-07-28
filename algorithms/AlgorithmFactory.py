import random

from algorithms.ActualExtendedIsolationForest import ActualExtendedIsolationForest
from algorithms.ActualIsolationForest import ActualIsolationForest
from algorithms.ActualProximityIsolationForest import ActualProximityIsolationForest


class AlgorithmFactory:
    @staticmethod
    def create(algorithm):
        name, short_name, params = algorithm

        if name == "ActualIsolationForest":
            return ActualIsolationForest(**params)
        elif name == "ActualExtendedIsolationForest":
            return ActualExtendedIsolationForest(**params)
        elif name == "ActualProximityIsolationForest":
            return ActualProximityIsolationForest(**params)
        else:
            raise ValueError(f"Unknown algorithm: {name}")