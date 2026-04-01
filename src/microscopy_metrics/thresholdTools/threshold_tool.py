from abc import ABC, abstractmethod


class Threshold(object):
    name = "middle"
    _thresholdClasses = {}

    def __init_subclass__(cls):
        name = cls.name
        if name in cls._thresholdClasses:
            raise ValueError("Class was already registered")
        cls._thresholdClasses[name] = cls

    @classmethod
    def getInstance(cls, methodName):
        thresholdClass = cls._thresholdClasses[methodName]
        return thresholdClass()

    def getThreshold(self, image):
        return max(image) / 2

    @property
    @abstractmethod
    def name(self):
        pass
