from abc import abstractmethod

from nuplan_extent.planning.training.preprocessing.features.cameras import \
    Cameras


class AbstractCameraPipelines(object):
    @abstractmethod
    def __call__(self, cameras: Cameras) -> Cameras:
        pass

    @abstractmethod
    def __repr__(self):
        pass
