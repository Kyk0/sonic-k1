"""Fake booster_robotics_sdk_python (only what booster_deploy and this app use)."""
from _world import world


class RobotMode:
    kUnknown, kDamping, kPrepare, kWalking, kCustom = -1, 0, 1, 2, 3


_NAMES = {0: "DAMPING", 1: "PREPARE", 2: "WALKING", 3: "CUSTOM"}


class _CF:
    def Init(self, domain, net):
        world().event(f"ChannelFactory.Init({domain}, {net})")


class ChannelFactory:
    _i = _CF()

    @staticmethod
    def Instance():
        return ChannelFactory._i


class B1LocoClient:
    def Init(self):
        world()

    def ChangeMode(self, mode):
        world().change_mode(_NAMES[mode])
        return 0
