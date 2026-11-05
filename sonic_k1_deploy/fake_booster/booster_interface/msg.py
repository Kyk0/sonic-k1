class MotorCmd:
    def __init__(self):
        self.q = self.dq = self.tau = self.kp = self.kd = self.weight = 0.0


class LowCmd:
    CMD_TYPE_PARALLEL, CMD_TYPE_SERIAL = 0, 1

    def __init__(self):
        self.cmd_type = 0
        self.motor_cmd = []


class LowState:
    pass
