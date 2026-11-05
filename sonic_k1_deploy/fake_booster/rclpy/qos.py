class QoSProfile:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ReliabilityPolicy:
    BEST_EFFORT, RELIABLE = 0, 1


class HistoryPolicy:
    KEEP_LAST = 0
