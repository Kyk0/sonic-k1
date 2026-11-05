import time

from _world import world


class ExternalShutdownException(Exception):
    pass


class SingleThreadedExecutor:
    def __init__(self):
        self.nodes, self.last = [], None

    def add_node(self, n):
        self.nodes.append(n)

    def spin_once(self, timeout_sec=None):
        t_end = time.monotonic() + (timeout_sec or 0)
        while True:
            m = world().latest
            if m is not None and m is not self.last:
                self.last = m
                for n in self.nodes:
                    for cb in n.subs:
                        cb(m)
                return
            if time.monotonic() >= t_end:
                return
            time.sleep(0.0005)
