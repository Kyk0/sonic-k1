"""Fake rclpy: /low_state from the fake robot, joint_ctrl into it."""
import time

from _world import world

_ok = False


def init():
    global _ok
    _ok = True


def ok():
    return _ok


def shutdown():
    global _ok
    _ok = False


class _Pub:
    def __init__(self, topic):
        self.topic = topic

    def get_subscription_count(self):
        return 1

    def publish(self, msg):
        if self.topic == "joint_ctrl":
            world().joint_ctrl(msg)


class Node:
    def __init__(self, name):
        self.subs = []

    def create_subscription(self, typ, topic, cb, qos):
        self.subs.append(cb)

    def create_publisher(self, typ, topic, qos):
        return _Pub(topic)

    def destroy_node(self):
        pass


def create_node(name):
    return Node(name)
