Fake Booster SDK + ROS 2 for rehearsing the **robot** code path of app.py on any machine: the same BoosterK1 backend,
ChannelFactory/B1LocoClient/ChangeMode calls, `/low_state` subscription and `joint_ctrl` publishing, against MuJoCo.

    PYTHONPATH=fake_booster python3 app.py        # robot path, fake robot; mode changes go to fake_booster.log

Never put this folder on the PYTHONPATH on the robot.
