import os
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# Motor specs + gain formula (stiffness = armature*(2*pi*nf)**2, damping =
# 2*damping_ratio*armature*(2*pi*nf)) taken from the official Booster BeyondMimic
# config (booster_train: assets/robots/booster.py + actuator.py,
# BOOSTER_K1_CFG). Ankle pitch/roll are parallel-actuated joints in reality (K1 has
# no direct-drive ankle); Booster models this as a 2x armature multiplier on the same
# E4310 motor (BoosterK1AnkleParaWrapperCfg, armature_ratio=(2.0, 2.0)) rather than a
# literal parallel-linkage sim — reused as-is here.
ARMATURE_E6408 = 0.0478125   # hip pitch
ARMATURE_E4315 = 0.0339552   # hip roll
ARMATURE_E4310 = 0.0282528   # hip yaw
ARMATURE_E6416 = 0.095625    # knee pitch
ARMATURE_ANKLE = 2.0 * ARMATURE_E4310   # ankle pitch/roll (parallel pair of E4310)
ARMATURE_R14 = 0.001         # shoulder/elbow
ARMATURE_HT4438 = 0.001      # head

EFFORT_E6408 = 68.0
EFFORT_E4315 = 76.0
EFFORT_E4310 = 38.3
EFFORT_E6416 = 112.0
EFFORT_ANKLE = 38.3          # effort_ratio=(1.0, 1.0) in BoosterK1AnkleParaWrapperCfg
EFFORT_R14 = 14.0
EFFORT_HT4438 = 6.0


def _gains(armature, natural_freq_hz, damping_ratio):
    omega = 2 * 3.1415926535 * natural_freq_hz
    return armature * omega**2, 2 * damping_ratio * armature * omega


STIFFNESS_HIP_PITCH, DAMPING_HIP_PITCH = _gains(ARMATURE_E6408, 4.0, 1.5)
STIFFNESS_HIP_ROLL, DAMPING_HIP_ROLL = _gains(ARMATURE_E4315, 4.0, 1.5)
STIFFNESS_HIP_YAW, DAMPING_HIP_YAW = _gains(ARMATURE_E4310, 4.0, 1.5)
STIFFNESS_KNEE, DAMPING_KNEE = _gains(ARMATURE_E6416, 4.0, 1.0)
STIFFNESS_ANKLE, DAMPING_ANKLE = _gains(ARMATURE_ANKLE, 4.0, 1.5)
STIFFNESS_ARM, DAMPING_ARM = _gains(ARMATURE_R14, 10.0, 2.0)
STIFFNESS_HEAD, DAMPING_HEAD = _gains(ARMATURE_HT4438, 10.0, 2.0)

# Verified by actually loading K1_CFG in Isaac Lab (tools/k1_inspect_isaaclab.py,
# headless, env_isaaclab) and k1.xml in plain MuJoCo (tools/k1_inspect_mujoco.py).
# IsaacLab's URDF importer merges the two fixed (no-joint) camera mount bodies
# (head_realsense_rgb_link, head_booster_stereo_rgb_link) into their parent, so
# IsaacLab only has 23 bodies (trunk + 22 joint-driven links) vs MuJoCo's 25 --
# those 2 camera bodies have no IsaacLab counterpart and must not be referenced by
# body-indexed code (reward/termination/body-position observations).
K1_ISAACLAB_JOINTS = [
    "trunk",
    "aahead_yaw_link",
    "aaleft_shoulder_pitch_link",
    "aaright_shoulder_pitch_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "aahead_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_elbow_pitch_link",
    "right_elbow_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
    "left_knee_pitch_link",
    "right_knee_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
# MuJoCo (k1.xml) DOF/body order (official UMR retarget order): aahead_yaw,
# aahead_pitch, aaleft_shoulder_pitch, left_shoulder_roll, left_elbow_pitch,
# left_elbow_yaw, aaright_shoulder_pitch, right_shoulder_roll, right_elbow_pitch,
# right_elbow_yaw, left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee_pitch,
# left_ankle_pitch, left_ankle_roll, right_hip_pitch, right_hip_roll, right_hip_yaw,
# right_knee_pitch, right_ankle_pitch, right_ankle_roll.
# isaaclab_to_mujoco_dof[k] = IsaacLab DOF index of the joint at MuJoCo position k.
K1_ISAACLAB_TO_MUJOCO_DOF = [
    0, 5, 1, 6, 10, 14, 2, 7, 11, 15, 3, 8, 12, 16, 18, 20, 4, 9, 13, 17, 19, 21,
]
# mujoco_to_isaaclab_dof[i] = MuJoCo DOF index of the joint at IsaacLab position i.
K1_MUJOCO_TO_ISAACLAB_DOF = [
    0, 2, 6, 10, 16, 1, 3, 7, 11, 17, 4, 8, 12, 18, 5, 9, 13, 19, 14, 20, 15, 21,
]
# Body lists include root (trunk) at index 0; = [0] + (dof mapping + 1), since body
# i+1 is the single link driven by joint i (no camera bodies in IsaacLab's 23).
K1_ISAACLAB_TO_MUJOCO_BODY = [0] + [i + 1 for i in K1_ISAACLAB_TO_MUJOCO_DOF]
K1_MUJOCO_TO_ISAACLAB_BODY = [0] + [i + 1 for i in K1_MUJOCO_TO_ISAACLAB_DOF]

K1_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": K1_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": K1_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": K1_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": K1_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": K1_MUJOCO_TO_ISAACLAB_BODY,
}


# K1 Robot Configuration (Booster official K1_22dof, 22 DOF, no waist, no wrists)
K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,  # as Booster K1: capsule caps made the leg colliders overlap at stance and shoved the feet ~9 cm apart in 0.2 s
        asset_path=f"{ASSET_DIR}/robot_description/urdf/k1/k1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=os.environ.get("K1_SELF_COLLISION", "1") == "1",  # debug switch
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    # pos/joint_pos taken from BOOSTER_K1_CFG (booster_train) -- the standing pose
    # used for K1's own BeyondMimic training, reused here instead of re-deriving.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            "left_shoulder_roll_joint": -1.3,
            "right_shoulder_roll_joint": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,  # K1 retargets sit at the hard limits (arms-down shoulder roll, straight knee); 0.9 put 89% of frames in the penalty/clamp zone
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": EFFORT_E6408,
                ".*_hip_roll_joint": EFFORT_E4315,
                ".*_hip_yaw_joint": EFFORT_E4310,
                ".*_knee_pitch_joint": EFFORT_E6416,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_HIP_PITCH,
                ".*_hip_roll_joint": STIFFNESS_HIP_ROLL,
                ".*_hip_yaw_joint": STIFFNESS_HIP_YAW,
                ".*_knee_pitch_joint": STIFFNESS_KNEE,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_HIP_PITCH,
                ".*_hip_roll_joint": DAMPING_HIP_ROLL,
                ".*_hip_yaw_joint": DAMPING_HIP_YAW,
                ".*_knee_pitch_joint": DAMPING_KNEE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_E6408,
                ".*_hip_roll_joint": ARMATURE_E4315,
                ".*_hip_yaw_joint": ARMATURE_E4310,
                ".*_knee_pitch_joint": ARMATURE_E6416,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=EFFORT_ANKLE,
            stiffness=STIFFNESS_ANKLE,
            damping=DAMPING_ANKLE,
            armature=ARMATURE_ANKLE,
        ),
        "arms": ImplicitActuatorCfg(
            # "aa"-prefixed shoulder-pitch joint names (aaleft_/aaright_) are an
            # upstream URDF quirk (alphabetical-sort hack), not a different joint --
            # match on the suffix only.
            joint_names_expr=[
                ".*shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim=EFFORT_R14,
            stiffness=STIFFNESS_ARM,
            damping=DAMPING_ARM,
            armature=ARMATURE_R14,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[".*head_yaw_joint", ".*head_pitch_joint"],
            effort_limit_sim=EFFORT_HT4438,
            stiffness=STIFFNESS_HEAD,
            damping=DAMPING_HEAD,
            armature=ARMATURE_HT4438,
        ),
    },
)

# K1 Action Scale
# Debug switch K1_ACTUATOR=ideal: explicit PD (IdealPDActuatorCfg) with identical gains/limits, like Booster's
# explicit BoosterDelayedPDActuator (without delay / torque-speed curve). Default: implicit PD.
if os.environ.get("K1_ACTUATOR", "implicit") == "ideal":
    from isaaclab.actuators import IdealPDActuatorCfg

    _vel = {"legs": 14.66, "feet": 17.59, "arms": 33.51, "head": 7.85}
    for _k, _a in list(K1_CFG.actuators.items()):
        K1_CFG.actuators[_k] = IdealPDActuatorCfg(
            joint_names_expr=_a.joint_names_expr,
            effort_limit=_a.effort_limit_sim,
            effort_limit_sim=_a.effort_limit_sim,
            velocity_limit=_vel.get(_k, 20.0),
            stiffness=_a.stiffness,
            damping=_a.damping,
            armature=_a.armature,
        )

K1_ACTION_SCALE = {}
for a in K1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            K1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
