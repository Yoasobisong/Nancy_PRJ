"""
Nancy Robot - Neck Differential Mechanism (PyBullet)
=====================================================
ALL 3 SERVOS ARE FIXED AT BASE (body/shoulders).

Structure:
  Base (body/shoulders) - FIXED
    ├── Yaw servo (FIXED at center) → drives rotating column
    ├── Left servo (FIXED at left)  → disc arm → linkage → head
    └── Right servo (FIXED at right) → disc arm → linkage → head

  Rotating column (driven by yaw servo)
    └── Universal joint at top
         └── Head platform (tilted by linkages)
              └── Skull

Key: When yaw rotates, ONLY the column and head turn.
     The servos and their arms stay put (like shoulders).
     The linkages must accommodate the yaw rotation.

Differential:
  L_servo = Pitch + Roll
  R_servo = Pitch - Roll
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
import time

# ============================================================
# Dimensions (meters)
# ============================================================
SERVO_SPACING = 0.075
SERVO_Z = 0.045
DISC_RADIUS = 0.018
COLUMN_BASE_Z = 0.038
COLUMN_TOP_Z = 0.095
HEAD_OFFSET_Z = 0.045
LINKAGE_ATTACH_X = 0.022

# ============================================================
# PyBullet init
# ============================================================
try:
    client = p.connect(p.GUI)
except Exception:
    print("Cannot connect to PyBullet GUI")
    exit(1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.setTimeStep(1.0 / 240.0)

p.resetDebugVisualizerCamera(0.32, 40, -22, [0, 0, 0.08])
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

plane = p.loadURDF("plane.urdf")
p.changeVisualShape(plane, -1, rgbaColor=[0.12, 0.12, 0.17, 1])

# ============================================================
# Create body helpers
# ============================================================
def make_box(hx, hy, hz, color):
    vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=color)
    return p.createMultiBody(0, baseVisualShapeIndex=vs, basePosition=[0, 0, -1])

def make_cyl(r, l, color):
    vs = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=l, rgbaColor=color)
    return p.createMultiBody(0, baseVisualShapeIndex=vs, basePosition=[0, 0, -1])

def make_sph(r, color):
    vs = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
    return p.createMultiBody(0, baseVisualShapeIndex=vs, basePosition=[0, 0, -1])

# ============================================================
# Create all parts
# ============================================================
# --- Base / body ---
base = make_box(0.065, 0.055, 0.006, [0.25, 0.25, 0.3, 1])

# --- 3 servos (ALL FIXED at base) ---
yaw_servo  = make_box(0.014, 0.007, 0.012, [0.2, 0.35, 0.85, 1])     # blue, center
left_servo = make_box(0.014, 0.007, 0.012, [0.85, 0.18, 0.18, 1])    # red, left
right_servo = make_box(0.014, 0.007, 0.012, [0.18, 0.72, 0.18, 1])   # green, right

# --- Left servo disc + arm ---
left_disc = make_cyl(0.011, 0.003, [0.9, 0.3, 0.3, 0.5])
left_arm = make_box(0.002, 0.002, DISC_RADIUS / 2, [1, 0.35, 0.35, 1])
left_tip = make_sph(0.003, [1, 0.2, 0.2, 1])

# --- Right servo disc + arm ---
right_disc = make_cyl(0.011, 0.003, [0.3, 0.85, 0.3, 0.5])
right_arm = make_box(0.002, 0.002, DISC_RADIUS / 2, [0.35, 1, 0.35, 1])
right_tip = make_sph(0.003, [0.2, 1, 0.2, 1])

# --- Yaw rotating column ---
column = make_cyl(0.005, COLUMN_TOP_Z - COLUMN_BASE_Z, [0.55, 0.55, 0.6, 1])

# --- Universal joint ---
u_joint = make_sph(0.007, [1, 0.85, 0, 1])

# --- Head platform ---
head_plat = make_box(0.035, 0.028, 0.003, [0.9, 0.55, 0.1, 0.85])
left_att = make_sph(0.003, [1, 0.5, 0.5, 1])    # linkage attach point
right_att = make_sph(0.003, [0.5, 1, 0.5, 1])

# --- Skull + face ---
skull = make_sph(0.038, [0.95, 0.85, 0.75, 0.8])
nose = make_sph(0.006, [1, 0.45, 0.45, 1])
eye_l = make_sph(0.007, [1, 1, 1, 1])
eye_r = make_sph(0.007, [1, 1, 1, 1])
pupil_l = make_sph(0.0035, [0.08, 0.08, 0.12, 1])
pupil_r = make_sph(0.0035, [0.08, 0.08, 0.12, 1])

# ============================================================
# Sliders
# ============================================================
sl_pitch = p.addUserDebugParameter("Pitch [deg]", -40, 40, 0)
sl_roll  = p.addUserDebugParameter("Roll [deg]", -30, 30, 0)
sl_yaw   = p.addUserDebugParameter("Yaw [deg]", -90, 90, 0)
sl_demo  = p.addUserDebugParameter("Demo Speed", 0, 3, 0)

# Line IDs
rod_l = -1
rod_r = -1
txt1 = -1
txt2 = -1
txt3 = -1
p.addUserDebugText("Nancy - Neck Mechanism (servos fixed at base)",
                   [0, 0, 0.21], [1, 0.85, 0.15], 1.4)

# ============================================================
# Helpers
# ============================================================
def qe(r, pi, y):
    return p.getQuaternionFromEuler([r, pi, y])

def qmul(q1, q2):
    return p.multiplyTransforms([0,0,0], q1, [0,0,0], q2)[1]

def rvec(q, v):
    m = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
    return m @ np.array(v)

def put(bid, pos, q=None):
    if q is None:
        q = [0, 0, 0, 1]
    p.resetBasePositionAndOrientation(bid, list(pos), list(q))

# ============================================================
# Main loop
# ============================================================
print("=" * 55)
print("  All servos FIXED at base. Only head rotates.")
print("  Drag sliders or set Demo > 0")
print("=" * 55)

t = 0.0
connected = True

while connected:
    try:
        # Check connection
        try:
            p.getConnectionInfo(client)
        except Exception:
            break

        # Read sliders
        try:
            pitch_d = p.readUserDebugParameter(sl_pitch)
            roll_d  = p.readUserDebugParameter(sl_roll)
            yaw_d   = p.readUserDebugParameter(sl_yaw)
            demo    = p.readUserDebugParameter(sl_demo)
        except Exception:
            break

        # Demo animation
        if demo > 0.1:
            s = demo
            pitch_d = 28 * math.sin(t * 0.8 * s)
            roll_d  = 20 * math.sin(t * 1.2 * s + 0.6)
            yaw_d   = 55 * math.sin(t * 0.45 * s + 1.2)

        # Differential
        L_deg = pitch_d + roll_d
        R_deg = pitch_d - roll_d
        L_deg = max(-70, min(70, L_deg))
        R_deg = max(-70, min(70, R_deg))

        L_rad = math.radians(L_deg)
        R_rad = math.radians(R_deg)
        yaw_r = math.radians(yaw_d)
        pitch_r = math.radians(pitch_d)
        roll_r = math.radians(roll_d)

        # =============================================
        # FIXED PARTS (never move)
        # =============================================
        put(base, [0, 0, 0.006])
        put(yaw_servo, [0, 0, 0.024])

        # Left servo - FIXED at base, left side
        L_servo_pos = np.array([-SERVO_SPACING / 2, 0, SERVO_Z])
        put(left_servo, L_servo_pos)

        # Right servo - FIXED at base, right side
        R_servo_pos = np.array([SERVO_SPACING / 2, 0, SERVO_Z])
        put(right_servo, R_servo_pos)

        # =============================================
        # LEFT SERVO ARM (rotates around Y axis, FIXED X position)
        # angle=0 → arm points up (+Z)
        # angle>0 → arm swings forward (-Y direction)
        # =============================================
        # Arm direction (in world frame, no yaw rotation!)
        L_arm_dir = np.array([0, -math.sin(L_rad), math.cos(L_rad)])
        L_arm_base = L_servo_pos + np.array([0, 0, 0.014])
        L_arm_tip_pos = L_arm_base + L_arm_dir * DISC_RADIUS
        L_arm_center = L_arm_base + L_arm_dir * DISC_RADIUS * 0.5

        q_L_arm = qe(L_rad, 0, 0)
        put(left_disc, L_arm_base, q_L_arm)
        put(left_arm, L_arm_center, q_L_arm)
        put(left_tip, L_arm_tip_pos)

        # =============================================
        # RIGHT SERVO ARM (same logic)
        # =============================================
        R_arm_dir = np.array([0, -math.sin(R_rad), math.cos(R_rad)])
        R_arm_base = R_servo_pos + np.array([0, 0, 0.014])
        R_arm_tip_pos = R_arm_base + R_arm_dir * DISC_RADIUS
        R_arm_center = R_arm_base + R_arm_dir * DISC_RADIUS * 0.5

        q_R_arm = qe(R_rad, 0, 0)
        put(right_disc, R_arm_base, q_R_arm)
        put(right_arm, R_arm_center, q_R_arm)
        put(right_tip, R_arm_tip_pos)

        # =============================================
        # YAW COLUMN (rotates around Z, driven by yaw servo)
        # This is the ONLY thing that rotates for yaw
        # =============================================
        q_yaw = qe(0, 0, yaw_r)
        col_center = np.array([0, 0, (COLUMN_BASE_Z + COLUMN_TOP_Z) / 2])
        put(column, col_center, q_yaw)

        # =============================================
        # UNIVERSAL JOINT (at top of column)
        # =============================================
        uj_pos = np.array([0, 0, COLUMN_TOP_Z])
        put(u_joint, uj_pos)

        # =============================================
        # HEAD PLATFORM
        # Orientation = Yaw * Pitch * Roll
        # Position = on top of universal joint
        # =============================================
        q_head = qmul(q_yaw, qmul(qe(0, -pitch_r, 0), qe(roll_r, 0, 0)))
        hp_pos = uj_pos + np.array([0, 0, 0.005])
        put(head_plat, hp_pos, q_head)

        # Linkage attach points on head platform
        # These ROTATE with the head (yaw + pitch + roll)
        L_attach = np.array(hp_pos) + rvec(q_head, [-LINKAGE_ATTACH_X, 0, 0])
        R_attach = np.array(hp_pos) + rvec(q_head, [LINKAGE_ATTACH_X, 0, 0])
        put(left_att, L_attach)
        put(right_att, R_attach)

        # =============================================
        # SKULL + FACE (all follow head orientation)
        # =============================================
        skull_pos = uj_pos + rvec(q_head, [0, 0, HEAD_OFFSET_Z])
        put(skull, skull_pos, q_head)

        put(nose, skull_pos + rvec(q_head, [0, -0.038, -0.005]), q_head)
        put(eye_l, skull_pos + rvec(q_head, [-0.016, -0.033, 0.008]), q_head)
        put(eye_r, skull_pos + rvec(q_head, [0.016, -0.033, 0.008]), q_head)
        put(pupil_l, skull_pos + rvec(q_head, [-0.016, -0.039, 0.008]), q_head)
        put(pupil_r, skull_pos + rvec(q_head, [0.016, -0.039, 0.008]), q_head)

        # =============================================
        # LINKAGE RODS
        # From FIXED servo arm tips → ROTATING head attach points
        # These rods change angle as yaw changes!
        # =============================================
        lc = [1.0, 0.35, 0.35]
        rc = [0.35, 0.95, 0.35]

        if rod_l >= 0:
            p.addUserDebugLine(L_arm_tip_pos.tolist(), L_attach.tolist(),
                              lc, 3, replaceItemUniqueId=rod_l)
        else:
            rod_l = p.addUserDebugLine(L_arm_tip_pos.tolist(), L_attach.tolist(), lc, 3)

        if rod_r >= 0:
            p.addUserDebugLine(R_arm_tip_pos.tolist(), R_attach.tolist(),
                              rc, 3, replaceItemUniqueId=rod_r)
        else:
            rod_r = p.addUserDebugLine(R_arm_tip_pos.tolist(), R_attach.tolist(), rc, 3)

        # =============================================
        # INFO TEXT
        # =============================================
        try:
            s1 = f"Pitch:{pitch_d:+.1f}  Roll:{roll_d:+.1f}  Yaw:{yaw_d:+.1f}"
            s2 = f"L servo:{L_deg:+.1f}  R servo:{R_deg:+.1f}"
            s3 = "Servos FIXED. Only head rotates."
            tp = [0.07, -0.07, 0.19]

            if txt1 >= 0:
                p.addUserDebugText(s1, tp, [1, 0.9, 0.3], 1.1, replaceItemUniqueId=txt1)
                p.addUserDebugText(s2, [tp[0], tp[1], tp[2]-0.015],
                                  [0.8, 0.8, 0.8], 1.0, replaceItemUniqueId=txt2)
                p.addUserDebugText(s3, [tp[0], tp[1], tp[2]-0.03],
                                  [0.5, 0.7, 0.5], 0.9, replaceItemUniqueId=txt3)
            else:
                txt1 = p.addUserDebugText(s1, tp, [1, 0.9, 0.3], 1.1)
                txt2 = p.addUserDebugText(s2, [tp[0], tp[1], tp[2]-0.015],
                                         [0.8, 0.8, 0.8], 1.0)
                txt3 = p.addUserDebugText(s3, [tp[0], tp[1], tp[2]-0.03],
                                         [0.5, 0.7, 0.5], 0.9)
        except Exception:
            pass

        p.stepSimulation()
        t += 1.0 / 240.0
        time.sleep(1.0 / 120.0)

    except (KeyboardInterrupt, SystemExit):
        break
    except Exception:
        break

try:
    p.disconnect()
except Exception:
    pass
print("Simulation ended.")
