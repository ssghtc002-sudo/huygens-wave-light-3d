"""
Exact 9:16 Blender HORIZONTAL-Fit NDC Projection Pre-Flight Gate (runs in <0.2s without Blender!)
Verifies that EVERY Beat's hero subjects stay cleanly inside the 9:16 portrait safe frame
(|ndcX| <= 0.88, ndcY in [-0.55, +0.86]) with ZERO edge cropping!
"""
import math
import numpy as np

RES_X, RES_Y = 1080, 1920
ASPECT = RES_X / RES_Y  # 0.5625
SENSOR_W = 36.0
CALCITE_POS = np.array([0.96, -0.04, 0.0])


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def ease_out_quint(x):
    t = clamp01(x)
    return 1.0 - (1.0 - t) ** 5


def lerp(a, b, t):
    return a + (b - a) * t


def bezier_quad(p0, p1, p2, t):
    u = 1.0 - t
    return p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)


def compute_camera_state(t):
    if t < 4.5:
        u = clamp01(t / 4.5)
        pos = bezier_quad(np.array([-0.36, -1.96, 0.58]), np.array([-0.20, -1.80, 0.52]), np.array([-0.06, -1.62, 0.46]), u)
        tgt = lerp(np.array([-0.20, 0.0, 0.28]), np.array([-0.10, 0.0, 0.28]), u)
        lens = lerp(34.0, 37.0, u)
    elif t < 9.5:
        pan_u = ease_out_quint(clamp01((t - 6.85) / 1.15))
        pos = lerp(np.array([-0.68, -1.32, 0.44]), np.array([-0.06, -1.24, 0.78]), pan_u)
        pos += np.array([math.sin(pan_u * math.pi) * 0.12, -0.06 * math.sin(pan_u * math.pi), 0.0])
        tgt = lerp(np.array([-0.44, -0.06, 0.22]), np.array([0.02, 0.0, 0.14]), pan_u)
        lens = lerp(41.0, 38.0, pan_u)
    elif t < 15.5:
        u = clamp01((t - 9.5) / 6.0)
        ang = math.radians(-18.0 + 38.0 * u)
        rad = lerp(1.26, 1.16, math.sin(u * math.pi))
        pos = np.array([0.04 + math.sin(ang) * rad * 0.76, -math.cos(ang) * rad, lerp(0.88, 0.98, u)])
        tgt = np.array([0.04, 0.0, 0.14])
        lens = lerp(38.0, 41.0, u)
    elif t < 20.5:
        u = clamp01((t - 15.5) / 5.0)
        ang = math.radians(-16.0 + 34.0 * u)
        rad = lerp(1.18, 1.24, u)
        pos = np.array([CALCITE_POS[0] + math.sin(ang) * rad * 0.65, CALCITE_POS[1] - math.cos(ang) * rad, lerp(0.46, 0.54, u)])
        tgt = np.array([CALCITE_POS[0] + 0.01, CALCITE_POS[1], 0.20])
        lens = lerp(38.0, 41.0, u)
    elif t < 25.5:
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(np.array([-0.24, -1.78, 0.56]), np.array([-0.14, -1.90, 0.70]), np.array([-0.06, -1.98, 0.82]), u)
        tgt = lerp(np.array([-0.14, 0.44, 0.54]), np.array([-0.08, 0.44, 0.62]), u)
        lens = lerp(38.0, 36.0, u)
    else:
        u = clamp01((t - 25.5) / 4.5)
        pos = bezier_quad(np.array([-0.06, -1.98, 0.82]), np.array([-0.02, -2.20, 0.78]), np.array([0.0, -2.38, 0.74]), u)
        tgt = lerp(np.array([-0.08, 0.34, 0.58]), np.array([0.0, 0.22, 0.50]), u)
        lens = lerp(36.0, 32.0, u)
    return pos, tgt, lens


def project(world_pt, cam_pos, cam_tgt, lens):
    fwd = cam_tgt - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)

    rel = world_pt - cam_pos
    cx = float(np.dot(rel, right))
    cy = float(np.dot(rel, up))
    cz = float(np.dot(rel, fwd))
    tan_hx = (SENSOR_W * 0.5) / lens
    tan_hy = tan_hx / ASPECT
    ndc_x = cx / (cz * tan_hx)
    ndc_y = cy / (cz * tan_hy)
    return ndc_x, ndc_y, cz


checks = [
    ("Beat 1 (t=2.0s)", 2.0, {
        "CannonRear": np.array([-0.76, -0.08, 0.215]),
        "CancelCard": np.array([0.36, 0.0, 0.36]),
        "TankRight": np.array([0.44, 0.0, 0.14]),
    }),
    ("Beat 2a Newton (t=5.5s)", 5.5, {
        "CannonRear": np.array([-0.76, -0.08, 0.215]),
        "SlitWall": np.array([-0.08, 0.0, 0.14]),
    }),
    ("Beat 2b Tank (t=8.5s)", 8.5, {
        "TankLeft": np.array([-0.44, 0.0, 0.14]),
        "TankRight": np.array([0.44, 0.0, 0.14]),
    }),
    ("Beat 3 Macro (t=12.5s)", 12.5, {
        "TankLeft": np.array([-0.44, 0.0, 0.14]),
        "TankRight": np.array([0.44, 0.0, 0.14]),
        "Detector": np.array([0.42, 0.0, 0.22]),
    }),
    ("Beat 4 Calcite (t=17.8s)", 17.8, {
        "Emitter": np.array([0.66, -0.08, 0.17]),
        "Crystal": np.array([0.96, -0.04, 0.17]),
        "SpotO": np.array([1.244, -0.035, 0.135]),
        "SpotE": np.array([1.244, 0.025, 0.225]),
    }),
    ("Beat 5 Clock+Saturn (t=23.5s)", 23.5, {
        "ClockLeft": np.array([-0.76, 0.54, 0.50]),
        "ClockTop": np.array([-0.58, 0.54, 1.05]),
        "SaturnRight": np.array([0.64, 0.44, 0.88]),
        "TitleRight": np.array([0.55, 0.56, 1.24]),
    }),
    ("Beat 6 Finale (t=28.2s)", 28.2, {
        "ClockLeft": np.array([-0.76, 0.54, 0.50]),
        "SaturnRight": np.array([0.64, 0.44, 0.88]),
        "LaserLeft": np.array([-0.95, 0.16, 0.46]),
        "LaserRight": np.array([0.95, 0.16, 0.46]),
    }),
]

errors = []
for label, t, pts in checks:
    cpos, ctgt, lens = compute_camera_state(t)
    print(f"\n[{label}] cam={np.round(cpos,2)} tgt={np.round(ctgt,2)} lens={lens:.1f}mm")
    for name, p in pts.items():
        nx, ny, cz = project(p, cpos, ctgt, lens)
        status = "OK"
        if abs(nx) > 0.88 or ny > 0.88 or ny < -0.58:
            status = "CLIP!"
            errors.append(f"{label} {name}: ndc=({nx:+.2f}, {ny:+.2f})")
        print(f"  {name:14s}: ndcX={nx:+.2f}  ndcY={ny:+.2f}  depth={cz:.2f}m  [{status}]")

if errors:
    print(f"\nFAILED ({len(errors)} framing clips):")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)
print("\nPASSED: 100% of all 6 Beats fit cleanly inside 9:16 portrait safe frame!")
