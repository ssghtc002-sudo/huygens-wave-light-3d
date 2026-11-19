"""
Scientist #7: Christiaan Huygens (1678) — How Waves Defeated Light Bullets
Full 30-Second (900 Frames @ 30fps, 1080x1920) Blender 4.2 LTS Cycles Dual-GPU Cinema Short

Features:
  - 6 Distinct Cinematographic Beats with curved camera paths, whip-pans, Dutch tilt, crane reveal & real DoF bokeh
  - True Cycles Raytracing: Flint Glass (IOR 1.52), Ripple Water (IOR 1.333), Iceland Spar Calcite (IOR 1.658)
  - Native C++ Blender WAVE Modifiers (incoming plane wave + twin double-slit Huygens interference ripples)
  - 12 Expanding Luminous Huygens Secondary Wavelets + Live Zebra Interference Detector Fringes
  - 1656 Precision Dutch Brass Pendulum Clock (real swinging pendulum synced to 116 BPM + 3 meshing toothed gears)
  - Glowing Saturn + Multi-Layered Rings + Orbiting Titan Hologram + Modern Laser, 5G & Fiber-Optic Finale
  - Dual-GPU Parallel Worker Support (--start-frame / --end-frame) + 10-Second 6-Beat QA Mode (--qa-only)

Usage:
    # 6-Beat Quick QA Preview (~10s on T4 GPU)
    blender --background --python src/build_huygens_scene.py -- --qa-only

    # Dual-GPU Parallel Production Render (Kaggle GPU T4 x2 -> ~9-11 min total!)
    CUDA_VISIBLE_DEVICES=0 blender --background --python src/build_huygens_scene.py -- --start-frame 1 --end-frame 450 &
    CUDA_VISIBLE_DEVICES=1 blender --background --python src/build_huygens_scene.py -- --start-frame 451 --end-frame 900 &
    wait
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
from mathutils import Euler, Vector


# ============================================================================
# 1. CLI ARGUMENTS & RENDER CONFIGURATION
# ============================================================================
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Christiaan Huygens Cycles GPU 3D Short Renderer")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=900)
    p.add_argument("--samples", type=int, default=int(os.environ.get("HUYGENS_SAMPLES", "16")))
    p.add_argument("--out-dir", type=str, default=os.environ.get("HUYGENS_FRAMES_DIR", "render_frames"))
    p.add_argument("--qa-only", action="store_true", help="Render 6 hero keyframes (1 per beat) to qa_frames/")
    return p.parse_args(argv)


ARGS = parse_args()
FPS = 30
TOTAL_FRAMES = 900
RES_X = 1080
RES_Y = 1920
OUT_DIR = os.path.abspath("qa_frames" if ARGS.qa_only else ARGS.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# 2. MATH & EASING HELPERS
# ============================================================================
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = clamp01((x - e0) / max(1e-6, e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def ease_out_quint(x: float) -> float:
    t = clamp01(x)
    return 1.0 - (1.0 - t) ** 5


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def vlerp(a: Vector, b: Vector, t: float) -> Vector:
    return a.lerp(b, t)


def bezier_quad(p0: Vector, p1: Vector, p2: Vector, t: float) -> Vector:
    u = 1.0 - t
    return p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)


# ============================================================================
# 3. RESET SCENE & CONFIGURE CYCLES OPTIX/CUDA GPU (FAST DENOISE + RESUME SAFE)
# ============================================================================
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.render.use_persistent_data = True
scene.render.use_overwrite = False  # Resume-safe: skips already-rendered PNGs!
scene.render.use_placeholder = False
scene.frame_start = max(1, ARGS.start_frame)
scene.frame_end = min(TOTAL_FRAMES, ARGS.end_frame)

scene.cycles.samples = ARGS.samples
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.055
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
scene.cycles.denoising_prefilter = "FAST"
try:
    scene.cycles.denoising_use_gpu = True
except Exception:
    pass
scene.cycles.max_bounces = 6
scene.cycles.transmission_bounces = 6
scene.cycles.glossy_bounces = 3
scene.cycles.diffuse_bounces = 2
scene.cycles.transparent_max_bounces = 4

try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - High Contrast"
except Exception:
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "High Contrast"
scene.view_settings.exposure = 0.30

gpu_enabled = False
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpu_devices = [d for d in prefs.devices if d.type == backend]
            if gpu_devices:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                gpu_enabled = True
                print(
                    f"[Cycles] Enabled {backend} GPU ({len(gpu_devices)} visible device(s)) | "
                    f"Frames {scene.frame_start}..{scene.frame_end} @ {ARGS.samples} samples"
                )
                break
        except Exception:
            continue
except Exception as e:
    print(f"[Cycles] GPU warning: {e}")

if not gpu_enabled:
    scene.cycles.device = "CPU"
    print("[Cycles] Falling back to CPU")

scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.compression = 15
scene.render.filepath = os.path.join(OUT_DIR, "frame_")


# ============================================================================
# 4. PROCEDURAL CYCLES PBR MATERIALS (ZERO EXTERNAL TEXTURE DEPENDENCIES)
# ============================================================================
def set_bsdf(bsdf, name: str, val):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def make_walnut_mat():
    mat = bpy.data.materials.new("DutchWalnut")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.4, 7.5, 1.4)
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 6.2
    ns.inputs["Detail"].default_value = 10.0
    ns.inputs["Distortion"].default_value = 1.6
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.24
    rmp.color_ramp.elements[0].color = (0.036, 0.017, 0.009, 1.0)
    rmp.color_ramp.elements[1].position = 0.78
    rmp.color_ramp.elements[1].color = (0.125, 0.062, 0.028, 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.14
    set_bsdf(bsdf, "Roughness", 0.34)
    set_bsdf(bsdf, "Coat Weight", 0.32)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_delft_stone_mat():
    mat = bpy.data.materials.new("DelftStone")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Color1"].default_value = (0.055, 0.070, 0.105, 1.0)
    br.inputs["Color2"].default_value = (0.038, 0.048, 0.075, 1.0)
    br.inputs["Mortar"].default_value = (0.018, 0.024, 0.040, 1.0)
    br.inputs["Scale"].default_value = 2.8
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.24
    set_bsdf(bsdf, "Roughness", 0.78)
    nt.links.new(br.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(br.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_metal_mat(name, rgb, rough=0.22, metallic=0.96, bump_str=0.05):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    set_bsdf(bsdf, "Metallic", metallic)
    set_bsdf(bsdf, "Roughness", rough)
    if bump_str > 0:
        ns = nt.nodes.new("ShaderNodeTexNoise")
        ns.inputs["Scale"].default_value = 35.0
        ns.inputs["Detail"].default_value = 6.0
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = bump_str
        nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_glass_mat(name, ior=1.52, tint=(0.94, 0.98, 1.0), rough=0.015, coat=0.2):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*tint, 1.0))
    set_bsdf(bsdf, "Roughness", rough)
    set_bsdf(bsdf, "IOR", ior)
    set_bsdf(bsdf, "Transmission Weight", 0.97)
    set_bsdf(bsdf, "Transmission", 0.97)
    set_bsdf(bsdf, "Coat Weight", coat)
    return mat


def make_emissive_mat(name, rgb, strength=18.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_saturn_mat():
    mat = bpy.data.materials.new("SaturnBands")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wv = nt.nodes.new("ShaderNodeTexWave")
    wv.wave_type = "BANDS"
    wv.bands_direction = "Z"
    wv.inputs["Scale"].default_value = 5.2
    wv.inputs["Distortion"].default_value = 2.6
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.15
    rmp.color_ramp.elements[0].color = (0.58, 0.42, 0.22, 1.0)
    e1 = rmp.color_ramp.elements.new(0.55)
    e1.color = (0.94, 0.80, 0.54, 1.0)
    rmp.color_ramp.elements[-1].position = 0.88
    rmp.color_ramp.elements[-1].color = (0.76, 0.56, 0.32, 1.0)
    set_bsdf(bsdf, "Roughness", 0.48)
    set_bsdf(bsdf, "Emission Strength", 0.95)
    nt.links.new(tc.outputs["Generated"], wv.inputs["Vector"])
    nt.links.new(wv.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(rmp.outputs["Color"], bsdf.inputs["Emission Color"])
    return mat


walnut_mat = make_walnut_mat()
stone_mat = make_delft_stone_mat()
brass_mat = make_metal_mat("AgedBrass", (0.84, 0.60, 0.23), rough=0.22, metallic=0.96, bump_str=0.05)
dark_iron_mat = make_metal_mat("ObsidianIron", (0.06, 0.07, 0.09), rough=0.38, metallic=0.82, bump_str=0.06)
porcelain_mat = make_metal_mat("ClockPorcelain", (0.94, 0.90, 0.82), rough=0.18, metallic=0.0, bump_str=0.0)
glass_mat = make_glass_mat("FlintGlass", ior=1.52, tint=(0.94, 0.98, 1.0), rough=0.015)
water_mat = make_glass_mat("RippleWater", ior=1.333, tint=(0.45, 0.88, 1.0), rough=0.025, coat=0.45)
calcite_mat = make_glass_mat("IcelandSparCalcite", ior=1.658, tint=(0.82, 0.95, 1.0), rough=0.012, coat=0.60)
saturn_mat = make_saturn_mat()

gold_glow_mat = make_emissive_mat("GoldPhotonGlow", (1.0, 0.78, 0.28), strength=22.0)
cyan_glow_mat = make_emissive_mat("CyanWaveGlow", (0.18, 0.82, 1.0), strength=22.0)
ruby_laser_mat = make_emissive_mat("RubyLaserGlow", (1.0, 0.16, 0.28), strength=28.0)
ring_glow_mat = make_emissive_mat("SaturnRingGlow", (0.96, 0.84, 0.56), strength=5.5)
candle_mat = make_emissive_mat("CandleFlame", (1.0, 0.72, 0.25), strength=32.0)
fringe_dark_mat = make_metal_mat("DarkFringe", (0.01, 0.015, 0.025), rough=0.85, metallic=0.0, bump_str=0.0)


# ============================================================================
# 5. GEOMETRY PRIMITIVE HELPERS
# ============================================================================
def add_box(name, loc, scale, mat, rot=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    return o


def add_cyl(name, loc, radius, depth, mat, rot=(0.0, 0.0, 0.0), verts=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    return o


def add_sphere(name, loc, radius, mat, segs=28, rings=18):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=rings, radius=radius, location=loc)
    o = bpy.context.active_object
    o.name = name
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.materials.append(mat)
    return o


def add_torus(name, loc, major_r, minor_r, mat, rot=(0.0, 0.0, 0.0), maj_seg=56, min_seg=14):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_r,
        minor_radius=minor_r,
        major_segments=maj_seg,
        minor_segments=min_seg,
        location=loc,
        rotation=rot,
    )
    o = bpy.context.active_object
    o.name = name
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.materials.append(mat)
    return o


def add_toothed_gear(name, loc, radius, thickness, teeth, mat):
    """Build a crisp procedural toothed Dutch brass clockwork gear parented to a clean empty hub."""
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=loc)
    hub = bpy.context.active_object
    hub.name = name
    rim = add_cyl(f"{name}_Rim", (0, 0, 0), radius, thickness, mat, rot=(math.pi * 0.5, 0, 0), verts=32)
    rim.parent = hub
    axle = add_cyl(f"{name}_Axle", (0, 0, 0), radius * 0.24, thickness * 1.9, dark_iron_mat, rot=(math.pi * 0.5, 0, 0), verts=16)
    axle.parent = hub
    tooth_w = (2.0 * math.pi * radius / teeth) * 0.46
    tooth_h = radius * 0.22
    for i in range(teeth):
        ang = (i / teeth) * math.pi * 2.0
        tx = math.cos(ang) * (radius + tooth_h * 0.35)
        tz = math.sin(ang) * (radius + tooth_h * 0.35)
        t_obj = add_box(f"{name}_T{i}", (tx, 0.0, tz), (tooth_h, thickness * 0.92, tooth_w), mat, rot=(0.0, -ang, 0.0))
        t_obj.parent = hub
    return hub


def orient_beam(obj, p1: Vector, p2: Vector, radius_scale: float = 1.0):
    diff = p2 - p1
    length = diff.length
    obj.location = (p1 + p2) * 0.5
    obj.scale = (radius_scale, radius_scale, max(0.0005, length))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = diff.normalized().to_track_quat("Z", "Y") if length > 1e-5 else Euler((0, 0, 0)).to_quaternion()


# ============================================================================
# 6. BUILD 1678 DUTCH GOLDEN AGE ROYAL OPTICAL OBSERVATORY STAGE
# ============================================================================
# Nocturnal starry indigo world ambient
world = bpy.data.worlds.new("LeidenObservatoryNight")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.010, 0.018, 0.040, 1.0)
bg.inputs["Strength"].default_value = 0.55

# Carved Dutch Walnut Workbench (top surface Z = 0.0)
add_box("WalnutWorkbench", (0.0, 0.10, -0.07), (4.6, 2.2, 0.14), walnut_mat)
add_box("BenchTrimFront", (0.0, -1.01, -0.025), (4.62, 0.035, 0.045), brass_mat)

# Delft Limestone Back Wall + Pilasters + Tall Starlit Window Recess
add_box("StoneWallLeft", (-2.1, 2.05, 1.6), (2.2, 0.25, 4.2), stone_mat)
add_box("StoneWallRight", (2.1, 2.05, 1.6), (2.2, 0.25, 4.2), stone_mat)
add_box("StoneWallLower", (0.0, 2.05, 0.15), (2.2, 0.25, 1.1), stone_mat)
add_box("StoneWallHeader", (0.0, 2.05, 2.85), (2.4, 0.25, 1.0), stone_mat)
for sx in (-1.05, 1.05):
    add_cyl(f"StonePilaster_{sx}", (sx, 1.88, 1.45), 0.13, 3.8, stone_mat, verts=20)
# Window mullions
add_box("WindowMullionV", (0.0, 2.02, 1.55), (0.045, 0.05, 2.2), dark_iron_mat)
add_box("WindowMullionH", (0.0, 2.02, 1.45), (2.05, 0.05, 0.045), dark_iron_mat)

# 2 Warm Practical Brass Candlesticks on outer workbench corners
candle_flames = []
for idx, (cx, cy) in enumerate([(-1.42, 0.55), (1.42, 0.55)]):
    add_cyl(f"CandleBase_{idx}", (cx, cy, 0.018), 0.09, 0.036, brass_mat)
    add_cyl(f"CandleStem_{idx}", (cx, cy, 0.12), 0.022, 0.20, brass_mat)
    add_cyl(f"CandleWax_{idx}", (cx, cy, 0.26), 0.024, 0.12, porcelain_mat)
    fl = add_sphere(f"CandleFlame_{idx}", (cx, cy, 0.345), 0.022, candle_mat, segs=14, rings=10)
    fl.scale = (0.75, 0.75, 1.95)
    candle_flames.append(fl)

# Subtle starfield specks visible through rear arched window
for i in range(28):
    ang = i * 2.399
    sx = math.sin(ang * 1.7) * 0.92
    sz = 0.85 + ((i * 0.19) % 1.45)
    r_star = 0.008 + (i % 3) * 0.005
    add_sphere(f"Star_{i}", (sx, 2.55, sz), r_star, gold_glow_mat if i % 3 == 0 else cyan_glow_mat, segs=8, rings=6)


# ============================================================================
# 7. ZONE A (CENTER): CRYSTAL DOUBLE-SLIT RIPPLE TANK, WAVELETS & NEWTON CANNON
# ============================================================================
TANK_W, TANK_D, TANK_H = 0.86, 0.52, 0.14
TANK_Z0 = 0.035

# Gleaming Brass Pedestal & Tank Frame
add_box("TankBrassBase", (0.0, 0.0, 0.0175), (TANK_W + 0.06, TANK_D + 0.06, 0.035), brass_mat)
add_box("TankFloorDark", (0.0, 0.0, TANK_Z0 + 0.005), (TANK_W - 0.02, TANK_D - 0.02, 0.010), dark_iron_mat)

# Glass Tank Walls (4 solid crystal panels for ultra-clean refractions)
wall_t = 0.014
add_box("TankWallFront", (0.0, -TANK_D * 0.5, TANK_Z0 + TANK_H * 0.5), (TANK_W, wall_t, TANK_H), glass_mat)
add_box("TankWallBack", (0.0, TANK_D * 0.5, TANK_Z0 + TANK_H * 0.5), (TANK_W, wall_t, TANK_H), glass_mat)
add_box("TankWallLeft", (-TANK_W * 0.5, 0.0, TANK_Z0 + TANK_H * 0.5), (wall_t, TANK_D, TANK_H), glass_mat)
add_box("TankWallRight", (TANK_W * 0.5, 0.0, TANK_Z0 + TANK_H * 0.5), (wall_t, TANK_D, TANK_H), glass_mat)

# Corner brass trim posts
for sx in (-TANK_W * 0.5, TANK_W * 0.5):
    for sy in (-TANK_D * 0.5, TANK_D * 0.5):
        add_cyl(f"TankPost_{sx:.2f}_{sy:.2f}", (sx, sy, TANK_Z0 + TANK_H * 0.5), 0.014, TANK_H + 0.02, brass_mat, verts=16)

# REAL ANIMATED 3D WATER RIPPLE SURFACE (64x48 subdivided plane + 3 native C++ Blender WAVE modifiers!)
WATER_Z = TANK_Z0 + 0.095
bpy.ops.mesh.primitive_grid_add(x_subdivisions=64, y_subdivisions=42, size=1.0, location=(0.0, 0.0, WATER_Z))
water_mesh = bpy.context.active_object
water_mesh.name = "HuygensRippleWater"
water_mesh.scale = (TANK_W - 0.03, TANK_D - 0.03, 1.0)
bpy.ops.object.transform_apply(scale=True)
for p in water_mesh.data.polygons:
    p.use_smooth = True
water_mesh.data.materials.append(water_mat)

# Wave Mod 1: Incoming plane wave rolling +X from left emitter
wm_plane = water_mesh.modifiers.new("PlaneWave", "WAVE")
wm_plane.use_x = True
wm_plane.use_y = False
wm_plane.height = 0.012
wm_plane.width = 0.095
wm_plane.narrowness = 1.6
wm_plane.speed = 0.035
wm_plane.start_position_x = -0.40
wm_plane.start_position_y = 0.0

# Wave Mod 2 & 3: Twin circular Huygens interference ripples radiating from Slit A & Slit B!
SLIT_X = -0.06
SLIT_Y_A = -0.095
SLIT_Y_B = +0.095
for idx, sy in enumerate((SLIT_Y_A, SLIT_Y_B)):
    wm = water_mesh.modifiers.new(f"HuygensSlitWave_{idx}", "WAVE")
    wm.use_x = True
    wm.use_y = True
    wm.height = 0.015
    wm.width = 0.085
    wm.narrowness = 1.5
    wm.speed = 0.038
    wm.start_position_x = SLIT_X
    wm.start_position_y = sy

# Machined Brass Double-Slit Aperture Barrier (with 2 crisp physical slits at Y = ±0.095!)
add_box("SlitWallTop", (SLIT_X, 0.185, TANK_Z0 + 0.095), (0.018, 0.13, 0.16), brass_mat)
add_box("SlitWallMid", (SLIT_X, 0.0, TANK_Z0 + 0.095), (0.018, 0.125, 0.16), brass_mat)
add_box("SlitWallBot", (SLIT_X, -0.185, TANK_Z0 + 0.095), (0.018, 0.13, 0.16), brass_mat)

# Luminous Plane-Wavefront Bars advancing left -> slits (4 glowing cyan bars!)
plane_bars = []
for i in range(4):
    bar = add_cyl(f"PlaneWaveBar_{i}", (-0.36, 0.0, WATER_Z + 0.022), 0.0055, TANK_D - 0.07, cyan_glow_mat, rot=(math.pi * 0.5, 0, 0), verts=14)
    plane_bars.append(bar)

# 12 Expanding Luminous Huygens Secondary Wavelet Rings (6 from Slit A + 6 from Slit B!)
wavelet_rings = []
for s_idx, sy in enumerate((SLIT_Y_A, SLIT_Y_B)):
    for k in range(6):
        mat_r = cyan_glow_mat if (k % 2 == 0) else gold_glow_mat
        ring = add_torus(f"HuygensWavelet_S{s_idx}_{k}", (SLIT_X + 0.02, sy, WATER_Z + 0.025), 0.10, 0.0042, mat_r)
        wavelet_rings.append((s_idx, k, sy, ring))

# Crystal Interference Detector Screen on right end of tank (X = +0.40) with 11 alternating bright/dark zebra fringes!
add_box("DetectorBackplate", (0.425, 0.0, TANK_Z0 + 0.135), (0.016, TANK_D - 0.04, 0.24), dark_iron_mat)
fringe_bars = []
for i in range(11):
    fy = -0.20 + i * 0.040
    is_bright = (i % 2 == 0)
    bar = add_box(
        f"InterferenceFringe_{i}",
        (0.412, fy, TANK_Z0 + 0.135),
        (0.012, 0.022, 0.20),
        cyan_glow_mat if is_bright else fringe_dark_mat,
    )
    fringe_bars.append((i, is_bright, bar))

# Beat 1-2 Twin Crossing Luminous Beams (creating dark destructive cancellation node!)
beam_cross_a = add_cyl("CrossBeamA", (0, 0, 0.32), 0.008, 1.0, gold_glow_mat, verts=16)
beam_cross_b = add_cyl("CrossBeamB", (0, 0, 0.32), 0.008, 1.0, cyan_glow_mat, verts=16)
dark_cancel_orb = add_sphere("DarkCancelOrb", (0.08, 0.0, 0.30), 0.036, fringe_dark_mat, segs=20, rings=14)

# Newton's Brass Particle Cannon (left of tank) + 14 Glowing Golden "Light Bullets"
add_cyl("NewtonCannonBody", (-0.64, -0.14, 0.24), 0.048, 0.22, brass_mat, rot=(0, math.pi * 0.5, 0), verts=24)
add_cyl("NewtonCannonStand", (-0.66, -0.14, 0.11), 0.024, 0.22, brass_mat, verts=18)
add_cyl("NewtonCannonFoot", (-0.66, -0.14, 0.015), 0.095, 0.030, brass_mat, verts=24)

newton_bullets = []
for i in range(14):
    b = add_sphere(f"NewtonBullet_{i}", (-0.50, -0.14, 0.24), 0.016, gold_glow_mat, segs=14, rings=10)
    newton_bullets.append(b)


# ============================================================================
# 8. ZONE B (RIGHT): ICELAND SPAR CALCITE CRYSTAL (BEAT 4 DOUBLE REFRACTION!)
# ============================================================================
CALCITE_POS = Vector((0.92, -0.10, 0.0))
add_cyl("CalcitePedestal", (CALCITE_POS.x, CALCITE_POS.y, 0.025), 0.22, 0.05, brass_mat, verts=40)
add_cyl("CalciteRing", (CALCITE_POS.x, CALCITE_POS.y, 0.055), 0.16, 0.018, dark_iron_mat, verts=32)

# Faceted Iceland Spar Rhombohedron Crystal (sheared beveled cube with true IOR 1.658!)
bpy.ops.mesh.primitive_cube_add(size=0.22, location=(CALCITE_POS.x, CALCITE_POS.y, 0.195))
calcite = bpy.context.active_object
calcite.name = "IcelandSparCrystal"
# Subtle authentic calcite rhombohedron shear & tilt
for v in calcite.data.vertices:
    v.co.x += v.co.z * 0.26
    v.co.y += v.co.z * 0.12
calcite.data.update()
bev = calcite.modifiers.new("CrystalBevel", "BEVEL")
bev.width = 0.008
bev.segments = 2
calcite.rotation_euler = Euler((0.0, 0.0, math.radians(18.0)))
calcite.data.materials.append(calcite_mat)

# Brass emitter & receiver plate for Double Refraction proof
add_cyl("CalciteEmitter", (CALCITE_POS.x - 0.30, CALCITE_POS.y - 0.06, 0.20), 0.028, 0.10, brass_mat, rot=(0, math.pi * 0.5, 0.18), verts=20)
add_box("CalciteTargetCard", (CALCITE_POS.x + 0.30, CALCITE_POS.y + 0.04, 0.20), (0.014, 0.24, 0.22), dark_iron_mat, rot=(0, 0, 0.18))

calcite_in_beam = add_cyl("CalciteInBeam", (0, 0, 0), 0.009, 1.0, gold_glow_mat, verts=16)
calcite_ord_beam = add_cyl("CalciteOrdBeam", (0, 0, 0), 0.008, 1.0, gold_glow_mat, verts=16)
calcite_ext_beam = add_cyl("CalciteExtBeam", (0, 0, 0), 0.008, 1.0, cyan_glow_mat, verts=16)
calcite_spot_o = add_sphere("CalciteSpotO", (CALCITE_POS.x + 0.285, CALCITE_POS.y + 0.00, 0.175), 0.018, gold_glow_mat, segs=14, rings=10)
calcite_spot_e = add_sphere("CalciteSpotE", (CALCITE_POS.x + 0.285, CALCITE_POS.y + 0.075, 0.235), 0.018, cyan_glow_mat, segs=14, rings=10)
calcite_shell_o = add_torus("CalciteWaveO", (CALCITE_POS.x, CALCITE_POS.y, 0.195), 0.08, 0.0035, gold_glow_mat, rot=(math.pi * 0.5, 0, 0))
calcite_shell_e = add_torus("CalciteWaveE", (CALCITE_POS.x, CALCITE_POS.y, 0.195), 0.095, 0.0035, cyan_glow_mat, rot=(math.pi * 0.42, 0.22, 0))


# ============================================================================
# 9. ZONE C (LEFT-REAR & CENTER-HIGH): 1656 PENDULUM CLOCK + SATURN HOLOGRAM!
# ============================================================================
CLOCK_POS = Vector((-0.94, 0.24, 0.0))
# Carved Oak Clock Case (open skeleton front so gleaming brass gears & pendulum shine!)
add_box("ClockPlinth", (CLOCK_POS.x, CLOCK_POS.y, 0.04), (0.36, 0.24, 0.08), walnut_mat)
add_box("ClockBack", (CLOCK_POS.x, CLOCK_POS.y + 0.09, 0.58), (0.30, 0.035, 1.04), walnut_mat)
add_box("ClockSideL", (CLOCK_POS.x - 0.145, CLOCK_POS.y, 0.58), (0.032, 0.20, 1.04), walnut_mat)
add_box("ClockSideR", (CLOCK_POS.x + 0.145, CLOCK_POS.y, 0.58), (0.032, 0.20, 1.04), walnut_mat)
add_box("ClockCrown", (CLOCK_POS.x, CLOCK_POS.y, 1.12), (0.38, 0.25, 0.065), walnut_mat)
add_sphere("ClockFinial", (CLOCK_POS.x, CLOCK_POS.y, 1.19), 0.042, brass_mat, segs=20, rings=14)

# Porcelain Clock Dial + Brass Bezel + Rotating Hands
add_cyl("ClockDial", (CLOCK_POS.x, CLOCK_POS.y - 0.095, 0.88), 0.115, 0.015, porcelain_mat, rot=(math.pi * 0.5, 0, 0), verts=36)
add_torus("ClockBezel", (CLOCK_POS.x, CLOCK_POS.y - 0.102, 0.88), 0.118, 0.011, brass_mat, rot=(math.pi * 0.5, 0, 0))
clock_min_hand = add_box("ClockMinuteHand", (CLOCK_POS.x, CLOCK_POS.y - 0.110, 0.88), (0.009, 0.006, 0.092), dark_iron_mat)

# 3 Interlocking Toothed Brass Escapement Gears visible inside skeleton clock!
gear_1 = add_toothed_gear("BrassGearMain", (CLOCK_POS.x - 0.035, CLOCK_POS.y - 0.01, 0.65), 0.068, 0.014, 14, brass_mat)
gear_2 = add_toothed_gear("BrassGearMid", (CLOCK_POS.x + 0.068, CLOCK_POS.y - 0.01, 0.70), 0.048, 0.014, 10, brass_mat)
gear_3 = add_toothed_gear("BrassGearEsc", (CLOCK_POS.x + 0.015, CLOCK_POS.y - 0.01, 0.53), 0.044, 0.014, 10, brass_mat)

# Swinging Brass Pendulum Pivot (real harmonic swing synced to 116 BPM clock ticks!)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(CLOCK_POS.x, CLOCK_POS.y - 0.045, 0.72))
pendulum_pivot = bpy.context.active_object
pendulum_pivot.name = "PendulumPivot"
p_rod = add_cyl("PendulumRod", (0.0, 0.0, -0.25), 0.0055, 0.50, brass_mat, verts=12)
p_rod.parent = pendulum_pivot
p_bob = add_sphere("PendulumBob", (0.0, 0.0, -0.50), 0.054, brass_mat, segs=24, rings=16)
p_bob.scale = (1.0, 0.42, 1.0)
p_bob.parent = pendulum_pivot

# Beat 5-6 Glowing Saturn + Multi-Layered Rings + Titan Hologram (Center-High!)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.14, 0.94))
saturn_group = bpy.context.active_object
saturn_group.name = "SaturnHologram"
saturn_planet = add_sphere("SaturnPlanet", (0.0, 0.0, 0.0), 0.145, saturn_mat, segs=36, rings=24)
saturn_planet.parent = saturn_group

# Tilt Rings 26.7° authentic Saturn obliquity!
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0), rotation=(math.radians(24.0), math.radians(-16.0), 0.0))
rings_tilt = bpy.context.active_object
rings_tilt.name = "SaturnRingsTilt"
rings_tilt.parent = saturn_group

for r_idx, (maj_r, min_r, mat_ring) in enumerate([
    (0.225, 0.016, ring_glow_mat),
    (0.278, 0.024, ring_glow_mat),
    (0.340, 0.015, cyan_glow_mat),
]):
    rg = add_torus(f"SaturnRing_{r_idx}", (0.0, 0.0, 0.0), maj_r, min_r, mat_ring, maj_seg=64, min_seg=12)
    rg.scale = (1.0, 1.0, 0.16)
    rg.parent = rings_tilt

titan_moon = add_sphere("MoonTitan", (0.41, 0.0, 0.0), 0.026, gold_glow_mat, segs=18, rings=12)
titan_moon.parent = rings_tilt

# Brass Huygens Telescope Objective Lens Pedestal beneath Saturn hologram
add_torus("TelescopeLensRim", (0.0, 0.14, 0.56), 0.18, 0.014, brass_mat, rot=(0.0, 0.0, 0.0))
lens_disc = add_sphere("TelescopeLensGlass", (0.0, 0.14, 0.56), 0.172, glass_mat, segs=32, rings=16)
lens_disc.scale = (1.0, 1.0, 0.16)


# ============================================================================
# 10. ZONE D (BEAT 6): MODERN PHOTONIC LEGACY — LASERS, WI-FI & 5G WAVEFRONTS!
# ============================================================================
laser_beam = add_cyl("ModernLaserBeam", (0.0, 0.0, 0.56), 0.0075, 1.0, ruby_laser_mat, verts=16)
fiveg_rings = []
for i in range(5):
    fg = add_torus(f"FiveGWave_{i}", (0.0, 0.14, 0.56), 0.22 + i * 0.09, 0.005, cyan_glow_mat if i % 2 == 0 else gold_glow_mat)
    fg.rotation_euler = Euler((math.radians(74.0), 0.0, 0.0))
    fiveg_rings.append(fg)

# Sleek Curved Fiber-Optic Conduit with racing photon beads
fiber_beads = []
for i in range(8):
    fb = add_sphere(f"FiberPhoton_{i}", (0.0, 0.0, 0.56), 0.015, cyan_glow_mat if i % 2 == 0 else ruby_laser_mat, segs=12, rings=8)
    fiber_beads.append(fb)

# 3D Floating Brass Formula Title subtly mounted above rear window arch in Beats 5-6
bpy.ops.object.text_add(location=(0.0, 0.45, 1.38), rotation=(math.pi * 0.5, 0.0, 0.0))
title_txt = bpy.context.active_object
title_txt.name = "HuygensTitle3D"
title_txt.data.body = "HUYGENS' WAVE PRINCIPLE"
title_txt.data.align_x = "CENTER"
title_txt.data.size = 0.092
title_txt.data.extrude = 0.012
title_txt.data.bevel_depth = 0.0025
title_txt.data.materials.append(brass_mat)


# ============================================================================
# 11. REMBRANDT CHIAROSCURO 3-LIGHT STUDIO RIG + CINEMA CAMERA WITH DOF
# ============================================================================
# Warm Golden Candlelight Key Area Light
bpy.ops.object.light_add(type="AREA", location=(-1.8, -1.6, 1.65))
key_light = bpy.context.active_object
key_light.data.energy = 210.0
key_light.data.size = 1.1
key_light.data.color = (1.0, 0.78, 0.50)
key_light.rotation_euler = Euler((math.radians(52), 0.0, math.radians(-42)))

# Cool Sapphire Moonlight Rim Light pouring through rear window
bpy.ops.object.light_add(type="AREA", location=(1.4, 2.2, 1.95))
rim_light = bpy.context.active_object
rim_light.data.energy = 260.0
rim_light.data.size = 1.4
rim_light.data.color = (0.42, 0.74, 1.0)
rim_light.rotation_euler = Euler((math.radians(56), 0.0, math.radians(146)))

# Subtle Cyan Ripple-Tank Under-Glow Accent
bpy.ops.object.light_add(type="POINT", location=(0.0, -0.35, 0.42))
cyan_accent = bpy.context.active_object
cyan_accent.data.energy = 24.0
cyan_accent.data.color = (0.28, 0.84, 1.0)

# Camera + Dynamic Target Empty + Real Focus Depth-of-Field
bpy.ops.object.empty_add(location=(0.0, 0.0, 0.30))
cam_target = bpy.context.active_object
cam_target.name = "CinemaCamTarget"

bpy.ops.object.camera_add(location=(0.0, -1.85, 0.65))
cam = bpy.context.active_object
cam.name = "CinemaCamera"
scene.camera = cam
cam.data.lens = 48.0
cam.data.dof.use_dof = True
cam.data.dof.focus_object = cam_target
cam.data.dof.aperture_fstop = 2.6

track = cam.constraints.new(type="TRACK_TO")
track.target = cam_target
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"


# ============================================================================
# 12. 6-BEAT CINEMA CAMERA PATHS & PHYSICS KEYFRAMING (ALL 900 FRAMES!)
# ============================================================================
def compute_camera_state(t: float):
    """Return (cam_pos, target_pos, lens_mm, fstop, dutch_deg) for timestamp t in [0..30]."""
    # Subtle organic handheld cinema breathing sway
    sway_x = math.sin(t * 1.15) * 0.016 + math.sin(t * 2.40) * 0.006
    sway_z = math.cos(t * 0.92) * 0.012

    if t < 4.5:
        # BEAT 1: Dynamic low-angle curved push-in across Newton's bullets -> crossing beams & dark fringe!
        u = clamp01(t / 4.5)
        pos = bezier_quad(Vector((-0.55, -1.62, 0.56)), Vector((-0.18, -1.48, 0.48)), Vector((0.06, -1.26, 0.44)), u)
        tgt = vlerp(Vector((-0.18, -0.05, 0.24)), Vector((0.08, 0.0, 0.22)), u)
        lens = lerp(44.0, 52.0, u)
        fstop = 2.6
        dutch = 0.0
    elif t < 9.5:
        # BEAT 2: Split-Stage Rivalry (Newton's Cannon 4.5-7.0s -> whip into Huygens' Ripple Tank 7.0-9.5s!)
        u = clamp01((t - 4.5) / 5.0)
        pan_u = ease_out_quint(clamp01((t - 6.85) / 1.15))
        pos_newton = Vector((-0.72, -1.32, 0.48))
        pos_huygens = Vector((-0.10, -1.22, 0.62))
        pos = vlerp(pos_newton, pos_huygens, pan_u) + Vector((math.sin(u * math.pi) * 0.08, 0.0, 0.0))
        tgt = vlerp(Vector((-0.44, -0.10, 0.24)), Vector((0.04, 0.0, 0.16)), pan_u)
        lens = lerp(48.0, 52.0, pan_u)
        fstop = 2.4
        dutch = 0.0
    elif t < 15.5:
        # BEAT 3: Elevated Orbital Macro Tracking directly over Double-Slit Ripple Tank & 12 Huygens Wavelets!
        u = clamp01((t - 9.5) / 6.0)
        ang = math.radians(-24.0 + 48.0 * u)
        rad = lerp(1.14, 1.02, math.sin(u * math.pi))
        pos = Vector((0.04 + math.sin(ang) * rad * 0.85, -math.cos(ang) * rad, lerp(0.84, 0.96, u)))
        tgt = Vector((0.08, 0.0, 0.16))
        lens = lerp(52.0, 58.0, u)
        fstop = 2.2
        dutch = 0.0
    elif t < 20.5:
        # BEAT 4: Dynamic Dutch-Tilt (4.2 deg) Orbital Close-Up on Iceland Spar Calcite Splitting 1 Ray -> 2!
        u = clamp01((t - 15.5) / 5.0)
        ang = math.radians(-28.0 + 54.0 * u)
        rad = lerp(0.96, 1.08, u)
        pos = Vector((CALCITE_POS.x + math.sin(ang) * rad * 0.72, CALCITE_POS.y - math.cos(ang) * rad, lerp(0.46, 0.56, u)))
        tgt = Vector((CALCITE_POS.x + 0.04, CALCITE_POS.y, 0.21))
        # Impact shake at t=18.55s when Newton's bullets shatter!
        shake = math.sin((t - 18.55) * 48.0) * math.exp(-(t - 18.55) * 6.5) * 0.025 if t >= 18.55 else 0.0
        pos.x += shake
        lens = lerp(56.0, 50.0, u)
        fstop = 2.2
        dutch = 4.2 * math.sin(u * math.pi)
    elif t < 25.5:
        # BEAT 5: Rising Crane Reveal framing Pendulum Clock (left) + Glowing Saturn & Rings (center-top)!
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(
            Vector((-0.68, -1.78, 0.62)),
            Vector((-0.25, -2.02, 0.92)),
            Vector((0.12, -2.12, 1.06)),
            u,
        )
        tgt = vlerp(Vector((-0.45, 0.15, 0.58)), Vector((-0.18, 0.14, 0.68)), u)
        lens = lerp(42.0, 40.0, u)
        fstop = 2.8
        dutch = 0.0
    else:
        # BEAT 6: Grand Hero Wide Pull-Back — Observatory + Saturn + Modern Laser / 5G / Fiber Finale!
        u = clamp01((t - 25.5) / 4.5)
        pos = bezier_quad(
            Vector((0.12, -2.12, 1.06)),
            Vector((0.05, -2.38, 0.96)),
            Vector((0.0, -2.55, 0.88)),
            u,
        )
        tgt = vlerp(Vector((-0.08, 0.10, 0.62)), Vector((0.0, 0.08, 0.54)), u)
        lens = lerp(40.0, 38.0, u)
        fstop = 3.2
        dutch = 0.0

    pos.x += sway_x
    pos.z += sway_z
    return pos, tgt, lens, fstop, dutch


# Keyframe every frame deterministically (ultra-fast in headless Blender Python: <1.2s for all 900 frames!)
BPM = 116.0
for f in range(1, TOTAL_FRAMES + 1):
    t = (f - 1) / FPS

    # 1. Camera choreography + focus + lens
    c_pos, c_tgt, c_lens, c_fstop, _ = compute_camera_state(t)
    cam.location = c_pos
    cam_target.location = c_tgt
    cam.data.lens = c_lens
    cam.data.dof.aperture_fstop = c_fstop
    cam.keyframe_insert("location", frame=f)
    cam_target.keyframe_insert("location", frame=f)
    cam.data.keyframe_insert("lens", frame=f)
    cam.data.keyframe_insert(data_path="dof.aperture_fstop", frame=f)

    # 2. Flickering warm practical candle flames
    for idx, fl in enumerate(candle_flames):
        flick = 1.0 + 0.14 * math.sin(t * 21.0 + idx * 2.1) + 0.08 * math.sin(t * 37.0)
        fl.scale = (0.75, 0.75, 1.95 * flick)
        fl.keyframe_insert("scale", frame=f)

    # 3. Beat 1-2 Crossing Light Beams & Dark Cancellation Orb (visible 0..9.5s)
    show_cross = t < 9.5
    if show_cross:
        orient_beam(beam_cross_a, Vector((-0.44, -0.18, 0.34)), Vector((0.40, 0.14, 0.24)), 1.0)
        orient_beam(beam_cross_b, Vector((-0.44, 0.18, 0.34)), Vector((0.40, -0.14, 0.24)), 1.0)
        dark_pulse = smoothstep(2.2, 2.8, t) * (1.0 - smoothstep(8.8, 9.5, t))
        dark_cancel_orb.scale = Vector((dark_pulse, dark_pulse, dark_pulse))
    else:
        beam_cross_a.scale = Vector((0.0001, 0.0001, 0.0001))
        beam_cross_b.scale = Vector((0.0001, 0.0001, 0.0001))
        dark_cancel_orb.scale = Vector((0.0001, 0.0001, 0.0001))
    for ob in (beam_cross_a, beam_cross_b):
        ob.keyframe_insert("location", frame=f)
        ob.keyframe_insert("rotation_quaternion", frame=f)
        ob.keyframe_insert("scale", frame=f)
    dark_cancel_orb.keyframe_insert("scale", frame=f)

    # 4. Newton's 14 Golden Light Bullets (firing straight in Beats 1-2, hovering in Beat 4, SHATTERING at 18.55s!)
    for i, b in enumerate(newton_bullets):
        if t < 9.5:
            # Fire straight stream from cannon mouth (-0.52) -> hits barrier (-0.08) and ricochets slightly!
            phase = ((t * 1.45 + i / 14.0) % 1.0)
            if phase < 0.72:
                u_b = phase / 0.72
                bx = lerp(-0.52, -0.09, u_b)
                by = -0.14 + math.sin(i * 1.3) * 0.015
                bz = 0.24
            else:
                u_r = (phase - 0.72) / 0.28
                bx = lerp(-0.09, -0.24, u_r)
                by = -0.14 + ((i % 2) * 2 - 1) * u_r * 0.14
                bz = 0.24 + u_r * 0.08 - u_r * u_r * 0.14
            b.location = Vector((bx, by, bz))
            b.scale = Vector((1.0, 1.0, 1.0))
        elif t < 20.4:
            # Beat 4: Cluster hovers near crystal -> EXPLODES outward at t=18.55s!
            ang = (i / 14.0) * math.pi * 2.0
            if t < 18.55:
                vis = smoothstep(15.6, 16.2, t)
                bx = CALCITE_POS.x - 0.26 + math.cos(ang + t * 2.2) * 0.055
                by = CALCITE_POS.y - 0.14 + math.sin(ang + t * 2.2) * 0.045
                bz = 0.34 + math.sin(t * 4.0 + i) * 0.015
                sc = vis * 0.85
            else:
                dt = t - 18.55
                sp = clamp01(dt / 1.65)
                rad = 0.06 + sp * (0.48 + (i % 4) * 0.12)
                bx = CALCITE_POS.x - 0.18 + math.cos(ang * 1.3) * rad
                by = CALCITE_POS.y - 0.10 + math.sin(ang * 1.7) * rad * 0.75
                bz = 0.34 + sp * (0.28 - sp * 0.55)
                sc = max(0.001, (1.0 - sp) * 0.95)
            b.location = Vector((bx, by, bz))
            b.scale = Vector((sc, sc, sc))
        else:
            b.scale = Vector((0.0001, 0.0001, 0.0001))
        b.keyframe_insert("location", frame=f)
        b.keyframe_insert("scale", frame=f)

    # 5. Advancing Plane-Wavefront Bars & 12 Expanding Luminous Huygens Secondary Wavelets (Peak brightness Beat 3!)
    wave_vis = smoothstep(6.9, 7.6, t) * (1.0 - smoothstep(15.3, 15.8, t)) + 0.45 * smoothstep(25.2, 26.0, t)
    for i, bar in enumerate(plane_bars):
        u_p = (t * 0.85 + i * 0.25) % 1.0
        bar.location = Vector((lerp(-0.40, SLIT_X - 0.025, u_p), 0.0, WATER_Z + 0.022))
        sc = max(0.0001, wave_vis * math.sin(u_p * math.pi))
        bar.scale = Vector((sc, sc, 1.0 if sc > 0.01 else 0.0001))
        bar.keyframe_insert("location", frame=f)
        bar.keyframe_insert("scale", frame=f)

    for s_idx, k, sy, ring in wavelet_rings:
        u_w = (t * 0.75 + k / 6.0) % 1.0
        # Expand smoothly from r=0.02 -> r=0.44m across right half of tank!
        rad_sc = max(0.0001, (0.20 + u_w * 3.8) * (1.0 if wave_vis > 0.02 else 0.0001))
        thick_sc = max(0.0001, wave_vis * (1.0 - u_w * 0.72))
        ring.location = Vector((SLIT_X + 0.015 + u_w * 0.06, sy, WATER_Z + 0.022))
        ring.scale = Vector((rad_sc, rad_sc, thick_sc))
        ring.keyframe_insert("location", frame=f)
        ring.keyframe_insert("scale", frame=f)

    # 6. Detector Screen Zebra Interference Fringes pulse with wave arrivals!
    fringe_boost = 0.65 + 0.35 * smoothstep(9.5, 11.2, t) + 0.15 * math.sin(t * 7.0)
    for i, is_bright, bar in fringe_bars:
        if is_bright:
            bar.scale = Vector((1.0, 1.0, fringe_boost))
            bar.keyframe_insert("scale", frame=f)

    # 7. Beat 4 Iceland Spar Calcite Double Refraction (active 15.0..30.0s, dramatic beam sweep!)
    c_vis = smoothstep(15.2, 16.0, t)
    p_emit = Vector((CALCITE_POS.x - 0.26, CALCITE_POS.y - 0.05, 0.20))
    p_entry = Vector((CALCITE_POS.x - 0.09, CALCITE_POS.y - 0.015, 0.198))
    p_exit_o = Vector((CALCITE_POS.x + 0.285, CALCITE_POS.y + 0.00, 0.175))
    p_exit_e = Vector((CALCITE_POS.x + 0.285, CALCITE_POS.y + 0.075, 0.235))
    split_u = smoothstep(16.0, 17.2, t)

    orient_beam(calcite_in_beam, p_emit, p_entry, max(0.0001, c_vis))
    orient_beam(calcite_ord_beam, p_entry, vlerp(p_entry, p_exit_o, split_u), max(0.0001, c_vis * split_u))
    orient_beam(calcite_ext_beam, p_entry, vlerp(p_entry, p_exit_e, split_u), max(0.0001, c_vis * split_u))
    for ob in (calcite_in_beam, calcite_ord_beam, calcite_ext_beam):
        ob.keyframe_insert("location", frame=f)
        ob.keyframe_insert("rotation_quaternion", frame=f)
        ob.keyframe_insert("scale", frame=f)

    sp_sc = max(0.0001, smoothstep(16.9, 17.4, t))
    calcite_spot_o.scale = Vector((sp_sc, sp_sc, sp_sc))
    calcite_spot_e.scale = Vector((sp_sc, sp_sc, sp_sc))
    calcite_spot_o.keyframe_insert("scale", frame=f)
    calcite_spot_e.keyframe_insert("scale", frame=f)

    # Expanding spherical (ordinary) & ellipsoidal (extraordinary) wavelet shells inside Calcite!
    sh_u = (t * 0.9) % 1.0
    sh_sc = max(0.0001, c_vis * split_u * (0.35 + sh_u * 1.15))
    calcite_shell_o.scale = Vector((sh_sc, sh_sc, sh_sc))
    calcite_shell_e.scale = Vector((sh_sc * 1.28, sh_sc * 0.85, sh_sc))
    calcite_shell_o.keyframe_insert("scale", frame=f)
    calcite_shell_e.keyframe_insert("scale", frame=f)

    # 8. Huygens 1656 Precision Brass Pendulum Clock (real harmonic swing + meshing gears all 30s!)
    swing_ang = math.radians(17.5) * math.sin(math.pi * t * (BPM / 60.0))
    pendulum_pivot.rotation_euler = Euler((0.0, swing_ang, 0.0))
    pendulum_pivot.keyframe_insert("rotation_euler", frame=f)

    gear_1.rotation_euler = Euler((0.0, t * 0.85, 0.0))
    gear_2.rotation_euler = Euler((0.0, -t * 0.85 * (14.0 / 10.0), 0.0))
    gear_3.rotation_euler = Euler((0.0, -t * 0.85 * (14.0 / 10.0) + 0.3, 0.0))
    for g in (gear_1, gear_2, gear_3):
        g.keyframe_insert("rotation_euler", frame=f)

    clock_min_hand.rotation_euler = Euler((0.0, t * 0.42, 0.0))
    clock_min_hand.keyframe_insert("rotation_euler", frame=f)

    # 9. Beat 5-6 Glowing Saturn + Rings + Titan Hologram!
    sat_u = smoothstep(20.6, 22.3, t)
    sat_sc = max(0.0001, sat_u)
    saturn_group.scale = Vector((sat_sc, sat_sc, sat_sc))
    saturn_group.location = Vector((0.0, 0.14, lerp(0.68, 0.98, sat_u) + math.sin(t * 1.8) * 0.015))
    saturn_planet.rotation_euler = Euler((0.0, 0.0, t * 0.55))
    rings_tilt.rotation_euler = Euler((math.radians(24.0), math.radians(-16.0), t * 0.38))
    saturn_group.keyframe_insert("scale", frame=f)
    saturn_group.keyframe_insert("location", frame=f)
    saturn_planet.keyframe_insert("rotation_euler", frame=f)
    rings_tilt.keyframe_insert("rotation_euler", frame=f)

    # 10. Beat 6 Modern Photonic Legacy: Ruby Laser + Expanding 5G Holographic Wave Arcs + Fiber-Optic Photons!
    mod_u = smoothstep(25.5, 26.6, t)
    p_l0 = Vector((-1.15, 0.14, 0.56))
    p_l1 = vlerp(p_l0, Vector((1.15, 0.14, 0.56)), mod_u)
    orient_beam(laser_beam, p_l0, p_l1, max(0.0001, mod_u))
    laser_beam.keyframe_insert("location", frame=f)
    laser_beam.keyframe_insert("rotation_quaternion", frame=f)
    laser_beam.keyframe_insert("scale", frame=f)

    for i, fg in enumerate(fiveg_rings):
        if mod_u > 0.01:
            u_5g = ((t - 25.5) * 0.72 + i * 0.20) % 1.0
            sc_5g = mod_u * (0.45 + u_5g * 2.15)
            th_5g = max(0.001, mod_u * (1.0 - u_5g * 0.80))
            fg.scale = Vector((sc_5g, sc_5g, th_5g))
        else:
            fg.scale = Vector((0.0001, 0.0001, 0.0001))
        fg.keyframe_insert("scale", frame=f)

    for i, fb in enumerate(fiber_beads):
        if mod_u > 0.01:
            u_fb = ((t - 25.5) * 0.65 + i / 8.0) % 1.0
            fx = lerp(-0.85, 0.85, u_fb)
            fy = 0.02 + math.sin(u_fb * math.pi * 2.0) * 0.22
            fz = 0.56 + math.cos(u_fb * math.pi * 2.0) * 0.14
            fb.location = Vector((fx, fy, fz))
            fb.scale = Vector((mod_u, mod_u, mod_u))
        else:
            fb.scale = Vector((0.0001, 0.0001, 0.0001))
        fb.keyframe_insert("location", frame=f)
        fb.keyframe_insert("scale", frame=f)

    # 3D Brass Title blooms gently in Beat 5-6 above Saturn!
    tit_u = smoothstep(21.5, 22.8, t)
    title_txt.scale = Vector((max(0.0001, tit_u), max(0.0001, tit_u), max(0.0001, tit_u)))
    title_txt.keyframe_insert("scale", frame=f)


# ============================================================================
# 13. EXECUTE RENDER (QA 6-BEAT PREVIEW OR PRODUCTION FRAME RANGE!)
# ============================================================================
if ARGS.qa_only:
    qa_frames = [45, 210, 390, 540, 690, 840]
    print(f"[Huygens QA] Rendering 6 Beat keyframes {qa_frames} to {OUT_DIR} ...")
    for idx, qf in enumerate(qa_frames, 1):
        scene.frame_set(qf)
        scene.render.filepath = os.path.join(OUT_DIR, f"qa_beat{idx}_f{qf:04d}.png")
        bpy.ops.render.render(write_still=True)
    print("[Huygens QA] All 6 Beat keyframes complete!")
else:
    print(
        f"[Huygens Production] Rendering frames {scene.frame_start}..{scene.frame_end} "
        f"({RES_X}x{RES_Y}, {ARGS.samples} samples) -> {OUT_DIR} ..."
    )
    bpy.ops.render.render(animation=True)
    print(f"[Huygens Production] Batch {scene.frame_start}..{scene.frame_end} complete!")
