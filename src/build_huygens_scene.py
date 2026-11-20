"""
Scientist #7: Christiaan Huygens (1678) — How Waves Defeated Light Bullets
V2 Cinema-Mastered 30-Second (900 Frames @ 30fps, 1080x1920) Blender 4.2 LTS Cycles Dual-GPU Short

Visual & Technical Master Fixes:
  1. True 9:16 Horizontal Sensor Fit (sensor_fit='HORIZONTAL', 36mm, 32-44mm lenses) — ZERO cropped subjects!
  2. AgX-Calibrated Emissive Shaders (strength 3.8-6.8) — Rich molten gold, electric cyan & ruby red (zero white clipping!)
  3. Forward-Propagating Semi-Circular Huygens Wavelet Arcs strictly confined inside the crystal ripple tank!
  4. Real 3D Sinusoidal Wave-Cancellation Demo in Beat 1 (replaces floating grey orb), cleanly hidden by t=4.45s!
  5. Sculpted Smooth-Shaded Brass Optical Cannon, Beveled Calcite Pedestal, Realistic Teardrop Candle Flames,
     Object-Mapped Delft Masonry + Recessed Warm Oak Bookshelves & Crisp Pinpoint Night Stars!
  6. Authentic Dutch Brass Refractor Telescope on Tripod (zero floating lens disc!), Complete Pendulum Clock + Saturn Framing,
     and Real Look-At Quaternion Camera with true Dutch Tilt & Silky Whip-Pan Transitions!

Usage:
    # 6-Beat Visual QA Preview (~12s on T4 GPU)
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
from mathutils import Euler, Quaternion, Vector


# ============================================================================
# 1. CLI ARGUMENTS & CONFIGURATION
# ============================================================================
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Christiaan Huygens V2 Cycles GPU 3D Short Renderer")
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
# 2. MATH, EASING & EXACT CAMERA ORIENTATION
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


def look_at_quat(cam_pos: Vector, target: Vector, roll_deg: float = 0.0) -> Quaternion:
    fwd = (target - cam_pos).normalized()
    q = fwd.to_track_quat("-Z", "Y")
    if abs(roll_deg) > 1e-4:
        q = q @ Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(roll_deg))
    return q


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
scene.render.use_overwrite = False
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
    scene.view_settings.look = "AgX - Punchy"
except Exception:
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - High Contrast"
    except Exception:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "High Contrast"
scene.view_settings.exposure = 0.15

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
                    f"[Cycles] Enabled {backend} GPU ({len(gpu_devices)} device(s)) | "
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
# 4. AGX-CALIBRATED PROCEDURAL CYCLES PBR MATERIALS
# ============================================================================
def set_bsdf(bsdf, name: str, val):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def make_walnut_mat(name="DutchWalnut", scale_vec=(1.2, 5.5, 1.2), dark=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = scale_vec
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 5.4
    ns.inputs["Detail"].default_value = 8.0
    ns.inputs["Distortion"].default_value = 1.3
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.25
    rmp.color_ramp.elements[0].color = (0.025, 0.011, 0.006, 1.0) if dark else (0.042, 0.019, 0.009, 1.0)
    rmp.color_ramp.elements[1].position = 0.78
    rmp.color_ramp.elements[1].color = (0.082, 0.038, 0.017, 1.0) if dark else (0.135, 0.066, 0.030, 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.10
    set_bsdf(bsdf, "Roughness", 0.34 if not dark else 0.42)
    set_bsdf(bsdf, "Coat Weight", 0.28)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_delft_stone_mat():
    """Object-mapped Delft limestone masonry so bricks never stretch or vanish on walls!"""
    mat = bpy.data.materials.new("DelftStone")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    # Map world X & Z onto Brick X & Y so horizontal courses align crisply across the rear wall!
    mp.inputs["Rotation"].default_value = (math.pi * 0.5, 0.0, 0.0)
    mp.inputs["Scale"].default_value = (1.15, 1.15, 1.15)
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.offset = 0.5
    br.inputs["Color1"].default_value = (0.085, 0.105, 0.145, 1.0)
    br.inputs["Color2"].default_value = (0.058, 0.072, 0.105, 1.0)
    br.inputs["Mortar"].default_value = (0.022, 0.028, 0.044, 1.0)
    br.inputs["Scale"].default_value = 3.2
    br.inputs["Mortar Size"].default_value = 0.025
    br.inputs["Mortar Smooth"].default_value = 0.15
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.28
    set_bsdf(bsdf, "Roughness", 0.74)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(br.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_metal_mat(name, rgb, rough=0.22, metallic=0.96, bump_str=0.03):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    set_bsdf(bsdf, "Metallic", metallic)
    set_bsdf(bsdf, "Roughness", rough)
    if bump_str > 0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        ns = nt.nodes.new("ShaderNodeTexNoise")
        ns.inputs["Scale"].default_value = 42.0
        ns.inputs["Detail"].default_value = 5.0
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = bump_str
        nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
        nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_glass_mat(name, ior=1.52, tint=(0.94, 0.98, 1.0), rough=0.012, coat=0.25):
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


def make_emissive_mat(name, rgb, strength=4.8):
    """AgX-calibrated saturated emission (rich vibrant hue, zero pure-white blow-out!)."""
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
    wv.inputs["Scale"].default_value = 4.8
    wv.inputs["Distortion"].default_value = 2.2
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.14
    rmp.color_ramp.elements[0].color = (0.48, 0.30, 0.12, 1.0)
    e1 = rmp.color_ramp.elements.new(0.52)
    e1.color = (0.90, 0.68, 0.34, 1.0)
    rmp.color_ramp.elements[-1].position = 0.88
    rmp.color_ramp.elements[-1].color = (0.68, 0.46, 0.22, 1.0)
    set_bsdf(bsdf, "Roughness", 0.52)
    set_bsdf(bsdf, "Emission Strength", 0.55)
    nt.links.new(tc.outputs["Generated"], wv.inputs["Vector"])
    nt.links.new(wv.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(rmp.outputs["Color"], bsdf.inputs["Emission Color"])
    return mat


def make_saturn_ring_mat(name, rgb, emit_strength=1.35):
    """Rich translucent golden-amber Saturn rings — never clipped white!"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    set_bsdf(bsdf, "Roughness", 0.35)
    set_bsdf(bsdf, "Alpha", 0.82)
    set_bsdf(bsdf, "Emission Color", (*rgb, 1.0))
    set_bsdf(bsdf, "Emission Strength", emit_strength)
    return mat


walnut_mat = make_walnut_mat("DutchWalnut", (1.2, 5.5, 1.2), dark=False)
clock_wood_mat = make_walnut_mat("ClockWalnut", (2.6, 2.6, 2.6), dark=True)
stone_mat = make_delft_stone_mat()
brass_mat = make_metal_mat("AgedBrass", (0.86, 0.60, 0.20), rough=0.20, metallic=0.96, bump_str=0.025)
bronze_mat = make_metal_mat("DarkBronze", (0.28, 0.19, 0.10), rough=0.34, metallic=0.88, bump_str=0.03)
dark_iron_mat = make_metal_mat("ObsidianIron", (0.04, 0.05, 0.07), rough=0.40, metallic=0.78, bump_str=0.02)
velvet_mat = make_metal_mat("RoyalSlateVelvet", (0.04, 0.06, 0.11), rough=0.88, metallic=0.0, bump_str=0.0)
porcelain_mat = make_metal_mat("ClockPorcelain", (0.94, 0.90, 0.80), rough=0.16, metallic=0.0, bump_str=0.0)
book_red_mat = make_metal_mat("LeatherBookRed", (0.24, 0.04, 0.05), rough=0.68, metallic=0.0, bump_str=0.0)
book_blue_mat = make_metal_mat("LeatherBookBlue", (0.05, 0.11, 0.24), rough=0.68, metallic=0.0, bump_str=0.0)
book_green_mat = make_metal_mat("LeatherBookGreen", (0.05, 0.18, 0.11), rough=0.68, metallic=0.0, bump_str=0.0)

glass_mat = make_glass_mat("FlintGlass", ior=1.52, tint=(0.94, 0.98, 1.0), rough=0.012)
water_mat = make_glass_mat("RippleWater", ior=1.333, tint=(0.24, 0.72, 0.96), rough=0.020, coat=0.50)
calcite_mat = make_glass_mat("IcelandSparCalcite", ior=1.658, tint=(0.86, 0.95, 1.0), rough=0.010, coat=0.65)
saturn_mat = make_saturn_mat()
ring_gold_mat = make_saturn_ring_mat("SaturnRingGold", (0.94, 0.72, 0.34), emit_strength=1.45)
ring_amber_mat = make_saturn_ring_mat("SaturnRingAmber", (0.78, 0.54, 0.24), emit_strength=1.10)
ring_icy_mat = make_saturn_ring_mat("SaturnRingCyan", (0.36, 0.78, 0.98), emit_strength=1.35)

# AgX-calibrated vivid saturated glowing materials (ZERO blinding white washout!)
gold_glow_mat = make_emissive_mat("GoldPhotonGlow", (1.0, 0.52, 0.05), strength=4.6)
gold_bullet_mat = make_emissive_mat("GoldBulletGlow", (1.0, 0.58, 0.08), strength=5.4)
cyan_glow_mat = make_emissive_mat("CyanWaveGlow", (0.03, 0.66, 1.0), strength=4.8)
cyan_soft_mat = make_emissive_mat("CyanWaveSoft", (0.04, 0.58, 0.96), strength=3.2)
ruby_laser_mat = make_emissive_mat("RubyLaserGlow", (1.0, 0.05, 0.10), strength=6.8)
candle_mat = make_emissive_mat("CandleFlame", (1.0, 0.48, 0.06), strength=8.5)
star_warm_mat = make_emissive_mat("StarWarm", (1.0, 0.82, 0.52), strength=3.6)
star_cool_mat = make_emissive_mat("StarCool", (0.58, 0.80, 1.0), strength=3.2)
fringe_dark_mat = make_metal_mat("DarkFringe", (0.008, 0.010, 0.018), rough=0.92, metallic=0.0, bump_str=0.0)


# ============================================================================
# 5. SMOOTH-SHADED & PROCEDURAL MESH HELPERS
# ============================================================================
def shade_smooth(obj, angle_deg: float = 38.0):
    for p in obj.data.polygons:
        p.use_smooth = True
    # EdgeSplit preserves crisp machined caps while making cylinder walls silky-smooth!
    es = obj.modifiers.new("SmoothSplit", "EDGE_SPLIT")
    es.split_angle = math.radians(angle_deg)


def add_box(name, loc, scale, mat, rot=(0.0, 0.0, 0.0), bevel_w=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    if bevel_w > 0:
        bv = o.modifiers.new("Bevel", "BEVEL")
        bv.width = bevel_w
        bv.segments = 2
    o.data.materials.append(mat)
    return o


def add_cyl(name, loc, radius, depth, mat, rot=(0.0, 0.0, 0.0), verts=48, smooth=True, bevel_w=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    if smooth:
        shade_smooth(o)
    if bevel_w > 0:
        bv = o.modifiers.new("Bevel", "BEVEL")
        bv.width = bevel_w
        bv.segments = 2
    o.data.materials.append(mat)
    return o


def add_cone(name, loc, r1, r2, depth, mat, rot=(0.0, 0.0, 0.0), verts=40):
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r1, radius2=r2, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    shade_smooth(o)
    o.data.materials.append(mat)
    return o


def add_sphere(name, loc, radius, mat, segs=32, rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=rings, radius=radius, location=loc)
    o = bpy.context.active_object
    o.name = name
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.materials.append(mat)
    return o


def add_torus(name, loc, major_r, minor_r, mat, rot=(0.0, 0.0, 0.0), maj_seg=64, min_seg=14):
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


def build_wavelet_arc_mesh(name, mat, arc_deg=156.0, segs=36, half_width=0.0032):
    """Build a crisp forward-propagating (+X) semi-circular Huygens wavelet ribbon mesh (unit radius r=1.0)."""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)

    verts = []
    faces = []
    half_ang = math.radians(arc_deg * 0.5)
    for i in range(segs + 1):
        a = lerp(-half_ang, half_ang, i / segs)
        ca, sa = math.cos(a), math.sin(a)
        r_in = 1.0 - half_width
        r_out = 1.0 + half_width
        verts.append((ca * r_in, sa * r_in, 0.0))
        verts.append((ca * r_out, sa * r_out, 0.0))
        verts.append((ca * 1.0, sa * 1.0, half_width * 0.9))
    for i in range(segs):
        b = i * 3
        n = (i + 1) * 3
        faces.append((b, b + 1, n + 1, n))
        faces.append((b, b + 2, n + 2, n))
        faces.append((b + 1, n + 1, n + 2, b + 2))
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for p in mesh.polygons:
        p.use_smooth = True
    return obj


def build_sinusoid_ribbon(name, length, amp, cycles, phase, mat, segs=56, radius=0.0055):
    """Build a true 3D sinusoidal light-wave ribbon along +X for Beat 1 interference visualization!"""
    curve_data = bpy.data.curves.new(name + "_Curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4
    spline = curve_data.splines.new("POLY")
    spline.points.add(segs)
    for i in range(segs + 1):
        u = i / segs
        x = u * length
        # Envelope smoothly fades into dark cancellation node at end!
        env = math.sin(u * math.pi) ** 0.35
        z = math.sin(u * cycles * math.pi * 2.0 + phase) * amp * env
        spline.points[i].co = (x, 0.0, z, 1.0)
    obj = bpy.data.objects.new(name, curve_data)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def add_toothed_gear(name, loc, radius, thickness, teeth, mat):
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=loc)
    hub = bpy.context.active_object
    hub.name = name
    rim = add_cyl(f"{name}_Rim", (0, 0, 0), radius, thickness, mat, rot=(math.pi * 0.5, 0, 0), verts=36)
    rim.parent = hub
    axle = add_cyl(f"{name}_Axle", (0, 0, 0), radius * 0.24, thickness * 1.9, dark_iron_mat, rot=(math.pi * 0.5, 0, 0), verts=20)
    axle.parent = hub
    tooth_w = (2.0 * math.pi * radius / teeth) * 0.45
    tooth_h = radius * 0.22
    for i in range(teeth):
        ang = (i / teeth) * math.pi * 2.0
        tx = math.cos(ang) * (radius + tooth_h * 0.34)
        tz = math.sin(ang) * (radius + tooth_h * 0.34)
        t_obj = add_box(f"{name}_T{i}", (tx, 0.0, tz), (tooth_h, thickness * 0.90, tooth_w), mat, rot=(0.0, -ang, 0.0))
        t_obj.parent = hub
    return hub


def orient_beam(obj, p1: Vector, p2: Vector, radius_scale: float = 1.0):
    diff = p2 - p1
    length = diff.length
    obj.location = (p1 + p2) * 0.5
    obj.scale = (radius_scale, radius_scale, max(0.0005, length))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = diff.normalized().to_track_quat("Z", "Y") if length > 1e-5 else Quaternion((1, 0, 0, 0))


# ============================================================================
# 6. BUILD 1678 DUTCH GOLDEN AGE OBSERVATORY ARCHITECTURE (RICH DEPTH & BOKEH!)
# ============================================================================
world = bpy.data.worlds.new("LeidenObservatoryNight")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.010, 0.018, 0.044, 1.0)
bg.inputs["Strength"].default_value = 0.65

# Varnished Dutch Walnut Workbench (top Z = 0.0)
add_box("WalnutWorkbench", (0.0, 0.14, -0.07), (4.8, 2.3, 0.14), walnut_mat, bevel_w=0.008)
add_box("BenchBrassTrim", (0.0, -1.015, -0.025), (4.82, 0.032, 0.044), brass_mat, bevel_w=0.004)

# Delft Stone Back Wall with Recessed Wainscoting, Molded Cornice & Tall Starlit Window
add_box("StoneWallLeft", (-1.95, 1.95, 1.55), (2.1, 0.24, 4.2), stone_mat)
add_box("StoneWallRight", (1.95, 1.95, 1.55), (2.1, 0.24, 4.2), stone_mat)
add_box("StoneWallLower", (0.0, 1.95, 0.22), (2.2, 0.24, 1.05), stone_mat)
add_box("StoneWallHeader", (0.0, 1.95, 2.75), (2.4, 0.24, 1.15), stone_mat)

# Carved Oak Wainscoting & Window Sill Trim along back wall (eliminates flat grey wall!)
add_box("OakWainscot", (0.0, 1.81, 0.36), (5.6, 0.06, 0.72), clock_wood_mat, bevel_w=0.006)
add_box("WindowSill", (0.0, 1.79, 0.74), (2.10, 0.14, 0.055), walnut_mat, bevel_w=0.006)
for sx in (-0.98, 0.98):
    add_cyl(f"StonePilaster_{sx}", (sx, 1.78, 1.48), 0.115, 3.6, stone_mat, verts=28)
    add_cyl(f"PilasterCap_{sx}", (sx, 1.78, 2.22), 0.145, 0.08, brass_mat, verts=28, bevel_w=0.006)

# Window Frame & Leaded Mullions
add_box("WindowMullionV", (0.0, 1.92, 1.48), (0.040, 0.045, 1.65), dark_iron_mat)
add_box("WindowMullionH1", (0.0, 1.92, 1.24), (1.92, 0.045, 0.036), dark_iron_mat)
add_box("WindowMullionH2", (0.0, 1.92, 1.74), (1.92, 0.045, 0.036), dark_iron_mat)

# Warm Dutch Scholar Bookshelves on Left & Right Walls (creates gorgeous f/2.6 background bokeh!)
book_mats = [book_red_mat, book_blue_mat, book_green_mat, clock_wood_mat]
for side_idx, bx in enumerate((-1.55, 1.58)):
    add_box(f"ShelfBack_{side_idx}", (bx, 1.72, 1.22), (0.86, 0.04, 1.42), clock_wood_mat)
    for sz in (0.76, 1.14, 1.52, 1.88):
        add_box(f"ShelfPlank_{side_idx}_{sz:.2f}", (bx, 1.60, sz), (0.88, 0.24, 0.035), walnut_mat)
    for row_idx, rz in enumerate((0.88, 1.26, 1.64)):
        for k in range(7):
            bh = 0.20 + ((k * 3 + row_idx) % 4) * 0.022
            bt = 0.042 + (k % 2) * 0.010
            add_box(
                f"Book_{side_idx}_{row_idx}_{k}",
                (bx - 0.32 + k * 0.105, 1.60, rz + bh * 0.5 - 0.10),
                (bt, 0.16, bh),
                book_mats[(k + row_idx + side_idx) % len(book_mats)],
                rot=(0.0, 0.04 if k == 6 else 0.0, 0.0),
            )

# Delicate Realistic Brass Candlesticks with Warm Teardrop Flames
candle_flames = []
for idx, (cx, cy) in enumerate([(-1.32, 0.58), (1.44, 0.42)]):
    add_cyl(f"CandleFoot_{idx}", (cx, cy, 0.014), 0.072, 0.028, brass_mat, bevel_w=0.004)
    add_cyl(f"CandleStem_{idx}", (cx, cy, 0.095), 0.016, 0.145, brass_mat)
    add_cyl(f"CandleCup_{idx}", (cx, cy, 0.172), 0.032, 0.018, brass_mat, bevel_w=0.003)
    add_cyl(f"CandleWax_{idx}", (cx, cy, 0.225), 0.017, 0.095, porcelain_mat)
    # Delicate tapered flame teardrop (NOT a giant white egg!)
    fl = add_cone(f"CandleFlame_{idx}", (cx, cy, 0.292), 0.011, 0.001, 0.044, candle_mat, verts=16)
    candle_flames.append(fl)

# Distant Pinpoint Night Stars outside rear arched window (tiny & realistic!)
for i in range(42):
    ang = i * 2.39996
    sx = math.sin(ang * 1.9) * 1.35
    sz = 0.90 + ((i * 0.23) % 1.45)
    r_star = 0.0045 + (i % 3) * 0.0022
    add_sphere(f"Star_{i}", (sx, 3.85, sz), r_star, star_warm_mat if i % 3 == 0 else star_cool_mat, segs=8, rings=6)


# ============================================================================
# 7. ZONE A (CENTER): CRYSTAL DOUBLE-SLIT RIPPLE TANK, FORWARD WAVELETS & CANNON
# ============================================================================
TANK_W, TANK_D, TANK_H = 0.88, 0.60, 0.135
TANK_Z0 = 0.032

add_box("TankBrassBase", (0.0, 0.0, 0.016), (TANK_W + 0.05, TANK_D + 0.05, 0.032), brass_mat, bevel_w=0.005)
add_box("TankFloorVelvet", (0.0, 0.0, TANK_Z0 + 0.004), (TANK_W - 0.02, TANK_D - 0.02, 0.008), velvet_mat)

wall_t = 0.012
add_box("TankWallFront", (0.0, -TANK_D * 0.5, TANK_Z0 + TANK_H * 0.5), (TANK_W, wall_t, TANK_H), glass_mat)
add_box("TankWallBack", (0.0, TANK_D * 0.5, TANK_Z0 + TANK_H * 0.5), (TANK_W, wall_t, TANK_H), glass_mat)
add_box("TankWallLeft", (-TANK_W * 0.5, 0.0, TANK_Z0 + TANK_H * 0.5), (wall_t, TANK_D, TANK_H), glass_mat)
add_box("TankWallRight", (TANK_W * 0.5, 0.0, TANK_Z0 + TANK_H * 0.5), (wall_t, TANK_D, TANK_H), glass_mat)

for sx in (-TANK_W * 0.5, TANK_W * 0.5):
    for sy in (-TANK_D * 0.5, TANK_D * 0.5):
        add_cyl(f"TankPost_{sx:.2f}_{sy:.2f}", (sx, sy, TANK_Z0 + TANK_H * 0.5), 0.012, TANK_H + 0.018, brass_mat, verts=24, bevel_w=0.002)

# Real 3D Animated Water Ripple Surface (64x42 subdivided plane + 3 native C++ Blender WAVE modifiers!)
WATER_Z = TANK_Z0 + 0.088
bpy.ops.mesh.primitive_grid_add(x_subdivisions=64, y_subdivisions=42, size=1.0, location=(0.0, 0.0, WATER_Z))
water_mesh = bpy.context.active_object
water_mesh.name = "HuygensRippleWater"
water_mesh.scale = (TANK_W - 0.028, TANK_D - 0.028, 1.0)
bpy.ops.object.transform_apply(scale=True)
for p in water_mesh.data.polygons:
    p.use_smooth = True
water_mesh.data.materials.append(water_mat)

wm_plane = water_mesh.modifiers.new("PlaneWave", "WAVE")
wm_plane.use_x = True
wm_plane.use_y = False
wm_plane.height = 0.011
wm_plane.width = 0.085
wm_plane.narrowness = 1.7
wm_plane.speed = 0.032
wm_plane.start_position_x = -0.40
wm_plane.start_position_y = 0.0

SLIT_X = -0.08
SLIT_Y_A = -0.080
SLIT_Y_B = +0.080
for idx, sy in enumerate((SLIT_Y_A, SLIT_Y_B)):
    wm = water_mesh.modifiers.new(f"HuygensSlitWave_{idx}", "WAVE")
    wm.use_x = True
    wm.use_y = True
    wm.height = 0.014
    wm.width = 0.078
    wm.narrowness = 1.6
    wm.speed = 0.035
    wm.start_position_x = SLIT_X
    wm.start_position_y = sy

# Machined Brass Double-Slit Aperture Wall (crisp physical slits at Y = ±0.080!)
add_box("SlitWallTop", (SLIT_X, 0.200, TANK_Z0 + 0.088), (0.016, 0.175, 0.145), brass_mat, bevel_w=0.002)
add_box("SlitWallMid", (SLIT_X, 0.0, TANK_Z0 + 0.088), (0.016, 0.105, 0.145), brass_mat, bevel_w=0.002)
add_box("SlitWallBot", (SLIT_X, -0.200, TANK_Z0 + 0.088), (0.016, 0.175, 0.145), brass_mat, bevel_w=0.002)

# 4 Delicate Incoming Plane-Wavefront Filaments (strictly inside left half of tank!)
plane_bars = []
for i in range(4):
    bar = add_cyl(
        f"PlaneWaveBar_{i}",
        (-0.36, 0.0, WATER_Z + 0.014),
        0.0038,
        TANK_D - 0.09,
        cyan_soft_mat,
        rot=(math.pi * 0.5, 0, 0),
        verts=16,
    )
    plane_bars.append(bar)

# 10 Forward-Propagating Huygens Secondary Wavelet Arcs (84 deg -> 100% mathematically inside tank walls!)
wavelet_arcs = []
for s_idx, sy in enumerate((SLIT_Y_A, SLIT_Y_B)):
    for k in range(5):
        mat_r = cyan_glow_mat if (k % 2 == 0) else gold_glow_mat
        arc = build_wavelet_arc_mesh(f"HuygensWavelet_S{s_idx}_{k}", mat_r, arc_deg=84.0, segs=32, half_width=0.014)
        arc.location = Vector((SLIT_X + 0.012, sy, WATER_Z + 0.014))
        wavelet_arcs.append((s_idx, k, sy, arc))

# Interference Detector Screen on Right Wall of Tank (X = +0.425) with 11 crisp alternating Cyan/Dark zebra fringes!
add_box("DetectorFrame", (0.435, 0.0, TANK_Z0 + 0.125), (0.016, TANK_D - 0.04, 0.22), brass_mat, bevel_w=0.003)
add_box("DetectorBackplate", (0.426, 0.0, TANK_Z0 + 0.125), (0.012, TANK_D - 0.07, 0.195), dark_iron_mat)
fringe_bars = []
for i in range(11):
    fy = -0.22 + i * 0.044
    is_bright = (i % 2 == 0)
    bar = add_box(
        f"InterferenceFringe_{i}",
        (0.418, fy, TANK_Z0 + 0.125),
        (0.008, 0.022, 0.175),
        cyan_glow_mat if is_bright else fringe_dark_mat,
    )
    fringe_bars.append((i, is_bright, bar))

# Beat 1 ONLY (0..4.45s): True 3D Sinusoidal Destructive Interference Demo above tank (NO grey floating ball!)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(-0.38, 0.0, 0.36))
wave_cancel_group = bpy.context.active_object
wave_cancel_group.name = "Beat1WaveCancelDemo"
sin_wave_a = build_sinusoid_ribbon("SinWaveGold", 0.72, 0.052, 3.0, 0.0, gold_glow_mat, segs=60, radius=0.0055)
sin_wave_a.location = Vector((0.0, -0.02, 0.0))
sin_wave_a.parent = wave_cancel_group
sin_wave_b = build_sinusoid_ribbon("SinWaveCyan", 0.72, 0.052, 3.0, math.pi, cyan_glow_mat, segs=60, radius=0.0055)
sin_wave_b.location = Vector((0.0, 0.02, 0.0))
sin_wave_b.parent = wave_cancel_group
# Dark cancellation fringe badge where out-of-phase waves flatten to darkness!
cancel_screen = add_box("CancelDarkCard", (0.74, 0.0, 0.0), (0.012, 0.16, 0.16), dark_iron_mat, bevel_w=0.002)
cancel_screen.parent = wave_cancel_group
cancel_rim = add_box("CancelDarkRim", (0.746, 0.0, 0.0), (0.008, 0.178, 0.178), brass_mat, bevel_w=0.002)
cancel_rim.parent = wave_cancel_group
for idx_s, sz_off in enumerate((-0.050, 0.0, 0.050)):
    st = add_box(
        f"CancelStripe_{idx_s}",
        (0.732, 0.0, sz_off),
        (0.006, 0.135, 0.022),
        fringe_dark_mat if idx_s == 1 else (gold_glow_mat if idx_s == 0 else cyan_glow_mat),
    )
    st.parent = wave_cancel_group

# Sculpted Antique Brass Optical Particle Emitter ("Newton's Cannon" — Smooth-Shaded & Detailed!)
CANNON_X, CANNON_Y = -0.66, -0.08
add_cyl("CannonBaseStep1", (CANNON_X, CANNON_Y, 0.012), 0.105, 0.024, brass_mat, verts=48, bevel_w=0.004)
add_cyl("CannonBaseStep2", (CANNON_X, CANNON_Y, 0.032), 0.075, 0.018, bronze_mat, verts=48, bevel_w=0.003)
add_cyl("CannonPillar", (CANNON_X, CANNON_Y, 0.115), 0.022, 0.155, brass_mat, verts=36)
add_cyl("CannonCollar", (CANNON_X, CANNON_Y, 0.175), 0.034, 0.020, bronze_mat, verts=36, bevel_w=0.003)
# Tapered optical barrel + brass reinforcing rings + glowing aperture lens!
add_cone("CannonBarrel", (CANNON_X + 0.02, CANNON_Y, 0.215), 0.046, 0.034, 0.24, brass_mat, rot=(0, math.pi * 0.5, 0), verts=48)
add_torus("CannonRingRear", (CANNON_X - 0.085, CANNON_Y, 0.215), 0.047, 0.008, bronze_mat, rot=(0, math.pi * 0.5, 0))
add_torus("CannonRingMid", (CANNON_X + 0.015, CANNON_Y, 0.215), 0.041, 0.007, bronze_mat, rot=(0, math.pi * 0.5, 0))
add_torus("CannonMuzzleLip", (CANNON_X + 0.140, CANNON_Y, 0.215), 0.036, 0.007, brass_mat, rot=(0, math.pi * 0.5, 0))
add_sphere("CannonRearKnob", (CANNON_X - 0.112, CANNON_Y, 0.215), 0.022, brass_mat)
add_cyl("CannonGlowLens", (CANNON_X + 0.136, CANNON_Y, 0.215), 0.027, 0.008, gold_glow_mat, rot=(0, math.pi * 0.5, 0), verts=32)

# 14 Rich Molten-Gold Newton "Light Bullets"
newton_bullets = []
for i in range(14):
    b = add_sphere(f"NewtonBullet_{i}", (-0.50, CANNON_Y, 0.215), 0.014, gold_bullet_mat, segs=18, rings=12)
    newton_bullets.append(b)


# ============================================================================
# 8. ZONE B (RIGHT): ICELAND SPAR CALCITE DOUBLE REFRACTION SET (BEAT 4 HERO!)
# ============================================================================
CALCITE_POS = Vector((0.96, -0.04, 0.0))
add_cyl("CalcitePedestalBase", (CALCITE_POS.x, CALCITE_POS.y, 0.022), 0.21, 0.044, brass_mat, verts=64, bevel_w=0.005)
add_cyl("CalciteVelvetPad", (CALCITE_POS.x, CALCITE_POS.y, 0.048), 0.165, 0.014, velvet_mat, verts=64)

# Authentic Sheared Faceted Iceland Spar Rhombohedron Crystal (IOR 1.658)
bpy.ops.mesh.primitive_cube_add(size=0.19, location=(CALCITE_POS.x, CALCITE_POS.y, 0.165))
calcite = bpy.context.active_object
calcite.name = "IcelandSparCrystal"
for v in calcite.data.vertices:
    v.co.x += v.co.z * 0.26
    v.co.y += v.co.z * 0.10
calcite.data.update()
bev = calcite.modifiers.new("CrystalBevel", "BEVEL")
bev.width = 0.007
bev.segments = 3
calcite.rotation_euler = Euler((0.0, 0.0, math.radians(14.0)))
calcite.data.materials.append(calcite_mat)

# Left Brass Beam Emitter & Right Slate Target Card (ALL framed cleanly together in Beat 4!)
add_cyl("CalciteEmitterFoot", (CALCITE_POS.x - 0.30, CALCITE_POS.y - 0.04, 0.012), 0.050, 0.024, brass_mat, verts=32, bevel_w=0.003)
add_cyl("CalciteEmitterStem", (CALCITE_POS.x - 0.30, CALCITE_POS.y - 0.04, 0.090), 0.012, 0.145, brass_mat, verts=24)
add_cone("CalciteEmitterBarrel", (CALCITE_POS.x - 0.28, CALCITE_POS.y - 0.04, 0.170), 0.024, 0.018, 0.095, brass_mat, rot=(0, math.pi * 0.5, 0.14), verts=32)

add_box("CalciteTargetStand", (CALCITE_POS.x + 0.295, CALCITE_POS.y + 0.035, 0.014), (0.065, 0.18, 0.028), brass_mat, rot=(0, 0, 0.14), bevel_w=0.003)
add_box("CalciteTargetCard", (CALCITE_POS.x + 0.295, CALCITE_POS.y + 0.035, 0.175), (0.012, 0.22, 0.24), dark_iron_mat, rot=(0, 0, 0.14), bevel_w=0.002)
add_box("CalciteTargetRim", (CALCITE_POS.x + 0.302, CALCITE_POS.y + 0.036, 0.175), (0.010, 0.236, 0.256), brass_mat, rot=(0, 0, 0.14), bevel_w=0.002)

calcite_in_beam = add_cyl("CalciteInBeam", (0, 0, 0), 0.0075, 1.0, gold_glow_mat, verts=20)
calcite_ord_beam = add_cyl("CalciteOrdBeam", (0, 0, 0), 0.0068, 1.0, gold_glow_mat, verts=20)
calcite_ext_beam = add_cyl("CalciteExtBeam", (0, 0, 0), 0.0068, 1.0, cyan_glow_mat, verts=20)
calcite_spot_o = add_sphere("CalciteSpotO", (CALCITE_POS.x + 0.284, CALCITE_POS.y + 0.005, 0.135), 0.018, gold_glow_mat, segs=18, rings=12)
calcite_spot_e = add_sphere("CalciteSpotE", (CALCITE_POS.x + 0.284, CALCITE_POS.y + 0.065, 0.225), 0.018, cyan_glow_mat, segs=18, rings=12)
calcite_shell_o = add_torus("CalciteWaveO", (CALCITE_POS.x, CALCITE_POS.y, 0.170), 0.062, 0.0024, gold_glow_mat, rot=(math.pi * 0.5, 0, 0))
calcite_shell_e = add_torus("CalciteWaveE", (CALCITE_POS.x, CALCITE_POS.y, 0.170), 0.072, 0.0024, cyan_glow_mat, rot=(math.pi * 0.42, 0.22, 0))


# ============================================================================
# 9. ZONE C (BEAT 5 HERO): 1656 PENDULUM CLOCK + TELESCOPE + SATURN HOLOGRAM!
# ============================================================================
# Positioned at X = -0.58 so Clock (left) + Saturn (center-right) BOTH fit 100% inside 9:16 portrait frame!
CLOCK_POS = Vector((-0.58, 0.54, 0.0))
add_box("ClockPlinth", (CLOCK_POS.x, CLOCK_POS.y, 0.036), (0.35, 0.22, 0.072), clock_wood_mat, bevel_w=0.006)
add_box("ClockBack", (CLOCK_POS.x, CLOCK_POS.y + 0.085, 0.50), (0.29, 0.032, 0.88), clock_wood_mat)
add_box("ClockSideL", (CLOCK_POS.x - 0.140, CLOCK_POS.y, 0.50), (0.030, 0.19, 0.88), clock_wood_mat, bevel_w=0.004)
add_box("ClockSideR", (CLOCK_POS.x + 0.140, CLOCK_POS.y, 0.50), (0.030, 0.19, 0.88), clock_wood_mat, bevel_w=0.004)
add_box("ClockCrown", (CLOCK_POS.x, CLOCK_POS.y, 0.96), (0.37, 0.24, 0.058), clock_wood_mat, bevel_w=0.006)
add_sphere("ClockFinial", (CLOCK_POS.x, CLOCK_POS.y, 1.02), 0.036, brass_mat, segs=24, rings=16)

# Porcelain Clock Dial + 12 Brass Hour Ticks + Bezel + Hands
DIAL_Z = 0.77
add_cyl("ClockDial", (CLOCK_POS.x, CLOCK_POS.y - 0.090, DIAL_Z), 0.106, 0.014, porcelain_mat, rot=(math.pi * 0.5, 0, 0), verts=48)
add_torus("ClockBezel", (CLOCK_POS.x, CLOCK_POS.y - 0.097, DIAL_Z), 0.108, 0.010, brass_mat, rot=(math.pi * 0.5, 0, 0))
for hr in range(12):
    ha = (hr / 12.0) * math.pi * 2.0
    add_box(
        f"DialTick_{hr}",
        (CLOCK_POS.x + math.sin(ha) * 0.086, CLOCK_POS.y - 0.098, DIAL_Z + math.cos(ha) * 0.086),
        (0.006 if hr % 3 != 0 else 0.010, 0.004, 0.018 if hr % 3 != 0 else 0.024),
        dark_iron_mat,
        rot=(0.0, ha, 0.0),
    )
clock_min_hand = add_box("ClockMinuteHand", (CLOCK_POS.x, CLOCK_POS.y - 0.104, DIAL_Z), (0.008, 0.005, 0.082), dark_iron_mat)

# 3 Interlocking Brass Escapement Gears
gear_1 = add_toothed_gear("BrassGearMain", (CLOCK_POS.x - 0.032, CLOCK_POS.y - 0.01, 0.56), 0.060, 0.013, 14, brass_mat)
gear_2 = add_toothed_gear("BrassGearMid", (CLOCK_POS.x + 0.060, CLOCK_POS.y - 0.01, 0.60), 0.044, 0.013, 10, brass_mat)
gear_3 = add_toothed_gear("BrassGearEsc", (CLOCK_POS.x + 0.012, CLOCK_POS.y - 0.01, 0.45), 0.040, 0.013, 10, brass_mat)

# Real Swinging Brass Pendulum (synced to 116 BPM clock ticks!)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(CLOCK_POS.x, CLOCK_POS.y - 0.045, 0.63))
pendulum_pivot = bpy.context.active_object
pendulum_pivot.name = "PendulumPivot"
p_rod = add_cyl("PendulumRod", (0.0, 0.0, -0.22), 0.005, 0.44, brass_mat, verts=16)
p_rod.parent = pendulum_pivot
p_bob = add_sphere("PendulumBob", (0.0, 0.0, -0.44), 0.048, brass_mat, segs=28, rings=18)
p_bob.scale = (1.0, 0.40, 1.0)
p_bob.parent = pendulum_pivot

# Authentic 17th-Century Dutch Brass Refractor Telescope on Tripod (replaces ugly floating lens!)
TEL_POS = Vector((-0.04, 0.52, 0.0))
for leg_i in range(3):
    la = math.radians(leg_i * 120.0 + 30.0)
    p_top = TEL_POS + Vector((0.0, 0.0, 0.36))
    p_foot = TEL_POS + Vector((math.cos(la) * 0.14, math.sin(la) * 0.14, 0.0))
    leg = add_cyl(f"TelescopeLeg_{leg_i}", (0, 0, 0), 0.009, 1.0, bronze_mat, verts=16)
    orient_beam(leg, p_foot, p_top, 1.0)
add_sphere("TelescopePivot", TEL_POS + Vector((0.0, 0.0, 0.365)), 0.024, brass_mat, segs=20, rings=14)
tel_tube = add_cone("TelescopeTube", (0, 0, 0), 0.026, 0.044, 0.54, brass_mat, verts=40)
orient_beam(tel_tube, TEL_POS + Vector((-0.14, -0.04, 0.25)), TEL_POS + Vector((0.20, 0.04, 0.55)), 1.0)
tel_collar = add_torus("TelescopeFrontRim", TEL_POS + Vector((0.20, 0.04, 0.55)), 0.045, 0.007, bronze_mat)
tel_collar.rotation_mode = "QUATERNION"
tel_collar.rotation_quaternion = tel_tube.rotation_quaternion

# Beat 5-6 Glowing Saturn + Layered Translucent Amber-Gold Rings + Titan Hologram (Center-Right Upper!)
SATURN_BASE = Vector((0.30, 0.44, 0.88))
bpy.ops.object.empty_add(type="PLAIN_AXES", location=SATURN_BASE)
saturn_group = bpy.context.active_object
saturn_group.name = "SaturnHologram"
saturn_planet = add_sphere("SaturnPlanet", (0.0, 0.0, 0.0), 0.135, saturn_mat, segs=40, rings=26)
saturn_planet.parent = saturn_group

bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0), rotation=(math.radians(24.0), math.radians(-16.0), 0.0))
rings_tilt = bpy.context.active_object
rings_tilt.name = "SaturnRingsTilt"
rings_tilt.parent = saturn_group

# 3 Concentric Layered Rings with crisp Cassini Division gap (rich amber-gold, ZERO white clipping!)
for r_idx, (maj_r, min_r, mat_ring) in enumerate([
    (0.205, 0.017, ring_amber_mat),
    (0.258, 0.026, ring_gold_mat),
    (0.320, 0.014, ring_icy_mat),
]):
    rg = add_torus(f"SaturnRing_{r_idx}", (0.0, 0.0, 0.0), maj_r, min_r, mat_ring, maj_seg=72, min_seg=14)
    rg.scale = (1.0, 1.0, 0.12)
    rg.parent = rings_tilt

titan_moon = add_sphere("MoonTitan", (0.375, 0.0, 0.0), 0.024, gold_glow_mat, segs=20, rings=14)
titan_moon.parent = rings_tilt


# ============================================================================
# 10. ZONE D (BEAT 6): MODERN PHOTONIC LEGACY — RUBY LASER, 5G ARCS & FIBER!
# ============================================================================
laser_beam = add_cyl("ModernLaserBeam", (0.0, 0.0, 0.48), 0.0065, 1.0, ruby_laser_mat, verts=20)
fiveg_rings = []
for i in range(4):
    fg = add_torus(
        f"FiveGWave_{i}",
        (0.0, 0.18, 0.54),
        0.18 + i * 0.08,
        0.0038,
        cyan_glow_mat if i % 2 == 0 else gold_glow_mat,
    )
    fg.rotation_euler = Euler((math.radians(74.0), 0.0, 0.0))
    fiveg_rings.append(fg)

fiber_beads = []
for i in range(8):
    fb = add_sphere(f"FiberPhoton_{i}", (0.0, 0.0, 0.48), 0.013, cyan_glow_mat if i % 2 == 0 else ruby_laser_mat, segs=14, rings=10)
    fiber_beads.append(fb)

# Subtle 3D Brass Title floating cleanly in upper window header during Beats 5-6!
bpy.ops.object.text_add(location=(0.0, 0.56, 1.24), rotation=(math.pi * 0.5, 0.0, 0.0))
title_txt = bpy.context.active_object
title_txt.name = "HuygensTitle3D"
title_txt.data.body = "HUYGENS' WAVE PRINCIPLE"
title_txt.data.align_x = "CENTER"
title_txt.data.size = 0.082
title_txt.data.extrude = 0.010
title_txt.data.bevel_depth = 0.0022
title_txt.data.materials.append(brass_mat)


# ============================================================================
# 11. REMBRANDT CHIAROSCURO 4-LIGHT STUDIO RIG + TRUE HORIZONTAL-FIT 9:16 CAMERA
# ============================================================================
# Warm Candlelight Key Area Light
bpy.ops.object.light_add(type="AREA", location=(-1.85, -1.65, 1.65))
key_light = bpy.context.active_object
key_light.data.energy = 230.0
key_light.data.size = 1.2
key_light.data.color = (1.0, 0.78, 0.48)
key_light.rotation_euler = Euler((math.radians(52), 0.0, math.radians(-42)))

# Cool Sapphire Moonlight Rim Light pouring through rear arched window
bpy.ops.object.light_add(type="AREA", location=(1.35, 2.35, 2.05))
rim_light = bpy.context.active_object
rim_light.data.energy = 290.0
rim_light.data.size = 1.5
rim_light.data.color = (0.42, 0.76, 1.0)
rim_light.rotation_euler = Euler((math.radians(58), 0.0, math.radians(146)))

# Warm Wall-Wash Accent lighting Delft masonry & Dutch bookshelves (kills flat dark wall!)
bpy.ops.object.light_add(type="AREA", location=(0.0, 0.85, 1.85))
wall_wash = bpy.context.active_object
wall_wash.data.energy = 95.0
wall_wash.data.size = 1.6
wall_wash.data.color = (1.0, 0.82, 0.58)
wall_wash.rotation_euler = Euler((math.radians(74), 0.0, 0.0))

# Subtle Cyan Ripple-Tank / Calcite Caustic Fill Light
bpy.ops.object.light_add(type="POINT", location=(0.25, -0.45, 0.44))
cyan_accent = bpy.context.active_object
cyan_accent.data.energy = 18.0
cyan_accent.data.color = (0.28, 0.82, 1.0)

# Camera configured with HORIZONTAL 36mm sensor fit -> wide cinema 9:16 portrait framing!
bpy.ops.object.empty_add(location=(0.0, 0.0, 0.28))
cam_target = bpy.context.active_object
cam_target.name = "CinemaCamTarget"

bpy.ops.object.camera_add(location=(0.0, -1.85, 0.65))
cam = bpy.context.active_object
cam.name = "CinemaCamera"
scene.camera = cam
cam.data.sensor_fit = "HORIZONTAL"
cam.data.sensor_width = 36.0
cam.data.lens = 38.0
cam.data.dof.use_dof = True
cam.data.dof.focus_object = cam_target
cam.data.dof.aperture_fstop = 3.2
cam.rotation_mode = "QUATERNION"


# ============================================================================
# 12. 6-BEAT CINEMA CAMERA GRAMMAR (VERIFIED 100% INSIDE 9:16 SAFE FRAME!)
# ============================================================================
def compute_beat_raw(beat_idx: int, t: float):
    """Raw camera pose (pos, tgt, lens, fstop, dutch_deg) for beat_idx in 0..5."""
    if beat_idx == 0:
        # BEAT 1 (0..4.5s): Curved low-angle hero push-in framing Cannon + Tank + 3D Wave Cancellation!
        u = clamp01(t / 4.5)
        pos = bezier_quad(Vector((-0.36, -1.96, 0.58)), Vector((-0.20, -1.80, 0.52)), Vector((-0.06, -1.62, 0.46)), u)
        tgt = vlerp(Vector((-0.20, 0.0, 0.28)), Vector((-0.10, 0.0, 0.28)), u)
        return pos, tgt, lerp(34.0, 37.0, u), 3.4, 0.0
    if beat_idx == 1:
        # BEAT 2 (4.5..9.5s): Newton's Cannon (4.5..6.85s) -> silky whip-pan into Huygens' Ripple Tank!
        pan_u = ease_out_quint(clamp01((t - 6.85) / 1.15))
        pos_newton = Vector((-0.68, -1.32, 0.44))
        pos_huygens = Vector((-0.06, -1.24, 0.78))
        pos = vlerp(pos_newton, pos_huygens, pan_u) + Vector((math.sin(pan_u * math.pi) * 0.12, -0.06 * math.sin(pan_u * math.pi), 0.0))
        tgt = vlerp(Vector((-0.44, -0.06, 0.22)), Vector((0.02, 0.0, 0.14)), pan_u)
        return pos, tgt, lerp(41.0, 38.0, pan_u), 2.8, 0.0
    if beat_idx == 2:
        # BEAT 3 (9.5..15.5s): Elevated Orbital Macro directly over Double-Slit Tank & 10 Forward Wavelet Arcs!
        u = clamp01((t - 9.5) / 6.0)
        ang = math.radians(-18.0 + 38.0 * u)
        rad = lerp(1.26, 1.16, math.sin(u * math.pi))
        pos = Vector((0.04 + math.sin(ang) * rad * 0.76, -math.cos(ang) * rad, lerp(0.88, 0.98, u)))
        tgt = Vector((0.04, 0.0, 0.14))
        return pos, tgt, lerp(38.0, 41.0, u), 2.8, 0.0
    if beat_idx == 3:
        # BEAT 4 (15.5..20.5s): True 3.8° Dutch-Tilt Macro framing Emitter + Calcite + BOTH Split Beams + Target Spots!
        u = clamp01((t - 15.5) / 5.0)
        ang = math.radians(-16.0 + 34.0 * u)
        rad = lerp(1.18, 1.24, u)
        pos = Vector((CALCITE_POS.x + math.sin(ang) * rad * 0.65, CALCITE_POS.y - math.cos(ang) * rad, lerp(0.46, 0.54, u)))
        tgt = Vector((CALCITE_POS.x + 0.01, CALCITE_POS.y, 0.20))
        shake = math.sin((t - 18.55) * 48.0) * math.exp(-(t - 18.55) * 6.5) * 0.022 if t >= 18.55 else 0.0
        pos.x += shake
        return pos, tgt, lerp(38.0, 41.0, u), 2.8, 3.8 * math.sin(u * math.pi)
    if beat_idx == 4:
        # BEAT 5 (20.5..25.5s): Rising Crane Reveal framing FULL Pendulum Clock (left) + Telescope + Saturn (right)!
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(
            Vector((-0.24, -1.78, 0.56)),
            Vector((-0.14, -1.90, 0.70)),
            Vector((-0.06, -1.98, 0.82)),
            u,
        )
        tgt = vlerp(Vector((-0.14, 0.44, 0.54)), Vector((-0.08, 0.44, 0.62)), u)
        return pos, tgt, lerp(38.0, 36.0, u), 3.4, 0.0
    # BEAT 6 (25.5..30.0s): Grand Hero Wide Pull-Back — Entire Glowing Observatory + Saturn + Ruby Laser & 5G!
    u = clamp01((t - 25.5) / 4.5)
    pos = bezier_quad(
        Vector((-0.06, -1.98, 0.82)),
        Vector((-0.02, -2.20, 0.78)),
        Vector((0.0, -2.38, 0.74)),
        u,
    )
    tgt = vlerp(Vector((-0.08, 0.34, 0.58)), Vector((0.0, 0.22, 0.50)), u)
    return pos, tgt, lerp(36.0, 32.0, u), 3.8, 0.0


BEAT_BOUNDS = [0.0, 4.5, 9.5, 15.5, 20.5, 25.5, 30.0]


def compute_camera_state(t: float):
    """Return smooth single-take cinema camera pose with 0.42s easeOutQuint whip-pans synced to SFX!"""
    sway_x = math.sin(t * 1.15) * 0.011 + math.sin(t * 2.40) * 0.004
    sway_z = math.cos(t * 0.92) * 0.009

    b_idx = 5
    for i in range(6):
        if BEAT_BOUNDS[i] <= t < BEAT_BOUNDS[i + 1]:
            b_idx = i
            break

    pos, tgt, lens, fstop, dutch = compute_beat_raw(b_idx, t)

    # Silky 0.42s easeOutQuint whip-pan transition from previous beat right on the SFX whip_whoosh!
    whip_dur = 0.42
    dt_in = t - BEAT_BOUNDS[b_idx]
    if b_idx > 0 and dt_in < whip_dur:
        w = ease_out_quint(dt_in / whip_dur)
        p0, t0, l0, f0, d0 = compute_beat_raw(b_idx - 1, BEAT_BOUNDS[b_idx])
        pos = vlerp(p0, pos, w)
        tgt = vlerp(t0, tgt, w)
        lens = lerp(l0, lens, w)
        fstop = lerp(f0, fstop, w)
        dutch = lerp(d0, dutch, w)

    pos.x += sway_x
    pos.z += sway_z
    return pos, tgt, lens, fstop, dutch


# ============================================================================
# 13. KEYFRAME ALL 900 FRAMES DETERMINISTICALLY
# ============================================================================
BPM = 116.0
for f in range(1, TOTAL_FRAMES + 1):
    t = (f - 1) / FPS

    # 1. Exact Camera Position, Look-At Quaternion + True Dutch Roll, Lens & Focus
    c_pos, c_tgt, c_lens, c_fstop, c_dutch = compute_camera_state(t)
    cam.location = c_pos
    cam_target.location = c_tgt
    cam.rotation_quaternion = look_at_quat(c_pos, c_tgt, c_dutch)
    cam.data.lens = c_lens
    cam.data.dof.aperture_fstop = c_fstop
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
    cam_target.keyframe_insert("location", frame=f)
    cam.data.keyframe_insert("lens", frame=f)
    cam.data.keyframe_insert(data_path="dof.aperture_fstop", frame=f)

    # 2. Delicate Practical Candle Flicker
    for idx, fl in enumerate(candle_flames):
        flick = 1.0 + 0.12 * math.sin(t * 21.0 + idx * 2.1) + 0.06 * math.sin(t * 35.0)
        fl.scale = (1.0, 1.0, flick)
        fl.keyframe_insert("scale", frame=f)

    # 3. Beat 1 ONLY: 3D Sinusoidal Wave Cancellation Demo (cleanly gone by t=4.45s!)
    w1_vis = smoothstep(0.6, 1.4, t) * (1.0 - smoothstep(4.0, 4.45, t))
    w1_sc = max(0.0001, w1_vis)
    wave_cancel_group.scale = Vector((w1_sc, w1_sc, w1_sc))
    sin_wave_a.location.z = math.sin(t * 9.0) * 0.008
    sin_wave_b.location.z = -math.sin(t * 9.0) * 0.008
    wave_cancel_group.keyframe_insert("scale", frame=f)
    sin_wave_a.keyframe_insert("location", frame=f)
    sin_wave_b.keyframe_insert("location", frame=f)

    # 4. Newton's 14 Golden Light Bullets (straight stream Beats 1-2, hover & SHATTER at t=18.55s in Beat 4!)
    for i, b in enumerate(newton_bullets):
        if t < 7.2:
            fade_b = 1.0 - smoothstep(6.6, 7.2, t)
            phase = (t * 1.35 + i / 14.0) % 1.0
            if phase < 0.74:
                u_b = phase / 0.74
                bx = lerp(-0.50, -0.10, u_b)
                by = CANNON_Y + math.sin(i * 1.4) * 0.012
                bz = 0.215
            else:
                u_r = (phase - 0.74) / 0.26
                bx = lerp(-0.10, -0.24, u_r)
                by = CANNON_Y + ((i % 2) * 2 - 1) * u_r * 0.13
                bz = 0.215 + u_r * 0.07 - u_r * u_r * 0.12
            b.location = Vector((bx, by, bz))
            sc = max(0.0001, fade_b)
            b.scale = Vector((sc, sc, sc))
        elif 15.5 <= t < 20.4:
            ang = (i / 14.0) * math.pi * 2.0
            if t < 18.55:
                vis = smoothstep(15.6, 16.2, t)
                bx = CALCITE_POS.x - 0.10 + math.cos(ang + t * 2.2) * 0.065
                by = CALCITE_POS.y - 0.04 + math.sin(ang + t * 2.2) * 0.045
                bz = 0.36 + math.sin(t * 4.2 + i) * 0.015
                sc = max(0.0001, vis * 0.85)
            else:
                dt = t - 18.55
                sp = clamp01(dt / 1.60)
                rad = 0.065 + sp * (0.42 + (i % 4) * 0.10)
                bx = CALCITE_POS.x - 0.04 + math.cos(ang * 1.3) * rad
                by = CALCITE_POS.y - 0.04 + math.sin(ang * 1.7) * rad * 0.65
                bz = 0.36 + sp * (0.24 - sp * 0.48)
                sc = max(0.0001, (1.0 - sp) * 0.90)
            b.location = Vector((bx, by, bz))
            b.scale = Vector((sc, sc, sc))
        else:
            b.scale = Vector((0.0001, 0.0001, 0.0001))
        b.keyframe_insert("location", frame=f)
        b.keyframe_insert("scale", frame=f)

    # 5. Plane-Wavefront Filaments & 10 Forward 84-Deg Huygens Wavelet Arcs (STRICTLY INSIDE 0.60m TANK!)
    wave_vis = smoothstep(6.9, 7.6, t) * (1.0 - smoothstep(15.2, 15.6, t)) + 0.55 * smoothstep(25.2, 26.0, t)
    for i, bar in enumerate(plane_bars):
        u_p = (t * 0.82 + i * 0.25) % 1.0
        bar.location = Vector((lerp(-0.40, SLIT_X - 0.024, u_p), 0.0, WATER_Z + 0.014))
        sc = max(0.0001, wave_vis * math.sin(u_p * math.pi))
        bar.scale = Vector((sc, sc, 1.0 if sc > 0.01 else 0.0001))
        bar.keyframe_insert("location", frame=f)
        bar.keyframe_insert("scale", frame=f)

    # Max wavelet radius 0.30m @ ±42° -> max_Y = 0.080 + 0.30*0.669 = 0.281m < 0.292m inner glass wall!
    for s_idx, k, sy, arc in wavelet_arcs:
        u_w = (t * 0.72 + k / 5.0) % 1.0
        r_cur = lerp(0.024, 0.300, u_w)
        fade_arc = math.sin(u_w * math.pi) ** 0.55
        if wave_vis > 0.02 and fade_arc > 0.04:
            arc.scale = Vector((r_cur, r_cur, max(0.05, 0.25 * fade_arc * wave_vis)))
        else:
            arc.scale = Vector((0.0001, 0.0001, 0.0001))
        arc.keyframe_insert("scale", frame=f)

    # 6. Detector Screen Zebra Interference Fringes
    fringe_boost = 0.65 + 0.35 * smoothstep(9.5, 11.2, t) + 0.14 * math.sin(t * 7.0)
    for i, is_bright, bar in fringe_bars:
        if is_bright:
            bar.scale = Vector((1.0, 1.0, fringe_boost))
            bar.keyframe_insert("scale", frame=f)

    # 7. Beat 4 Iceland Spar Calcite Double Refraction (1 Gold Ray -> Gold O-Ray + Cyan E-Ray!)
    c_vis = smoothstep(15.2, 15.9, t)
    p_emit = Vector((CALCITE_POS.x - 0.24, CALCITE_POS.y - 0.035, 0.170))
    p_entry = Vector((CALCITE_POS.x - 0.08, CALCITE_POS.y - 0.010, 0.170))
    p_exit_o = Vector((CALCITE_POS.x + 0.284, CALCITE_POS.y + 0.005, 0.135))
    p_exit_e = Vector((CALCITE_POS.x + 0.284, CALCITE_POS.y + 0.065, 0.225))
    split_u = smoothstep(15.9, 17.0, t)

    orient_beam(calcite_in_beam, p_emit, p_entry, max(0.0001, c_vis))
    orient_beam(calcite_ord_beam, p_entry, vlerp(p_entry, p_exit_o, split_u), max(0.0001, c_vis * split_u))
    orient_beam(calcite_ext_beam, p_entry, vlerp(p_entry, p_exit_e, split_u), max(0.0001, c_vis * split_u))
    for ob in (calcite_in_beam, calcite_ord_beam, calcite_ext_beam):
        ob.keyframe_insert("location", frame=f)
        ob.keyframe_insert("rotation_quaternion", frame=f)
        ob.keyframe_insert("scale", frame=f)

    sp_sc = max(0.0001, smoothstep(16.7, 17.3, t))
    calcite_spot_o.scale = Vector((sp_sc, sp_sc, sp_sc))
    calcite_spot_e.scale = Vector((sp_sc, sp_sc, sp_sc))
    calcite_spot_o.keyframe_insert("scale", frame=f)
    calcite_spot_e.keyframe_insert("scale", frame=f)

    sh_u = (t * 0.85) % 1.0
    sh_sc = max(0.0001, c_vis * split_u * (0.35 + sh_u * 0.85))
    calcite_shell_o.scale = Vector((sh_sc, sh_sc, sh_sc))
    calcite_shell_e.scale = Vector((sh_sc * 1.18, sh_sc * 0.85, sh_sc))
    calcite_shell_o.keyframe_insert("scale", frame=f)
    calcite_shell_e.keyframe_insert("scale", frame=f)

    # 8. Huygens 1656 Precision Brass Pendulum Clock (real 116 BPM harmonic swing + meshing gears!)
    swing_ang = math.radians(17.0) * math.sin(math.pi * t * (BPM / 60.0))
    pendulum_pivot.rotation_euler = Euler((0.0, swing_ang, 0.0))
    pendulum_pivot.keyframe_insert("rotation_euler", frame=f)

    gear_1.rotation_euler = Euler((0.0, t * 0.85, 0.0))
    gear_2.rotation_euler = Euler((0.0, -t * 0.85 * (14.0 / 10.0), 0.0))
    gear_3.rotation_euler = Euler((0.0, -t * 0.85 * (14.0 / 10.0) + 0.3, 0.0))
    for g in (gear_1, gear_2, gear_3):
        g.keyframe_insert("rotation_euler", frame=f)

    clock_min_hand.rotation_euler = Euler((0.0, t * 0.42, 0.0))
    clock_min_hand.keyframe_insert("rotation_euler", frame=f)

    # 9. Beat 5-6 Glowing Saturn + Layered Amber Rings + Titan Hologram!
    sat_u = smoothstep(20.5, 22.1, t)
    sat_sc = max(0.0001, sat_u)
    saturn_group.scale = Vector((sat_sc, sat_sc, sat_sc))
    saturn_group.location = Vector((SATURN_BASE.x, SATURN_BASE.y, lerp(0.68, SATURN_BASE.z, sat_u) + math.sin(t * 1.8) * 0.014))
    saturn_planet.rotation_euler = Euler((0.0, 0.0, t * 0.55))
    rings_tilt.rotation_euler = Euler((math.radians(24.0), math.radians(-16.0), t * 0.38))
    saturn_group.keyframe_insert("scale", frame=f)
    saturn_group.keyframe_insert("location", frame=f)
    saturn_planet.keyframe_insert("rotation_euler", frame=f)
    rings_tilt.keyframe_insert("rotation_euler", frame=f)

    # 10. Beat 6 Modern Photonic Legacy: Ruby Laser + Sleek 5G Wave Arcs + Fiber-Optic Photons!
    mod_u = smoothstep(25.5, 26.6, t)
    p_l0 = Vector((-0.95, 0.16, 0.46))
    p_l1 = vlerp(p_l0, Vector((0.95, 0.16, 0.46)), mod_u)
    orient_beam(laser_beam, p_l0, p_l1, max(0.0001, mod_u))
    laser_beam.keyframe_insert("location", frame=f)
    laser_beam.keyframe_insert("rotation_quaternion", frame=f)
    laser_beam.keyframe_insert("scale", frame=f)

    for i, fg in enumerate(fiveg_rings):
        if mod_u > 0.01:
            u_5g = ((t - 25.5) * 0.68 + i * 0.25) % 1.0
            sc_5g = mod_u * (0.45 + u_5g * 1.55)
            th_5g = max(0.001, mod_u * (1.0 - u_5g * 0.78))
            fg.scale = Vector((sc_5g, sc_5g, th_5g))
        else:
            fg.scale = Vector((0.0001, 0.0001, 0.0001))
        fg.keyframe_insert("scale", frame=f)

    for i, fb in enumerate(fiber_beads):
        if mod_u > 0.01:
            u_fb = ((t - 25.5) * 0.65 + i / 8.0) % 1.0
            fx = lerp(-0.78, 0.78, u_fb)
            fy = 0.06 + math.sin(u_fb * math.pi * 2.0) * 0.18
            fz = 0.46 + math.cos(u_fb * math.pi * 2.0) * 0.12
            fb.location = Vector((fx, fy, fz))
            fb.scale = Vector((mod_u, mod_u, mod_u))
        else:
            fb.scale = Vector((0.0001, 0.0001, 0.0001))
        fb.keyframe_insert("location", frame=f)
        fb.keyframe_insert("scale", frame=f)

    # 3D Brass Title cleanly framed in upper window during Beats 5-6!
    tit_u = smoothstep(21.4, 22.6, t)
    title_txt.scale = Vector((max(0.0001, tit_u), max(0.0001, tit_u), max(0.0001, tit_u)))
    title_txt.keyframe_insert("scale", frame=f)


# ============================================================================
# 14. EXECUTE RENDER (QA 6-BEAT PREVIEW OR PRODUCTION FRAME RANGE!)
# ============================================================================
if ARGS.qa_only:
    # 6 iconic hero frames (1 per beat) showing every key moment cleanly!
    qa_frames = [60, 165, 375, 535, 705, 845]
    print(f"[Huygens V2 QA] Rendering 6 Beat keyframes {qa_frames} to {OUT_DIR} ...")
    for idx, qf in enumerate(qa_frames, 1):
        scene.frame_set(qf)
        scene.render.filepath = os.path.join(OUT_DIR, f"qa_beat{idx}_f{qf:04d}.png")
        bpy.ops.render.render(write_still=True)
    print("[Huygens V2 QA] All 6 Beat keyframes complete!")
else:
    print(
        f"[Huygens V2 Production] Rendering frames {scene.frame_start}..{scene.frame_end} "
        f"({RES_X}x{RES_Y}, {ARGS.samples} samples) -> {OUT_DIR} ..."
    )
    bpy.ops.render.render(animation=True)
    print(f"[Huygens V2 Production] Batch {scene.frame_start}..{scene.frame_end} complete!")
