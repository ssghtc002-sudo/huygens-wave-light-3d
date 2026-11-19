"""
Christiaan Huygens (1678) — How Waves Defeated Light Bullets
V2 Cinema Audio, 116 BPM Dutch Golden Age Baroque & Photonic Wave Score + 24 Spatial Foley SFX
+ Exact Kokoro Word-Synced Boxless ASS Karaoke Subtitle Generator

Run with:
    /home/antigravity/shorts-engine/.venv/bin/python audio/generate_audio.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

SHORTS_ENGINE_ROOT = Path("/home/antigravity/shorts-engine")
if str(SHORTS_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(SHORTS_ENGINE_ROOT))

from audio.dspCore import (
    CHANNELS,
    DURATION,
    FPS,
    SR,
    TOTAL_FRAMES,
    TOTAL_SAMPLES,
    add_stereo_event,
    adsr_env,
    butter_filter,
    master_and_export,
    midi_to_hz,
    pan_stereo,
    peaking_eq,
    synthesize_dialogue,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMELINE_PATH = PROJECT_ROOT / "config" / "timeline.json"

BPM = 116.0
BEAT = 60.0 / BPM
EIGHTH = BEAT * 0.5
SIXTEENTH = BEAT * 0.25

# Natural pronunciation guidance for Kokoro TTS while subtitles preserve exact spelling
TTS_OVERRIDES: dict[str, str] = {
    "L1": "Everyone swore light was tiny bullets, until two beams crossed and created darkness.",
    "L2": "Light travels straight! Mere particles!",
    "L3": "Then watch it bend like ripples, Sir Isaac.",
    "L4": "In sixteen seventy-eight, Christian Hoy-gens proved every point on a wavefront sparks fresh baby ripples, forming interference.",
    "L5": "Inside Iceland spar crystal, one beam splits in two!",
    "L6": "Impossible! Bullets don't ripple!",
    "L7": "He also built the pendulum clock and unveiled Saturn's rings with wave-honed lenses.",
    "L8": "Lasers, Wi-Fi, and five-G today? All surfing Hoy-gens' waves.",
}


def add_chamber_reverb(stereo: np.ndarray, mix: float = 0.14) -> np.ndarray:
    """Short warm Dutch wood-and-stone observatory early reflections."""
    out = np.copy(stereo)
    taps = [(0.017, 0.32, 0.18), (0.031, 0.22, 0.30), (0.049, 0.15, 0.12), (0.071, 0.10, 0.20)]
    wet = np.zeros_like(stereo)
    for dt, gl, gr in taps:
        d = int(dt * SR)
        if d < len(stereo):
            wet[d:, 0] += stereo[:-d, 0] * gl
            wet[d:, 1] += stereo[:-d, 1] * gr
    wet = butter_filter(wet, "low", 4600.0, order=2)
    return (out * (1.0 - mix * 0.35) + wet * mix).astype(np.float32)


# ============================================================================
# INSTRUMENTS FOR 116 BPM DUTCH BAROQUE + PHOTONIC WAVE CINEMA SCORE
# ============================================================================
def synth_harpsichord(freq: float, dur: float, vel: float = 0.22) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    wave = (
        0.55 * np.sin(2 * math.pi * freq * t)
        + 0.35 * np.sin(2 * math.pi * 2 * freq * t)
        + 0.22 * np.sin(2 * math.pi * 3 * freq * t)
        + 0.14 * np.sin(2 * math.pi * 4 * freq * t)
        + 0.08 * np.sin(2 * math.pi * 6 * freq * t)
    )
    env = np.exp(-t * 5.8) * np.minimum(1.0, t / 0.004)
    return (wave * env * vel).astype(np.float32)


def synth_cello_pizz(freq: float, dur: float, vel: float = 0.32) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    wave = (
        0.78 * np.sin(2 * math.pi * freq * t)
        + 0.28 * np.sin(2 * math.pi * 2 * freq * t)
        + 0.10 * np.sin(2 * math.pi * 3 * freq * t)
    )
    env = np.exp(-t * 7.2) * np.minimum(1.0, t / 0.006)
    return butter_filter((wave * env * vel).astype(np.float32), "low", 950.0, order=2)


def synth_glass_harp(freq: float, dur: float, vel: float = 0.18) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    vib = 1.0 + 0.003 * np.sin(2 * math.pi * 5.2 * t)
    phase = 2 * math.pi * freq * np.cumsum(vib) / SR
    wave = 0.80 * np.sin(phase) + 0.16 * np.sin(2.76 * phase) * np.exp(-t * 4.0)
    env = adsr_env(n, attack=0.025, decay=0.18, sustain=0.55, release=min(0.35, dur * 0.45))
    return (wave * env * vel).astype(np.float32)


def synth_warm_strings(freqs: list[float], dur: float, vel: float = 0.16) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out_l = np.zeros(n, dtype=np.float32)
    out_r = np.zeros(n, dtype=np.float32)
    env = adsr_env(n, attack=min(0.42, dur * 0.22), decay=0.25, sustain=0.82, release=min(0.55, dur * 0.26))
    for i, f in enumerate(freqs):
        det_l = 1.0 - 0.0022 * ((i % 2) * 2 - 1)
        det_r = 1.0 + 0.0022 * ((i % 2) * 2 - 1)
        s_l = 0.65 * np.sin(2 * math.pi * f * det_l * t) + 0.24 * np.sin(4 * math.pi * f * det_l * t) + 0.11 * np.sin(6 * math.pi * f * det_l * t)
        s_r = 0.65 * np.sin(2 * math.pi * f * det_r * t) + 0.24 * np.sin(4 * math.pi * f * det_r * t) + 0.11 * np.sin(6 * math.pi * f * det_r * t)
        out_l += s_l
        out_r += s_r
    norm = max(1, len(freqs))
    stereo = np.stack([out_l / norm, out_r / norm], axis=-1) * env[:, None] * vel
    return butter_filter(stereo.astype(np.float32), "low", 3400.0, order=2)


def synth_french_horn(freq: float, dur: float, vel: float = 0.24) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    vib = 1.0 + 0.004 * np.sin(2 * math.pi * 4.6 * t) * np.minimum(1.0, t / 0.25)
    ph = 2 * math.pi * freq * np.cumsum(vib) / SR
    wave = 0.58 * np.sin(ph) + 0.34 * np.sin(2 * ph) + 0.22 * np.sin(3 * ph) + 0.10 * np.sin(4 * ph)
    env = adsr_env(n, attack=0.065, decay=0.14, sustain=0.78, release=min(0.28, dur * 0.32))
    return butter_filter((wave * env * vel).astype(np.float32), "low", 2100.0, order=2)


def synth_clock_tick(high: bool = True, vel: float = 0.16) -> np.ndarray:
    n = int(0.055 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f1 = 2450.0 if high else 1880.0
    f2 = 4120.0 if high else 3180.0
    click = (np.sin(2 * math.pi * f1 * t) + 0.5 * np.sin(2 * math.pi * f2 * t)) * np.exp(-t * 110.0)
    return (click * vel).astype(np.float32)


def compose_bgm() -> np.ndarray:
    bgm = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # 1. Steady 17th-century Dutch Pendulum Clock pulse during Beats 1-2 & Beat 5!
    for k in range(int(9.5 / BEAT)):
        t_tick = k * BEAT
        add_stereo_event(bgm, pan_stereo(synth_clock_tick(k % 2 == 0, vel=0.14), -0.32 if k % 2 == 0 else -0.20), t_tick)
    for k in range(int((25.5 - 20.5) / BEAT)):
        t_tick = 20.5 + k * BEAT
        add_stereo_event(bgm, pan_stereo(synth_clock_tick(k % 2 == 0, vel=0.18), -0.36 if k % 2 == 0 else -0.24), t_tick)

    # 2. Beats 1-2 (0.0-9.5s, D Minor): Suspenseful Baroque Harpsichord + Cello Pizzicato
    dm_arp = [62, 65, 69, 74, 69, 65, 64, 69, 61, 64, 69, 73, 69, 64, 62, 65]
    bass_b12 = [38, 38, 45, 45, 46, 46, 45, 45, 38, 38, 41, 41, 43, 43, 45, 45]
    step_idx = 0
    t_cur = 0.0
    while t_cur < 9.40:
        midi = dm_arp[step_idx % len(dm_arp)]
        pan = math.sin(step_idx * 0.7) * 0.35
        add_stereo_event(bgm, pan_stereo(synth_harpsichord(midi_to_hz(midi), 0.38, vel=0.19), pan), t_cur)
        if step_idx % 2 == 0:
            bm = bass_b12[(step_idx // 2) % len(bass_b12)]
            add_stereo_event(bgm, pan_stereo(synth_cello_pizz(midi_to_hz(bm), 0.45, vel=0.34), -0.10), t_cur)
        t_cur += EIGHTH
        step_idx += 1

    add_stereo_event(bgm, synth_warm_strings([midi_to_hz(m) for m in (50, 57, 62, 65)], 4.5, vel=0.18), 0.0)
    add_stereo_event(bgm, synth_warm_strings([midi_to_hz(m) for m in (46, 53, 58, 62)], 4.8, vel=0.19), 4.6)

    # 3. Beat 3 (9.5-15.5s, F Major / Aqueous Shimmer): Rippling Glass Harp + Rising Wave Pad!
    add_stereo_event(bgm, synth_warm_strings([midi_to_hz(m) for m in (53, 60, 65, 69)], 3.2, vel=0.22), 9.5)
    add_stereo_event(bgm, synth_warm_strings([midi_to_hz(m) for m in (55, 62, 67, 72)], 3.0, vel=0.24), 12.6)
    ripple_notes = [65, 69, 72, 77, 76, 72, 69, 74, 79, 76, 72, 81]
    t_cur = 9.55
    r_idx = 0
    while t_cur < 15.35:
        m = ripple_notes[r_idx % len(ripple_notes)]
        pan = math.sin(r_idx * 0.9) * 0.55
        add_stereo_event(bgm, pan_stereo(synth_glass_harp(midi_to_hz(m), 0.65, vel=0.18), pan), t_cur)
        t_cur += SIXTEENTH * 1.5
        r_idx += 1

    # 4. Beat 4 (15.5-20.5s): Crystal Proof Tension -> Crisp Orchestral Drop at 18.45s for Newton's Shock!
    add_stereo_event(bgm, synth_warm_strings([midi_to_hz(m) for m in (50, 57, 62, 66)], 2.85, vel=0.23), 15.55)
    for k, m in enumerate([69, 74, 78, 81, 78, 74, 81, 86]):
        t_n = 15.6 + k * EIGHTH * 1.15
        if t_n < 18.30:
            add_stereo_event(bgm, pan_stereo(synth_harpsichord(midi_to_hz(m), 0.40, vel=0.21), (k % 2 - 0.5) * 0.5), t_n)
    # 18.40..19.05 orchestral hush for Newton's "Impossible! Bullets don't ripple!", then triumphant pickup at 19.15s!
    for k, m in enumerate([62, 66, 69, 74]):
        add_stereo_event(bgm, pan_stereo(synth_cello_pizz(midi_to_hz(m - 12), 0.35, vel=0.35), 0.0), 19.20 + k * EIGHTH)
        add_stereo_event(bgm, pan_stereo(synth_harpsichord(midi_to_hz(m), 0.36, vel=0.22), 0.25), 19.20 + k * EIGHTH)

    # 5. Beats 5-6 (20.5-30.0s, Grand D Major Climax -> Futuristic Photonic Resolution!)
    add_stereo_event(bgm, synth_warm_strings([midi_to_hz(m) for m in (50, 57, 62, 66, 74)], 4.8, vel=0.28), 20.5)
    add_stereo_event(bgm, synth_warm_strings([midi_to_hz(m) for m in (55, 62, 67, 71, 79)], 2.3, vel=0.28), 25.3)
    add_stereo_event(bgm, synth_warm_strings([midi_to_hz(m) for m in (50, 57, 62, 66, 74)], 2.4, vel=0.30), 27.5)

    # Soaring French Horn heroic melody
    horn_melody = [
        (20.65, 62, 0.70), (21.40, 66, 0.50), (21.95, 69, 0.90),
        (22.90, 74, 1.10), (24.05, 73, 0.55), (24.65, 74, 0.75),
        (25.60, 79, 0.85), (26.50, 78, 0.65), (27.20, 74, 1.45),
    ]
    for t_h, m_h, d_h in horn_melody:
        add_stereo_event(bgm, pan_stereo(synth_french_horn(midi_to_hz(m_h), d_h, vel=0.22), -0.12), t_h)

    # Sparkling photonic glass harp finale arpeggio
    finale_notes = [74, 78, 81, 86, 81, 78, 79, 83, 86, 91, 86, 81, 74, 86]
    for k, m in enumerate(finale_notes):
        t_f = 25.65 + k * (EIGHTH * 1.1)
        if t_f < 29.4:
            add_stereo_event(bgm, pan_stereo(synth_glass_harp(midi_to_hz(m), 0.75, vel=0.17), math.sin(k) * 0.48), t_f)

    # Warm Dutch hall reverb & clean tail fade
    bgm = add_chamber_reverb(bgm, mix=0.20)
    tail = int(0.45 * SR)
    bgm[-tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)[:, None]
    return bgm


# ============================================================================
# 24 SPATIAL STEREO FOLEY SFX SYNTHESIZER
# ============================================================================
def synth_sfx(kind: str, pan: float) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(kind)) % 100000)

    if kind == "whip_whoosh":
        n = int(0.38 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        noise = rng.standard_normal(n).astype(np.float32)
        env = np.sin(math.pi * t / 0.38) ** 1.4
        bp = butter_filter(noise, "band", (260.0, 2600.0), order=2)
        tone = np.sin(2 * math.pi * (160 + 480 * (t / 0.38)) * t) * 0.18
        return pan_stereo((bp * 0.42 + tone) * env * 0.75, pan)

    if kind == "brass_cannon_ping":
        n = int(0.45 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        w = (np.sin(2 * math.pi * 680 * t) + 0.5 * np.sin(2 * math.pi * 1540 * t)) * np.exp(-t * 11.0)
        thump = np.sin(2 * math.pi * (140 * np.exp(-t * 14.0)) * t) * np.exp(-t * 16.0) * 0.6
        return pan_stereo((w * 0.45 + thump) * 0.85, pan)

    if kind == "bullet_ricochet":
        n = int(0.36 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f = 1850.0 * np.exp(-t * 4.2) + 420.0
        ph = 2 * math.pi * np.cumsum(f) / SR
        w = np.sin(ph) * np.exp(-t * 8.5)
        return pan_stereo(w * 0.55, pan)

    if kind in ("wave_cancel_hum", "dark_fringe_pulse", "interference_resonance"):
        n = int(0.95 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f0 = 146.83 if kind != "interference_resonance" else 220.0
        beat_mod = 0.5 + 0.5 * np.cos(2 * math.pi * 6.5 * t)
        w = (np.sin(2 * math.pi * f0 * t) + 0.45 * np.sin(2 * math.pi * f0 * 2.005 * t)) * beat_mod
        env = adsr_env(n, 0.10, 0.20, 0.65, 0.35)
        return pan_stereo(w * env * 0.52, pan)

    if kind == "pendulum_tick":
        return pan_stereo(synth_clock_tick(True, vel=0.55), pan)

    if kind == "water_droplet_ripple":
        n = int(0.42 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f = 520.0 + 1100.0 * (1.0 - np.exp(-t * 28.0))
        ph = 2 * math.pi * np.cumsum(f) / SR
        w = np.sin(ph) * np.exp(-t * 14.0) * np.minimum(1.0, t / 0.005)
        return pan_stereo(w * 0.68, pan)

    if kind in ("ripple_bloom_shimmer", "wavelet_spark_chime", "dual_ray_harmonic"):
        n = int(0.85 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        freqs = [880.0, 1318.5, 1760.0] if kind != "dual_ray_harmonic" else [659.25, 987.77, 1480.0]
        w = sum(np.sin(2 * math.pi * f * t) * (0.45 / (i + 1)) for i, f in enumerate(freqs))
        env = np.exp(-t * 4.2) * np.minimum(1.0, t / 0.012)
        return pan_stereo(w * env * 0.55, pan)

    if kind == "wavefront_slit_hit":
        n = int(0.55 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        sub = np.sin(2 * math.pi * (120 * np.exp(-t * 6.0)) * t) * np.exp(-t * 7.5)
        shimmer = np.sin(2 * math.pi * 1174.66 * t) * np.exp(-t * 8.0) * 0.35
        return pan_stereo((sub * 0.65 + shimmer) * 0.85, pan)

    if kind == "crystal_prism_split":
        n = int(1.05 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        # Two diverging crystal bell tones representing ordinary + extraordinary rays!
        ray_o = np.sin(2 * math.pi * 1046.5 * t) * np.exp(-t * 3.8)
        ray_e = np.sin(2 * math.pi * 1568.0 * t) * np.exp(-t * 3.2)
        l = (ray_o * 0.72 + ray_e * 0.25) * 0.55
        r = (ray_o * 0.25 + ray_e * 0.72) * 0.55
        return np.stack([l, r], axis=-1).astype(np.float32)

    if kind == "particle_dogma_shatter":
        n = int(0.90 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        noise = butter_filter(rng.standard_normal(n).astype(np.float32), "high", 1800.0, order=2)
        burst = noise * np.exp(-t * 9.0) * 0.55
        shards = sum(np.sin(2 * math.pi * (1900 + i * 620) * t) * np.exp(-t * (7 + i * 2)) * 0.16 for i in range(6))
        boom = np.sin(2 * math.pi * (110 * np.exp(-t * 9.0)) * t) * np.exp(-t * 10.0) * 0.65
        return pan_stereo((burst + shards + boom) * 0.90, pan)

    if kind == "pendulum_clock_chime":
        n = int(1.40 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        bell = (
            0.55 * np.sin(2 * math.pi * 587.33 * t)
            + 0.32 * np.sin(2 * math.pi * 1174.66 * t)
            + 0.18 * np.sin(2 * math.pi * 1760.0 * t)
        ) * np.exp(-t * 2.4) * np.minimum(1.0, t / 0.008)
        return pan_stereo(bell * 0.65, pan)

    if kind == "saturn_cosmic_bloom":
        n = int(1.50 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        env = adsr_env(n, 0.35, 0.25, 0.65, 0.55)
        w = (
            0.50 * np.sin(2 * math.pi * 146.83 * t)
            + 0.35 * np.sin(2 * math.pi * 293.66 * t)
            + 0.28 * np.sin(2 * math.pi * 440.0 * t)
            + 0.22 * np.sin(2 * math.pi * 880.0 * t)
        )
        return pan_stereo(w * env * 0.58, pan)

    if kind in ("laser_photonic_sweep", "fiveg_wave_pulse"):
        n = int(0.80 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f = 420.0 + 1800.0 * (t / 0.80) if kind == "laser_photonic_sweep" else 980.0 + 420.0 * np.sin(2 * math.pi * 10.0 * t)
        ph = 2 * math.pi * np.cumsum(f) / SR
        env = adsr_env(n, 0.06, 0.14, 0.65, 0.28)
        return pan_stereo(np.sin(ph) * env * 0.48, pan)

    return pan_stereo(np.zeros(int(0.1 * SR), dtype=np.float32), pan)


# ============================================================================
# WORD-BY-WORD KARAOKE BOXLESS ASS SUBTITLE WRITER (YOUTUBE SHORTS STANDARD)
# ============================================================================
def fmt_ass_time(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    h = cs // 360000
    m = (cs % 360000) // 6000
    s = (cs % 6000) // 100
    rem = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{rem:02d}"


HANGING_WORDS = {"a", "an", "the", "of", "on", "in", "to", "with", "and", "or", "was", "tiny"}


def chunk_words(words: list[dict[str, Any]], max_words: int = 6, max_chars: int = 42) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_chars = 0
    for idx, w in enumerate(words):
        wt = w["word"]
        # Split em-dash joined tokens only AFTER the em-dash stays on the preceding word
        add_len = len(wt) + (1 if cur else 0)
        next_is_emdash = (idx + 1 < len(words)) and words[idx + 1]["word"].startswith("—")
        if cur and (len(cur) >= max_words or cur_chars + add_len > max_chars) and not wt.startswith("—"):
            # If chunk would end on a hanging preposition/modifier, move it to start of next chunk!
            if len(cur) >= 3 and cur[-1]["word"].lower().strip(".,!?—") in HANGING_WORDS:
                carry = cur.pop()
                chunks.append(cur)
                cur = [carry, w]
                cur_chars = len(carry["word"]) + 1 + len(wt)
            else:
                chunks.append(cur)
                cur = [w]
                cur_chars = len(wt)
        else:
            cur.append(w)
            cur_chars += add_len
        if (wt.endswith(("—", ",", ".", "!", "?", ";")) or (not next_is_emdash and "—" in wt)) and len(cur) >= 2:
            chunks.append(cur)
            cur = []
            cur_chars = 0
    if cur:
        if len(cur) == 1 and chunks and len(chunks[-1]) <= 6:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


def write_word_karaoke_ass(manifest_lines: list[dict[str, Any]], ass_path: Path) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,60,&H00FFFFFF,&H00F8BD38,&H00120803,&H96000000,-1,0,0,0,100,100,0,0,1,7,2.5,2,85,85,225,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for line in manifest_lines:
        chunks = chunk_words(line["words"], max_words=6, max_chars=34)
        for ch in chunks:
            for idx, w in enumerate(ch):
                t0 = w["start"] if idx > 0 else ch[0]["start"]
                t1 = ch[idx + 1]["start"] if idx + 1 < len(ch) else w["end"]
                if t1 <= t0 + 0.01:
                    t1 = t0 + 0.04
                # Balance into 2 clean lines if chunk has >= 5 words
                split_at = (len(ch) + 1) // 2 if len(ch) >= 5 else len(ch)
                parts = []
                for j, cw in enumerate(ch):
                    token = cw["word"]
                    if j == idx:
                        styled = f"{{\\c&H00F8BD38&}}{token}{{\\c&H00FFFFFF&}}"
                    else:
                        styled = token
                    if j == split_at:
                        parts.append("\\N" + styled)
                    elif j == 0:
                        parts.append(styled)
                    else:
                        # No extra space around em-dash joins
                        sep = "" if (ch[j - 1]["word"].endswith("—") or token.startswith("—")) else " "
                        parts.append(sep + styled)
                text_str = "".join(parts)
                events.append(f"Dialogue: 0,{fmt_ass_time(t0)},{fmt_ass_time(t1)},Default,,0,0,0,,{text_str}")

    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[Subtitles] Exported word-synced boxless ASS: {ass_path} ({len(events)} highlight frames)")


def main() -> None:
    raw_tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    tl = json.loads(json.dumps(raw_tl))
    tl["sfxEvents"] = [{**ev, "time": float(ev["t"])} for ev in tl.get("sfx", [])]

    print("[1/4] Synthesizing Kokoro multi-voice dialogue (Narrator + Huygens + Newton)...")
    dialogue_bus, speaker_mono_buses, manifest_lines = synthesize_dialogue(tl, TTS_OVERRIDES)
    dialogue_bus = add_chamber_reverb(dialogue_bus, mix=0.10)

    print("[2/4] Composing 116 BPM Dutch Golden Age Baroque & Photonic Wave Cinema Score...")
    bgm_bus = compose_bgm()

    print("[3/4] Synthesizing 24 spatial stereo Foley SFX...")
    sfx_bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    for ev in tl["sfxEvents"]:
        add_stereo_event(sfx_bus, synth_sfx(ev["type"], float(ev.get("pan", 0.0))), float(ev["time"]))
    sfx_bus = add_chamber_reverb(sfx_bus, mix=0.12)

    print("[4/4] Mastering 30.000s stereo WAV (-14 LUFS, Peak <= -1.0 dBFS) + Boxless ASS Subtitles...")
    master_and_export(PROJECT_ROOT, tl, dialogue_bus, bgm_bus, sfx_bus, speaker_mono_buses, manifest_lines)
    write_word_karaoke_ass(manifest_lines, PROJECT_ROOT / raw_tl["audio"]["assPath"])
    print("DONE! Audio + Word-Synced Subtitles ready for Kaggle Blender GPU rendering.")


if __name__ == "__main__":
    main()
