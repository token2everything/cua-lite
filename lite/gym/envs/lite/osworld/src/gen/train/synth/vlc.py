"""VLC synth generator (dataclass form).

This module is one of the file-as-topic synth generators: each `File` defined
in the generator table IS both the structural shape AND the content seed (no
separate inner TopicTheme rotation). The scaffold (caps / dataclasses / files /
factory / FILE_TASKS / emit) mirrors the other synth/*.py modules.

Two on-disk source families:

  (a) **vlcrc text-edit rows** (config_setting bucket): pre_config writes the
      OPPOSITE value (so the trivial-pass guard fires); oracle replays the
      target write via `sed`; evaluator postconfig kills VLC and relaunches
      so the running process re-reads the just-edited vlcrc. Anchored on the
      OSWorld evaluator funcs check_qt_bgcone / check_qt_max_volume /
      check_qt_minimal_view / check_qt_slider_colours /
      check_global_key_play_pause / check_play_and_exit /
      check_one_instance_when_started_from_file / is_vlc_recordings_folder.

  (b) **media-transform rows** (media_transform bucket — requires ffmpeg in
      the container, see Dockerfile validation patch): pre_config invokes
      ffmpeg to build a deterministic source mp4 AND a gold sink file at
      `/tmp/gold_<template_id>.<ext>`, then launches VLC with the source.
      Evaluator's `expected.path` points at the gold file. Oracle =
      `cp <gold> <agent_sink>` so the agent's expected output file becomes
      byte-identical to gold → eval PASS. Trivial-pass guard: pre_config
      does NOT create the agent's sink path (only the source + gold), so
      without oracle/agent action the eval finds the sink missing → 0.
      Anchored on compare_images (SSIM-grayscale, snapshot family),
      compare_audios (MFCC+DTW, audio-extract family), compare_videos
      (pHash, rotate / trim family).

  (c) **live-playback rows** (is_vlc_playing / is_vlc_fullscreen buckets):
      pre_config plants a long-enough media file (>=60s) under ~/Desktop and
      launches an empty VLC with HTTP iface so the eval getter can curl the
      status XML. Oracle pkill+relaunch with the file loaded (or xdotool key
      `f` for fullscreen).

  (d) **playlist rows** (check_list bucket): pre_config stages the referenced
      mp3 tracks via ffmpeg sine-wave + a placeholder .m3u; oracle writes
      the gold .m3u. Eval anchors on check_list rule-pattern matching.

Usage:
    uv run python -m lite.gym.envs.lite.osworld.src.gen.train \\
        --track synth --domain vlc
"""

from __future__ import annotations

from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import SynthTemplate

# ---------------------------------------------------------------------------
# Helpers — vlcrc seed / oracle / evaluator builders
# ---------------------------------------------------------------------------


# VLC pre-open helper. Pre-loads target media file (or
# launches empty VLC for config-only tasks) + sleep + activate_window.
# Mirrors the validation cross-domain "launch + sleep + activate_window"
# pattern; matches upstream's `vlc <file>` pre-load narrative ("the video
# is playing in VLC") rather than the agent-must-navigate-Files alternative.
def _vlc_preopen_steps(media_path: str | None = None) -> list[dict]:
    """Launch VLC (optionally with a media file pre-loaded) + sleep + activate.

    Use `media_path=None` for config-only tasks (preferences edits, playlist
    building) where pre-loading would conflict with the task semantics or
    trivial-pass guard. Use `media_path=<file>` for media-transform / snapshot
    tasks where the agent's job is to manipulate an already-loaded file.
    """
    # Validation fix: add HTTP interface so `is_vlc_playing` / `vlc_playing_info`
    # evals can read VLC state via `/requests/status.xml` (default port 8080,
    # password "password" — matches upstream osworld eval expectation). Without
    # `--extraintf http` the evaluator's HTTP probe times out regardless of
    # whether the agent successfully started playback in the GUI, causing
    # f_vlc_25 / f_vlc_26 / f_vlc_27 to systematically fail.
    cmd = [
        "vlc",
        "--extraintf", "http",
        "--http-password", "password",
    ]
    if media_path:
        cmd.append(media_path)
    return [
        {"type": "launch", "parameters": {"command": cmd}},
        {"type": "execute", "parameters": {"command": "sleep 2", "shell": True}},
        {"type": "activate_window", "parameters": {"window_name": "VLC"}},
    ]


def _vlcrc_setup_step(key: str, value: str, *, also_minview_safe: bool = True) -> dict:
    """Pre-config: ensure vlcrc exists, drop any prior `key=` line, append `key=value`.

    `also_minview_safe`: also pre-write `qt-minimal-view=0` so the relaunched
    VLC opens in normal mode (otherwise minimal view blocks Preferences nav).
    Disable when this row IS toggling qt-minimal-view itself.
    """
    safety = ""
    if also_minview_safe and key != "qt-minimal-view":
        safety = (
            "if grep -q '^qt-minimal-view=' /home/user/.config/vlc/vlcrc; then "
            "sed -i 's/^qt-minimal-view=.*/qt-minimal-view=0/' /home/user/.config/vlc/vlcrc; "
            "else echo 'qt-minimal-view=0' >> /home/user/.config/vlc/vlcrc; fi && "
        )
    return {
        "type": "execute",
        "parameters": {
            "command": (
                "mkdir -p /home/user/.config/vlc && "
                "touch /home/user/.config/vlc/vlcrc && "
                + safety
                + f"sed -i '/^#\\?{key}=/d' /home/user/.config/vlc/vlcrc && "
                + f"echo '{key}={value}' >> /home/user/.config/vlc/vlcrc"
            ),
            "shell": True,
        },
    }


def _vlcrc_oracle(key: str, value: str) -> list[dict]:
    """Oracle replay: write target vlcrc value (mirrors agent's UI edit)."""
    return [
        {"type": "execute", "parameters": {
            "command": (
                "mkdir -p /home/user/.config/vlc && "
                "touch /home/user/.config/vlc/vlcrc && "
                f"sed -i '/^#\\?{key}=/d' /home/user/.config/vlc/vlcrc && "
                f"echo '{key}={value}' >> /home/user/.config/vlc/vlcrc"
            ),
            "shell": True,
        }},
    ]


# Postconfig used by every vlcrc-config eval task (matches eval.jsonl 215dfd39):
# kill VLC then relaunch so the running process re-reads the just-edited vlcrc.
_VLCRC_POSTCONFIG: list[dict] = [
    {"type": "launch", "parameters": {"command": ["pkill", "vlc"]}},
    {"type": "launch", "parameters": {
        "command": "vlc --no-audio --no-video-title-show",
        "shell": True,
    }},
]


def _vlcrc_evaluator(func: str, rule: dict) -> dict:
    return {
        "func": func,
        "expected": {"type": "rule", "rules": rule},
        "result": {"type": "vlc_config", "dest": "vlcrc"},
        "postconfig": _VLCRC_POSTCONFIG,
    }


# ---------------------------------------------------------------------------
# Module-level helpers used by the §I file-task templates below.
# ---------------------------------------------------------------------------


def _ffmpeg_make_mp4_cmd(src_path: str, *, color: str, duration: int,
                         with_audio: bool = False) -> str:
    """Return a shell snippet that ffmpeg-builds a deterministic mp4 at src_path.

    color: ffmpeg lavfi color name (e.g. "blue", "red"). duration in seconds.
    with_audio: if True, mux a 440Hz sine-wave audio track (libfaac/aac).
    """
    if with_audio:
        return (
            f"rm -f '{src_path}' && "
            f"ffmpeg -y -hide_banner -loglevel error "
            f"-f lavfi -i 'color=c={color}:s=320x240:d={duration}:r=25' "
            f"-f lavfi -i 'sine=frequency=440:duration={duration}' "
            f"-c:v libx264 -pix_fmt yuv420p -preset ultrafast "
            f"-c:a aac -shortest '{src_path}'"
        )
    return (
        f"rm -f '{src_path}' && "
        f"ffmpeg -y -hide_banner -loglevel error "
        f"-f lavfi -i 'testsrc=size=320x240:rate=25:duration={duration}' "
        f"-c:v libx264 -pix_fmt yuv420p -preset ultrafast '{src_path}'"
    )


# Module-level TEMPLATES bin. §I.f extends this at the bottom of the file.
TEMPLATES: list[SynthTemplate] = []




# ===========================================================================
# §I. File-task templates (dataclass form)
#
# Mirrors synth/libreoffice_calc.py + synth/libreoffice_impress.py §I.
# This domain is file-as-topic (no inner TopicTheme rotation): each File
# already encodes both the structural shape AND the content semantics.
# (Compare: synth/libreoffice_impress.py §I.b adds a TopicTheme pool because
# its decks are thin structural shapes that need topic-driven content +
# real-photo augmentation per seed.)
#
# A vlc File is one of:
#   - vlcrc-shape:  preference state seed (key/value pair under ~/.config/vlc/vlcrc)
#   - playlist:     a .m3u that references on-disk media tracks
#   - media-asset:  a synthesized audio / video clip (ffmpeg lavfi)
#   - live-target:  a media file the agent will play / fullscreen / interact with
#
# The `eval_kind` axis on `Param` (vlcrc_kv | playlist_check | media_compare |
# snapshot_check | live_state) tells the factory how to wire the evaluator +
# oracle. All the heavy-lifting helpers above (`_vlcrc_setup_step`,
# `_make_snapshot_template`, ...) are reused by the factory branches.
#
# Symmetric layout (all synth/*.py):
#   §I.a  Caps                — SYNTH_CAP_TASKS_PER_FILE / _PARAMS_PER_TASK
#   §I.b  Dataclasses         — File / Param / FileTask (frozen)
#   §I.c  File instances      — define each File ONCE
#   §I.d  Factory + emit      — _to_synth_template / _emit_templates
#   §I.e  FILE_TASKS          — flat list, one entry per (file, task) pair
#   §I.f  Emission            — TEMPLATES.extend(_emit_templates(FILE_TASKS))
# ===========================================================================

from dataclasses import dataclass as _I_dataclass, field as _I_field
from typing import Callable as _I_Callable


# §I.a — caps
SYNTH_CAP_TASKS_PER_FILE: int = 2
SYNTH_CAP_PARAMS_PER_TASK: int = 2


# §I.b — Dataclasses.

@_I_dataclass(frozen=True)
class File:
    """One structurally distinct source artifact / VLC state shape.

    `src(template_id, seed) -> list[dict]` returns the LIST of pre_config
    steps that materialise the source state inside the container. For
    vlcrc-shape files this is one `_vlcrc_setup_step`; for media files it
    is an ffmpeg `execute` step (+ optional VLC `launch` step).
    """
    id: str
    setup_class: str
    basename: str
    src: _I_Callable[[str, int], list[dict]]


@_I_dataclass(frozen=True)
class Param:
    """One concrete parameterization of a task.

    Fields:
      gold_args  — domain-internal args used by the factory's eval / oracle
                   branches (e.g. target vlcrc value, agent's expected output
                   path, gold media path, playlist content).
      eval_kind  — "vlcrc_kv" | "playlist_check" | "media_compare"
                   | "snapshot_check" | "live_state"
      eval_args  — kind-specific evaluator construction kwargs (e.g. func,
                   rule_key, rules dict, expected vm_file path).
      instr      — rendered instruction string.
    """
    gold_args: dict
    eval_kind: str
    eval_args: dict
    instr: str


@_I_dataclass(frozen=True)
class FileTask:
    """One (file, task) pair → one SynthTemplate at emit time.

    `gold` is unused for vlc (kept for cross-domain symmetry) — the gold
    construction is fully described by Param.gold_args + Param.eval_kind
    and resolved inside `_to_synth_template`.
    """
    file: File
    task_id: str
    eval_class: str
    gold: _I_Callable[..., list[dict]] | None = None
    params: list[Param] = _I_field(default_factory=list)


# ---------------------------------------------------------------------------
# §I.c — File instances. Each is defined ONCE; FileTask entries reference
# them. Five conceptual groups (~4 files each) per the loop plan.
# ---------------------------------------------------------------------------

# Loop 1 — vlcrc preference shapes. `src` plants the OPPOSITE / placeholder
# value so the trivial-pass eval guard fires before any oracle action.
# Each File = one vlcrc key the agent will edit.

def _vlcrc_src(key: str, init_value: str) -> _I_Callable[[str, int], list[dict]]:
    def _build(_template_id: str, _seed: int) -> list[dict]:
        # Pre-open empty VLC so the agent has the Preferences
        # surface immediately. Postconfig (`_VLCRC_POSTCONFIG`) pkills + relaunches
        # after the agent's edit so the running VLC re-reads vlcrc.
        return [
            _vlcrc_setup_step(key, init_value,
                              also_minview_safe=(key != "qt-minimal-view")),
            *_vlc_preopen_steps(None),
        ]
    return _build


F_VLC_1 = File(
    id="F-VLC-1", setup_class="vlcrc_shape",
    basename="vlcrc__qt_bgcone", src=_vlcrc_src("qt-bgcone", "1"),
)
F_VLC_2 = File(
    id="F-VLC-2", setup_class="vlcrc_shape",
    basename="vlcrc__qt_max_volume",
    src=_vlcrc_src("qt-max-volume", "125"),
)
F_VLC_3 = File(
    id="F-VLC-3", setup_class="vlcrc_shape",
    basename="vlcrc__input_record_path",
    src=_vlcrc_src("input-record-path", "/home/user/Music"),
)
F_VLC_4 = File(
    id="F-VLC-4", setup_class="vlcrc_shape",
    basename="vlcrc__qt_minimal_view",
    src=_vlcrc_src("qt-minimal-view", "0"),
)
# Extra vlcrc shapes to broaden config_setting bucket (K=9 ≫ 4 default Files).
# Each File is a distinct vlcrc key + initial-value combo so the agent learns
# different Preferences nav paths (Interface skin colours / Hotkeys clear /
# Behaviour bool / Behaviour bool).
F_VLC_4B = File(
    id="F-VLC-4B", setup_class="vlcrc_shape",
    basename="vlcrc__qt_slider_colours",
    src=_vlcrc_src("qt-slider-colours",
                   "200;200;200;255;255;255;100;100;100;0;0;0"),
)
F_VLC_4C = File(
    id="F-VLC-4C", setup_class="vlcrc_shape",
    basename="vlcrc__global_key_play_pause",
    src=_vlcrc_src("global-key-play-pause", "Ctrl+Space"),
)
F_VLC_4D = File(
    id="F-VLC-4D", setup_class="vlcrc_shape",
    basename="vlcrc__play_and_exit",
    src=_vlcrc_src("play-and-exit", "1"),
)
F_VLC_4E = File(
    id="F-VLC-4E", setup_class="vlcrc_shape",
    basename="vlcrc__one_instance_when_started_from_file",
    src=_vlcrc_src("one-instance-when-started-from-file", "1"),
)
# validation prefs-expand: extra File instances for the existing check_qt_*
# evaluator funcs so the prefs_vlcrc bucket can grow beyond the
# SYNTH_CAP_TASKS_PER_FILE=2 ceiling on F_VLC_1..4E. Each F_VLC_*X is the
# SAME vlcrc key as the corresponding F_VLC_*, only the initial (opposite)
# value differs — that distinct seed forces a structurally non-trivial
# Param target on the second File.
F_VLC_2X = File(
    id="F-VLC-2X", setup_class="vlcrc_shape",
    basename="vlcrc__qt_max_volume_alt",
    # Seed at 200 so non-trivial targets are 150 / 175 / 250 / 300.
    src=_vlcrc_src("qt-max-volume", "200"),
)
F_VLC_3X = File(
    id="F-VLC-3X", setup_class="vlcrc_shape",
    basename="vlcrc__input_record_path_alt",
    # Seed at /home/user/Downloads so non-trivial targets are any other
    # folder (Videos / tmp/captures / ...).
    src=_vlcrc_src("input-record-path", "/home/user/Downloads"),
)
F_VLC_4F = File(
    id="F-VLC-4F", setup_class="vlcrc_shape",
    basename="vlcrc__qt_slider_colours_alt",
    # Seed at a non-blackish / non-bright triplet so both `type=match`
    # and `type=blackish` non-trivial targets are reachable.
    src=_vlcrc_src("qt-slider-colours",
                   "180;180;180;255;255;255;128;128;128;0;0;0"),
)
F_VLC_4G = File(
    id="F-VLC-4G", setup_class="vlcrc_shape",
    basename="vlcrc__qt_bgcone_alt",
    # bgcone is bool; src=0 so target=1 is non-trivial. Mirror seed of
    # F_VLC_1 (which is src=1, target=0) — twin file for the opposite
    # initial state. Lets us add a SECOND prefs_vlcrc row exercising the
    # `enable the cone artwork` direction (counterpart to F_VLC_1's
    # `disable the cone artwork`).
    src=_vlcrc_src("qt-bgcone", "0"),
)
# validation (eval mirror osworld_vlc_386dbd0e): bind-global-play-pause
# direction. F_VLC_4C is seeded "Ctrl+Space" so target=1 (non-empty) is
# TRIVIAL_PASS. F_VLC_4H is seeded EMPTY so the agent's job is to BIND
# the hotkey (target=1 / non-empty value) — strictly distinct skill from
# F_VLC_4C's "clear" direction. Eval = check_global_key_play_pause with
# expected_global_key_play_pause=1 (any non-empty value passes).
F_VLC_4H = File(
    id="F-VLC-4H", setup_class="vlcrc_shape",
    basename="vlcrc__global_key_play_pause_unbound",
    src=_vlcrc_src("global-key-play-pause", ""),
)
# validation (eval mirror osworld_vlc_d06f0d4d slider-blackish-from-default):
# F_VLC_4B seed is a NON-blackish, NON-default value (RGB 153/255/20/0
# triplets), so `blackish` checks are non-trivial. F_VLC_4I is a sibling
# seeded with the VLC default RGBA-ish palette so we can run TWO blackish
# Params with different RGB hand-tuned values (a second prefs_vlcrc row
# exercising the `blackish` rule branch without paraphrase cloning).
F_VLC_4I = File(
    id="F-VLC-4I", setup_class="vlcrc_shape",
    basename="vlcrc__qt_slider_colours_default_like",
    src=_vlcrc_src("qt-slider-colours",
                   "240;240;240;220;220;220;200;200;200;180;180;180"),
)


# Loop 2 — .m3u playlist files. `src` stages the referenced mp3 tracks (sine
# waves at distinct frequencies) plus a placeholder playlist with a wrong
# entry so trivial-pass eval fails.

def _playlist_src(playlist_basename: str, track_basenames: list[str]) -> (
        _I_Callable[[str, int], list[dict]]):
    desktop = "/home/user/Desktop"
    playlist_path = f"{desktop}/{playlist_basename}"
    track_paths = [f"{desktop}/{n}" for n in track_basenames]
    mp3_cmds = [
        (f"rm -f '{p}' && "
         f"ffmpeg -y -hide_banner -loglevel error "
         f"-f lavfi -i 'sine=frequency={440 + 110 * i}:duration=3' '{p}'")
        for i, p in enumerate(track_paths)
    ]
    placeholder_lines = "#EXTM3U\\nplaceholder_track.mp3\\n"
    setup_cmd = (
        f"mkdir -p {desktop} && "
        + " && ".join(mp3_cmds)
        + f" && printf '{placeholder_lines}' > '{playlist_path}'"
    )

    def _build(_template_id: str, _seed: int) -> list[dict]:
        return [
            {"type": "execute",
             "parameters": {"command": setup_cmd, "shell": True}},
            *_vlc_preopen_steps(None),
        ]
    return _build


F_VLC_5 = File(
    id="F-VLC-5", setup_class="m3u_playlist",
    basename="morning_drive.m3u",
    src=_playlist_src("morning_drive.m3u",
                      ["track_1.mp3", "track_2.mp3", "track_3.mp3"]),
)
F_VLC_6 = File(
    id="F-VLC-6", setup_class="m3u_playlist",
    basename="study_session.m3u",
    src=_playlist_src("study_session.m3u",
                      ["focus_a.mp3", "focus_b.mp3"]),
)
F_VLC_7 = File(
    id="F-VLC-7", setup_class="m3u_playlist",
    basename="podcast_queue.m3u",
    src=_playlist_src("podcast_queue.m3u",
                      ["ep_intro.mp3", "ep_main.mp3", "ep_outro.mp3",
                       "ep_credits.mp3"]),
)
# Loop 3 — synthesized AUDIO source + audio-extract / media-compare pipelines.
# `src` builds an mp4 with embedded sine-wave audio + matching gold mp3 sink.
# Used by both `media_compare` (audio-extract) and snapshot-style ops.

def _av_mp4_src(*, color: str, with_sine_audio: bool, duration: int,
                size: str = "320x240", fps: int = 25,
                sine_freq: int = 440) -> (
        _I_Callable[[str, int], list[dict]]):
    """Build an mp4 source with the given solid color background overlaid
    with a `drawtext` filter that prints the color name + frame number per
    frame. The drawtext makes each frame visually distinct (pHash now varies
    across frames + rotation; SSIM no longer time-invariant), preventing
    vacuous snapshot matches. `sine_freq` makes audio per-task unique so
    cross-task audio extracts remain distinguishable."""
    # drawtext needs a font path; pick a Linux-default DejaVuSans that exists
    # both inside the docker image and on host (used during pre_config exec).
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    drawtext = (
        f"drawtext=fontfile={font}:"
        f"text='{color} frame %{{n}}':"
        f"x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5"
    )

    def _build(template_id: str, _seed: int) -> list[dict]:
        src_path = f"/tmp/src_{template_id}.mp4"
        if with_sine_audio:
            cmd = (
                f"rm -f '{src_path}' && "
                f"ffmpeg -y -hide_banner -loglevel error "
                f"-f lavfi -i 'color=c={color}:s={size}:d={duration}:r={fps}' "
                f"-f lavfi -i 'sine=frequency={sine_freq}:duration={duration}' "
                f"-vf \"{drawtext}\" "
                f"-c:v libx264 -pix_fmt yuv420p -preset ultrafast "
                f"-c:a aac -shortest '{src_path}'"
            )
        else:
            cmd = (
                f"rm -f '{src_path}' && "
                f"ffmpeg -y -hide_banner -loglevel error "
                f"-f lavfi -i 'color=c={color}:s={size}:d={duration}:r={fps}' "
                f"-vf \"{drawtext}\" "
                f"-c:v libx264 -pix_fmt yuv420p -preset ultrafast '{src_path}'"
            )
        return [
            {"type": "execute", "parameters": {"command": cmd, "shell": True}},
            *_vlc_preopen_steps(src_path),
        ]
    return _build


F_VLC_9 = File(
    id="F-VLC-9", setup_class="ffmpeg_av_mp4",
    basename="src_audio_red.mp4",
    # validation: duration ≥60s so clip outlasts agent's trajectory start.
    # 5/6s clips finished before turn_00 → VLC reverted to cone state →
    # Video→Snapshot disabled. Same duration choice as _live_media_src.
    # 440Hz baseline; F_VLC_10 uses 880Hz so MFCC+DTW
    # discriminates the two Files (sha256 also differs).
    src=_av_mp4_src(color="red", with_sine_audio=True, duration=60,
                    sine_freq=440),
)
F_VLC_10 = File(
    id="F-VLC-10", setup_class="ffmpeg_av_mp4",
    basename="src_audio_purple.mp4",
    # Distinct sine frequency vs F_VLC_9 (was both 440 → bit-identical
    # audio extracts; agent could submit either source's extract).
    src=_av_mp4_src(color="purple", with_sine_audio=True, duration=60,
                    sine_freq=880),
)
# Loop 4 — synthesized VIDEO source (no audio) + snapshot / video-filter ops.
# Same `_av_mp4_src` builder, with_sine_audio=False (saves ffmpeg time).

F_VLC_13 = File(
    id="F-VLC-13", setup_class="ffmpeg_v_mp4",
    basename="src_video_blue.mp4",
    src=_av_mp4_src(color="blue", with_sine_audio=False, duration=60),
)
F_VLC_14 = File(
    id="F-VLC-14", setup_class="ffmpeg_v_mp4",
    basename="src_video_green.mp4",
    src=_av_mp4_src(color="green", with_sine_audio=False, duration=60),
)
F_VLC_16 = File(
    id="F-VLC-16", setup_class="ffmpeg_v_mp4",
    basename="src_video_cyan.mp4",
    src=_av_mp4_src(color="cyan", with_sine_audio=False, duration=10),
)
# validation (eval mirror osworld_vlc_aa4b5023): rotate + Save-As at a
# caller-named path. The Save-As filename is part of the task — agent
# must export rotated video to an exact basename under /home/user/.
# Distinct from F_VLC_14.rotate_90 (which uses Desktop/rotated.mp4) at
# two axes: (a) destination directory is /home/user (not Desktop), and
# (b) the filename is user-specified ('1984_*_Commercial.mp4' etc.),
# mirroring the eval row's "save it for me with the name X" phrasing.
F_VLC_28 = File(
    id="F-VLC-28", setup_class="ffmpeg_v_mp4",
    basename="src_video_orange.mp4",
    src=_av_mp4_src(color="orange", with_sine_audio=False, duration=60),
)


# Loop 5 — live-playback target media + subtitle gap-filler.
# `src` plants a long-enough mp4/audio under ~/Desktop and launches an empty
# VLC (so the agent has the app open but the file is NOT loaded → trivial-
# pass guard fires for is_vlc_playing). Clamp duration ≥60s so
# the source outlasts the eval probe (oracle sleeps 8s + eval round-trip).

def _live_media_src(*, basename: str, kind: str, color: str, duration: int,
                    alt_basename: str | None = None,
                    alt_kind: str | None = None,
                    alt_color: str = "",
                    alt_duration: int | None = None
                    ) -> _I_Callable[[str, int], list[dict]]:
    """Plant a primary media file (basename) + optionally a structurally
    distinct ALTERNATE file (alt_basename). Per PD (3b), the second Param of
    each live-state FileTask targets `alt_basename` so gold_args/eval_args
    differ across the cap-2 Param pair (no paraphrase clones)."""
    eff_duration = max(duration, 60)
    src_path = f"/home/user/Desktop/{basename}"

    def _ffmpeg_for(path: str, k: str, c: str, d: int, freq: int) -> str:
        if k == "audio":
            return (
                f"rm -f '{path}' && "
                f"ffmpeg -y -hide_banner -loglevel error "
                f"-f lavfi -i 'sine=frequency={freq}:duration={d}' '{path}'"
            )
        return _ffmpeg_make_mp4_cmd(path, color=c, duration=d, with_audio=False)

    setup_parts = [
        "mkdir -p /home/user/Desktop",
        _ffmpeg_for(src_path, kind, color, eff_duration, 440),
    ]
    if alt_basename is not None:
        alt_path = f"/home/user/Desktop/{alt_basename}"
        a_kind = alt_kind or kind
        a_dur = max(alt_duration if alt_duration is not None else duration, 60)
        setup_parts.append(_ffmpeg_for(alt_path, a_kind, alt_color, a_dur, 660))
    setup_cmd = " && ".join(setup_parts)

    def _build(_template_id: str, _seed: int) -> list[dict]:
        return [
            {"type": "execute",
             "parameters": {"command": setup_cmd, "shell": True}},
            # Pre-open empty VLC (no file arg) so the trivial-pass
            # guard fires — agent still needs to open the target file. Oracle
            # later pkills + relaunches with `--extraintf http` so the eval's
            # vlc_playing_info getter can curl status.xml.
            *_vlc_preopen_steps(None),
        ]
    return _build


F_VLC_17 = File(
    id="F-VLC-17", setup_class="live_media",
    basename="sample.mp4",
    src=_live_media_src(basename="sample.mp4", kind="video",
                        color="blue", duration=8,
                        alt_basename="sample_b.mp4", alt_kind="video",
                        alt_color="yellow", alt_duration=8),
)
F_VLC_18 = File(
    id="F-VLC-18", setup_class="live_media",
    basename="demo_clip.mp4",
    src=_live_media_src(basename="demo_clip.mp4", kind="video",
                        color="red", duration=10,
                        alt_basename="demo_clip_alt.mp4", alt_kind="video",
                        alt_color="magenta", alt_duration=10),
)
F_VLC_19 = File(
    id="F-VLC-19", setup_class="live_media",
    basename="song.mp3",
    src=_live_media_src(basename="song.mp3", kind="audio",
                        color="", duration=5,
                        alt_basename="song_b.mp3", alt_kind="audio",
                        alt_color="", alt_duration=5),
)
F_VLC_20 = File(
    id="F-VLC-20", setup_class="live_media",
    basename="movie.mp4",
    src=_live_media_src(basename="movie.mp4", kind="video",
                        color="green", duration=10,
                        alt_basename="movie_trailer.mp4", alt_kind="video",
                        alt_color="orange", alt_duration=8),
)


# Subtitle-paired video: same launch shape as `_live_media_src` but pre_config
# also drops a sidecar .srt next to the mp4. VLC auto-loads the .srt when the
# agent opens the mp4; eval (`is_vlc_playing` on the basename) doesn't probe
# subtitles directly, but staging the .srt exercises the realistic path-pair
# pattern present in eval rows like 9b03d57c (subtitle-track lecture media).
def _live_media_with_srt_src(*, basename: str, color: str, duration: int,
                             alt_basename: str | None = None,
                             alt_color: str = "",
                             alt_duration: int | None = None
                             ) -> _I_Callable[[str, int], list[dict]]:
    eff_duration = max(duration, 60)
    src_path = f"/home/user/Desktop/{basename}"
    srt_path = f"/home/user/Desktop/{basename.rsplit('.', 1)[0]}.srt"
    srt_body = (
        "1\\n00:00:01,000 --> 00:00:05,000\\nCaption line one\\n\\n"
        "2\\n00:00:05,500 --> 00:00:09,000\\nCaption line two\\n"
    )
    parts = [
        "mkdir -p /home/user/Desktop",
        _ffmpeg_make_mp4_cmd(src_path, color=color,
                             duration=eff_duration, with_audio=False),
        f"printf '{srt_body}' > '{srt_path}'",
    ]
    if alt_basename is not None:
        alt_path = f"/home/user/Desktop/{alt_basename}"
        alt_srt = f"/home/user/Desktop/{alt_basename.rsplit('.', 1)[0]}.srt"
        a_dur = max(alt_duration if alt_duration is not None else duration, 60)
        parts.append(_ffmpeg_make_mp4_cmd(alt_path, color=alt_color,
                                          duration=a_dur, with_audio=False))
        parts.append(f"printf '{srt_body}' > '{alt_srt}'")
    setup_cmd = " && ".join(parts)

    def _build(_template_id: str, _seed: int) -> list[dict]:
        return [
            {"type": "execute",
             "parameters": {"command": setup_cmd, "shell": True}},
            # Empty pre-open; oracle handles http-iface relaunch.
            *_vlc_preopen_steps(None),
        ]
    return _build


F_VLC_21 = File(
    id="F-VLC-21", setup_class="live_media_srt",
    basename="lecture.mp4",
    src=_live_media_with_srt_src(basename="lecture.mp4", color="cyan",
                                 duration=12,
                                 alt_basename="lecture_part2.mp4",
                                 alt_color="white", alt_duration=12),
)


# Loop 6 (validation P2) — network stream target. pre_config builds two long
# sine-wave mp3s under a local web root and starts `python3 -m http.server`
# on port 48080 so VLC can reach `http://localhost:48080/<basename>.mp3` as
# a network URL (F10 mitigation: no external dependency). VLC's status XML
# reports the http URL in its play-state location, which the rule.type=url
# branch of upstream `is_vlc_playing` matches.

def _stream_http_src(stream_basenames: list[str], duration: int = 90,
                     port: int = 48080) -> _I_Callable[[str, int], list[dict]]:
    web_root = "/tmp/vlc_stream_root"
    ffmpeg_cmds = [
        (f"ffmpeg -y -hide_banner -loglevel error "
         f"-f lavfi -i 'sine=frequency={440 + 110 * i}:duration={duration}' "
         f"'{web_root}/{name}'")
        for i, name in enumerate(stream_basenames)
    ]
    # Kill any prior http.server on the port, then spawn fresh in background.
    # `nohup ... & disown` keeps the server alive after the setup step exits.
    server_cmd = (
        f"(fuser -k {port}/tcp 2>/dev/null; true) && sleep 1 && "
        f"cd '{web_root}' && "
        f"nohup python3 -m http.server {port} >/tmp/vlc_stream_server.log "
        f"2>&1 & disown; sleep 2"
    )
    setup_cmd = (
        f"rm -rf '{web_root}' && mkdir -p '{web_root}' && "
        + " && ".join(ffmpeg_cmds)
        + f" && {server_cmd}"
    )

    def _build(_template_id: str, _seed: int) -> list[dict]:
        return [
            {"type": "execute",
             "parameters": {"command": setup_cmd, "shell": True}},
            # Empty pre-open (no stream URL); oracle handles
            # the http-iface relaunch with the target URL as positional argv.
            *_vlc_preopen_steps(None),
        ]
    return _build


F_VLC_22 = File(
    id="F-VLC-22", setup_class="http_stream",
    basename="stream_sample.mp3",
    src=_stream_http_src(["stream_sample.mp3", "stream_alt.mp3"]),
)


# Loop 6b — remote-URL playback. Mirrors upstream eval rows that
# stream from public CDNs (e.g. eval `bba3381f` HLS m3u8). pre_config only
# pre-opens an empty VLC; the oracle relaunches VLC with the public URL as
# argv. Eval = `is_vlc_playing` with rule.type=url against the running VLC's
# status XML.
#
# URL-reachability verified at synth-generation time:
#   - https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4
#     → HTTP 200 (Blender Foundation CDN, stable since 2008).
#   - https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_fmp4/master.m3u8
#     → HTTP 200 (Apple's BipBop HLS example; same URL used by upstream
#       eval row bba3381f).
#   - https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8 → HTTP 200 (Mux
#     test HLS playlist; widely used in HLS player demos).
#
# `_remote_url_src` only spins up VLC empty — the URL is dialed from the
# oracle's relaunch step. No local http server, no ffmpeg dependency.


def _remote_url_src() -> _I_Callable[[str, int], list[dict]]:
    """Pre_config for remote-URL playback: launch empty VLC and let the
    oracle's pkill+relaunch step dial the URL. No on-disk asset is built."""
    def _build(_template_id: str, _seed: int) -> list[dict]:
        return _vlc_preopen_steps(None)
    return _build


# Dropped — F-VLC-24 File def commented along with its FileTask. See
# drop comment near `play_remote_http_video` below.
# F_VLC_24 = File(
#     id="F-VLC-24", setup_class="remote_http_video",
#     basename="BigBuckBunny_320x180.mp4",
#     src=_remote_url_src(),
# )
F_VLC_25 = File(
    id="F-VLC-25", setup_class="remote_hls_stream",
    basename="master.m3u8",
    src=_remote_url_src(),
)
F_VLC_26 = File(
    id="F-VLC-26", setup_class="remote_hls_stream",
    basename="x36xhzz.m3u8",
    src=_remote_url_src(),
)
# F-VLC-27 streams a plain MP3 over http. validation: switched from a public CDN
# (soundhelix.com) to the LOCAL `_stream_http_src` http.server pattern. The CDN
# URL was network-dependent and raced VLC's 15s status probe (audit: conditional
# FAIL); a plain MP3 (unlike the F-VLC-25/26 HLS m3u8 playlists) serves perfectly
# from the local sine-wave web root, so playback is deterministic + offline.
F_VLC_27 = File(
    id="F-VLC-27", setup_class="http_stream",
    basename="mpthreetest.mp3",
    src=_stream_http_src(["mpthreetest.mp3"]),
)


# Loop 7 (validation P3) — cross-app frame → desktop wallpaper. pre_config
# builds a source mp4 (cyan testsrc, ≥60s so the frame at later seconds is
# reachable) under ~/Desktop; agent will snapshot a frame in VLC and apply
# it as the GNOME desktop wallpaper. The gold png is extracted at the same
# timestamp via ffmpeg in _build_wallpaper_extras (mirrors snapshot family).

def _wallpaper_src_mp4() -> _I_Callable[[str, int], list[dict]]:
    """Plant a long-enough source mp4 under ~/Desktop and launch VLC with it
    loaded so the agent can pause / scrub / snapshot any frame."""
    basename = "scenic_clip.mp4"
    src_path = f"/home/user/Desktop/{basename}"
    # ≥60s clamp matches _live_media_src to keep state=playing through eval.
    duration = 60

    def _build(_template_id: str, _seed: int) -> list[dict]:
        return [
            {"type": "execute", "parameters": {
                "command": (
                    f"mkdir -p /home/user/Desktop && "
                    + _ffmpeg_make_mp4_cmd(src_path, color="teal",
                                           duration=duration,
                                           with_audio=False)
                ),
                "shell": True,
            }},
            *_vlc_preopen_steps(src_path),
        ]
    return _build


F_VLC_23 = File(
    id="F-VLC-23", setup_class="frame_wallpaper",
    basename="scenic_clip.mp4",
    src=_wallpaper_src_mp4(),
)


# ---------------------------------------------------------------------------
# §I.d — Factory + emit.
#
# `_to_synth_template` dispatches on Param.eval_kind. The five branches share
# a common shell: pre_config = ft.file.src(template_id, seed) + any per-task
# extras built from Param.gold_args; oracle + evaluator wired per kind.
# ---------------------------------------------------------------------------

def _eval_vlcrc_kv(eval_args: dict) -> dict:
    """Build a vlcrc-key evaluator dict. `eval_args` keys: func, rule."""
    return _vlcrc_evaluator(eval_args["func"], eval_args["rule"])


def _eval_media_compare(eval_args: dict, agent_path: str) -> dict:
    """Build a compare_images / compare_audios / compare_videos evaluator.
    `eval_args` keys: func, gold_path, agent_basename, gold_basename."""
    return {
        "func": eval_args["func"],
        "expected": {"type": "vm_file", "path": eval_args["gold_path"],
                     "dest": eval_args["gold_basename"]},
        "result": {"type": "vm_file", "path": agent_path,
                   "dest": eval_args["agent_basename"]},
        "options": {},
    }


def _eval_playlist_check(eval_args: dict, agent_path: str) -> dict:
    """Build a check_list evaluator over the agent's saved .m3u file."""
    return {
        "func": "check_list",
        "expected": {"type": "rule",
                     "rules": {"expect": eval_args["expect_patterns"]}},
        "result": {"type": "vm_file", "path": agent_path,
                   "dest": eval_args["agent_basename"]},
    }


def _eval_live_state(eval_args: dict) -> dict:
    """Build is_vlc_playing / is_vlc_fullscreen evaluator dict.

    Two rule shapes per upstream `is_vlc_playing`:
      - file_name  — agent opens local file; rule matches basename
      - url        — agent opens network stream; rule matches URL text
    validation P2 introduces the url branch for the http_stream FileTask.
    """
    func = eval_args["func"]
    if func == "is_vlc_playing":
        if "url" in eval_args:
            rules = {"type": "url", "url": eval_args["url"]}
        else:
            rules = {"type": "file_name",
                     "file_name": eval_args["file_name"]}
        return {
            "func": "is_vlc_playing",
            "expected": {"type": "rule", "rules": rules},
            "result": {"type": "vlc_playing_info", "dest": "status.xml"},
        }
    # is_vlc_fullscreen
    return {
        "func": "is_vlc_fullscreen",
        "expected": {"type": "vm_window_size", "app_class_name": "vlc"},
        "result": {"type": "vm_screen_size"},
    }


def _eval_wallpaper_check(eval_args: dict) -> dict:
    """validation P3 — compare_images between vm_wallpaper (current desktop
    background) and a gold png pre-built by `_build_wallpaper_extras`."""
    return {
        "func": "compare_images",
        "expected": {"type": "vm_file",
                     "path": eval_args["gold_path"],
                     "dest": eval_args["gold_basename"]},
        "result": {"type": "vm_wallpaper", "dest": "result_wallpaper.png"},
        "options": {},
    }


def _oracle_vlcrc_kv(gold_args: dict) -> list[dict]:
    return _vlcrc_oracle(gold_args["vlcrc_key"], gold_args["target_value"])


def _oracle_cp_gold(gold_args: dict) -> list[dict]:
    """Generic oracle for media-compare / snapshot families: `cp gold agent`.

    `gold_args` keys: gold_path, agent_path. The agent_path must NOT exist
    pre-oracle (trivial-pass guard); pre_config builds only the source +
    gold, never the agent's expected output.
    """
    return [{"type": "execute", "parameters": {
        "command": (
            f"mkdir -p $(dirname '{gold_args['agent_path']}') && "
            f"cp '{gold_args['gold_path']}' '{gold_args['agent_path']}'"
        ),
        "shell": True,
    }}]


def _oracle_playlist_write(gold_args: dict) -> list[dict]:
    """Write the gold .m3u over whatever the placeholder pre_config wrote."""
    return [{"type": "execute", "parameters": {
        "command": f"printf '{gold_args['gold_lines_shell']}' "
                   f"> '{gold_args['agent_path']}'",
        "shell": True,
    }}]


def _oracle_play_local(gold_args: dict) -> list[dict]:
    """Mirrors `_make_play_local_template` oracle: pkill then relaunch with
    the file loaded + HTTP iface (matching the runner getter's password list).
    `gold_args` keys: src_path, media_kind."""
    audio_flag = "" if gold_args["media_kind"] == "audio" else "--no-audio "
    return [
        {"type": "execute", "parameters": {
            "command": "pkill -9 -f vlc 2>/dev/null; sleep 2; true",
            "shell": True,
        }},
        {"type": "launch", "parameters": {
            "command": (
                f"VLC_VERBOSE=-1 vlc --extraintf http --http-password password "
                f"--no-video-title-show {audio_flag}'{gold_args['src_path']}'"
            ),
            "shell": True,
        }},
        {"type": "sleep", "parameters": {"seconds": 8}},
    ]


def _oracle_play_stream(gold_args: dict) -> list[dict]:
    """validation P2 — oracle for the network-stream FileTask. pkill+relaunch
    VLC with the http URL as argv (same `--extraintf http` so the eval
    getter can curl status.xml). Audio stream → no `--no-audio` flag, so
    VLC stays in state=playing past the eval probe.
    `gold_args` keys: url, media_kind (always 'audio' for this family).
    validation: sleep raised 8→15s — large MP4 streams (blender 64 MB) need
    longer first-byte+demux time before VLC reports state=playing on
    status.xml; HLS playlists are tiny and were fine at 8s."""
    audio_flag = "" if gold_args.get("media_kind") == "audio" else "--no-audio "
    return [
        {"type": "execute", "parameters": {
            "command": "pkill -9 -f vlc 2>/dev/null; sleep 2; true",
            "shell": True,
        }},
        {"type": "launch", "parameters": {
            "command": (
                f"VLC_VERBOSE=-1 vlc --extraintf http --http-password password "
                f"--no-video-title-show {audio_flag}'{gold_args['url']}'"
            ),
            "shell": True,
        }},
        {"type": "sleep", "parameters": {"seconds": 15}},
    ]


def _oracle_set_wallpaper(gold_args: dict) -> list[dict]:
    """validation P3 — oracle for the frame-to-wallpaper FileTask. Also copy
    the gold png to the agent's expected Desktop sink (so a future audit
    checking both can pass), then set the GNOME desktop background to point
    at the gold png. The wallpaper getter (`docker/server/main.py:1055`)
    reads `gsettings get org.gnome.desktop.background picture-uri` and
    serves the referenced file, which the eval then compares to gold."""
    gold_path = gold_args["gold_path"]
    agent_path = gold_args["agent_path"]
    return [
        {"type": "execute", "parameters": {
            "command": (
                f"mkdir -p $(dirname '{agent_path}') && "
                f"cp '{gold_path}' '{agent_path}' && "
                # GNOME desktop background; gsettings expects a file:// URI.
                f"gsettings set org.gnome.desktop.background picture-uri "
                f"'file://{agent_path}' 2>/dev/null; "
                f"gsettings set org.gnome.desktop.background picture-uri-dark "
                f"'file://{agent_path}' 2>/dev/null; true"
            ),
            "shell": True,
        }},
        {"type": "sleep", "parameters": {"seconds": 2}},
    ]


def _oracle_fullscreen(gold_args: dict) -> list[dict]:
    """Mirrors `_make_fullscreen_template` oracle: pkill, relaunch windowed,
    then `xdotool key f` to toggle fullscreen. `gold_args` keys: src_path."""
    src_path = gold_args["src_path"]
    return [
        {"type": "execute", "parameters": {
            "command": (
                "pkill -9 vlc 2>/dev/null; sleep 2 && "
                "mkdir -p /home/user/.config/vlc && "
                "touch /home/user/.config/vlc/vlcrc && "
                "sed -i '/^#\\?qt-privacy-ask=/d' /home/user/.config/vlc/vlcrc && "
                "echo 'qt-privacy-ask=0' >> /home/user/.config/vlc/vlcrc"
            ),
            "shell": True,
        }},
        {"type": "launch", "parameters": {
            "command": (
                f"DISPLAY=:1 vlc --no-audio --no-video-title-show '{src_path}'"
            ),
            "shell": True,
        }},
        {"type": "sleep", "parameters": {"seconds": 5}},
        {"type": "execute", "parameters": {
            "command": (
                "WID=$(xdotool search --class vlc 2>/dev/null | head -1); "
                "if [ -n \"$WID\" ]; then "
                "  xdotool windowactivate $WID 2>/dev/null; sleep 1; "
                "  xdotool key f; "
                "fi"
            ),
            "shell": True,
        }},
        {"type": "sleep", "parameters": {"seconds": 5}},
    ]


def _build_snapshot_extras(template_id: str, gold_args: dict) -> list[dict]:
    """Pre_config addendum for snapshot family: extract gold png from src mp4."""
    src_path = f"/tmp/src_{template_id}.mp4"
    gold_path = gold_args["gold_path"]
    return [{"type": "execute", "parameters": {
        "command": (
            f"rm -f '{gold_path}' && "
            f"ffmpeg -y -hide_banner -loglevel error "
            f"-ss {gold_args['snapshot_seconds']} -i '{src_path}' "
            f"-frames:v 1 '{gold_path}'"
        ),
        "shell": True,
    }}]


def _build_media_compare_extras(template_id: str, gold_args: dict) -> list[dict]:
    """Pre_config addendum for media_compare family: build the gold sink
    (mp3 / mp4) via ffmpeg from the source mp4."""
    src_path = f"/tmp/src_{template_id}.mp4"
    gold_path = gold_args["gold_path"]
    cmd = gold_args["gold_ffmpeg_cmd"].format(src=src_path, gold=gold_path)
    return [{"type": "execute",
             "parameters": {"command": cmd, "shell": True}}]


def _build_wallpaper_extras(_template_id: str, gold_args: dict) -> list[dict]:
    """validation P3 — pre_config addendum for the wallpaper FileTask. Extract
    the gold png frame from the Desktop-resident source mp4 (`scenic_clip.
    mp4`, planted by `_wallpaper_src_mp4`) at the target snapshot second."""
    src_path = "/home/user/Desktop/scenic_clip.mp4"
    gold_path = gold_args["gold_path"]
    return [{"type": "execute", "parameters": {
        "command": (
            f"rm -f '{gold_path}' && "
            f"ffmpeg -y -hide_banner -loglevel error "
            f"-ss {gold_args['snapshot_seconds']} -i '{src_path}' "
            f"-frames:v 1 '{gold_path}'"
        ),
        "shell": True,
    }}]


def _to_synth_template(ft: FileTask) -> SynthTemplate:
    """Turn ONE FileTask into ONE SynthTemplate.

    Per-seed: pick the i-th Param (i = seed % len(params)). Source steps
    come from File.src; per-eval-kind extras + evaluator + oracle come
    from Param.eval_kind dispatch.
    """
    pool = ft.params[:SYNTH_CAP_PARAMS_PER_TASK]
    template_id = f"{ft.file.id.lower().replace('-', '_')}__{ft.task_id}"

    def _params(seed: int) -> dict:
        variant = pool[seed % len(pool)]
        pre = list(ft.file.src(template_id, seed))
        if variant.eval_kind == "snapshot_check":
            pre = pre + _build_snapshot_extras(template_id, variant.gold_args)
        elif variant.eval_kind == "media_compare":
            pre = pre + _build_media_compare_extras(template_id, variant.gold_args)
        elif variant.eval_kind == "wallpaper_check":
            pre = pre + _build_wallpaper_extras(template_id, variant.gold_args)
        return {
            "instr": variant.instr,
            "eval_kind": variant.eval_kind,
            "eval_args": variant.eval_args,
            "gold_args": variant.gold_args,
            "pre_config_steps": pre,
        }

    def _instr(p: dict) -> str:
        return p["instr"]

    def _eval(p: dict) -> dict:
        kind = p["eval_kind"]
        if kind == "vlcrc_kv":
            return _eval_vlcrc_kv(p["eval_args"])
        if kind == "playlist_check":
            return _eval_playlist_check(p["eval_args"], p["gold_args"]["agent_path"])
        if kind in ("media_compare", "snapshot_check"):
            return _eval_media_compare(p["eval_args"], p["gold_args"]["agent_path"])
        if kind == "live_state":
            return _eval_live_state(p["eval_args"])
        if kind == "wallpaper_check":
            return _eval_wallpaper_check(p["eval_args"])
        raise ValueError(f"unknown eval_kind: {kind}")

    def _oracle(p: dict) -> list[dict]:
        kind = p["eval_kind"]
        gold = p["gold_args"]
        if kind == "vlcrc_kv":
            return _oracle_vlcrc_kv(gold)
        if kind == "playlist_check":
            return _oracle_playlist_write(gold)
        if kind in ("media_compare", "snapshot_check"):
            return _oracle_cp_gold(gold)
        if kind == "live_state":
            if p["eval_args"]["func"] == "is_vlc_fullscreen":
                return _oracle_fullscreen(gold)
            # Network-stream variant uses `url` in gold_args instead of src_path
            if "url" in gold:
                return _oracle_play_stream(gold)
            return _oracle_play_local(gold)
        if kind == "wallpaper_check":
            return _oracle_set_wallpaper(gold)
        raise ValueError(f"unknown eval_kind: {kind}")

    return SynthTemplate(
        template_id=template_id,
        domain="vlc",
        instruction_fn=_instr,
        evaluator_fn=_eval,
        oracle_fn=_oracle,
        postconfig_fn=lambda _p: None,
        param_fn=_params,
        n_rows=len(pool),
        setup_class=ft.file.setup_class,
        eval_class=ft.eval_class,
    )


def _emit_templates(file_tasks: list[FileTask]) -> list[SynthTemplate]:
    """Enforce SYNTH_CAP_TASKS_PER_FILE at emit time. Tasks beyond cap are
    headroom (kept in FILE_TASKS for ablation but not emitted)."""
    per_file: dict[str, int] = {}
    out: list[SynthTemplate] = []
    for ft in file_tasks:
        c = per_file.get(ft.file.id, 0)
        if c >= SYNTH_CAP_TASKS_PER_FILE:
            continue
        per_file[ft.file.id] = c + 1
        out.append(_to_synth_template(ft))
    return out


# ---------------------------------------------------------------------------
# §I.e — FILE_TASKS: flat list. Each entry is one (file × task) pair.
# Quality-ranked: the first ≤2 tasks/file × ≤2 params/task = the emitted set.
# ---------------------------------------------------------------------------

# Helpers for building common Param shapes ----------------------------------

def _vlcrc_param(*, vlcrc_key: str, target_value: str | int,
                 func: str, rule_key: str, rule_value: int | str | None,
                 instr: str) -> Param:
    """Build a vlcrc_kv Param. `rule_value` defaults to int(target_value)
    for bool/int keys; pass an explicit value for string-typed rules."""
    if rule_value is None:
        rule_value = target_value
    return Param(
        gold_args={"vlcrc_key": vlcrc_key,
                   "target_value": str(target_value)},
        eval_kind="vlcrc_kv",
        eval_args={"func": func,
                   "rule": {rule_key: rule_value}},
        instr=instr,
    )


def _snapshot_param(*, snapshot_seconds: int, agent_basename: str,
                    instr: str, file_id: str, task_id: str) -> Param:
    template_id = f"{file_id.lower().replace('-', '_')}__{task_id}"
    gold_path = f"/tmp/gold_{template_id}_t{snapshot_seconds}.png"
    agent_path = f"/home/user/Desktop/{agent_basename}"
    return Param(
        gold_args={"snapshot_seconds": snapshot_seconds,
                   "gold_path": gold_path,
                   "agent_path": agent_path},
        eval_kind="snapshot_check",
        eval_args={"func": "compare_images",
                   "gold_path": gold_path,
                   "gold_basename": f"gold_{template_id}_t{snapshot_seconds}.png",
                   "agent_basename": agent_basename},
        instr=instr,
    )


def _media_compare_param(*, ext: str, gold_ffmpeg_cmd: str,
                         agent_basename: str, func: str,
                         instr: str, file_id: str, task_id: str,
                         tag: str) -> Param:
    template_id = f"{file_id.lower().replace('-', '_')}__{task_id}"
    gold_path = f"/tmp/gold_{template_id}_{tag}.{ext}"
    agent_path = f"/home/user/Desktop/{agent_basename}"
    return Param(
        gold_args={"gold_path": gold_path,
                   "gold_ffmpeg_cmd": gold_ffmpeg_cmd,
                   "agent_path": agent_path},
        eval_kind="media_compare",
        eval_args={"func": func,
                   "gold_path": gold_path,
                   "gold_basename": f"gold_{template_id}_{tag}.{ext}",
                   "agent_basename": agent_basename},
        instr=instr,
    )


def _playlist_param(*, agent_basename: str, track_basenames: list[str],
                    instr: str) -> Param:
    agent_path = f"/home/user/Desktop/{agent_basename}"
    expect_patterns = [n.replace(".", "\\.") for n in track_basenames]
    gold_lines = "#EXTM3U\\n" + "\\n".join(track_basenames) + "\\n"
    return Param(
        gold_args={"agent_path": agent_path,
                   "gold_lines_shell": gold_lines},
        eval_kind="playlist_check",
        eval_args={"expect_patterns": expect_patterns,
                   "agent_basename": agent_basename},
        instr=instr,
    )


def _play_local_param(*, basename: str, kind: str, instr: str) -> Param:
    src_path = f"/home/user/Desktop/{basename}"
    return Param(
        gold_args={"src_path": src_path, "media_kind": kind},
        eval_kind="live_state",
        eval_args={"func": "is_vlc_playing", "file_name": basename},
        instr=instr,
    )


def _fullscreen_param(*, basename: str, instr: str) -> Param:
    src_path = f"/home/user/Desktop/{basename}"
    return Param(
        gold_args={"src_path": src_path},
        eval_kind="live_state",
        eval_args={"func": "is_vlc_fullscreen"},
        instr=instr,
    )


def _stream_param(*, url: str, instr: str) -> Param:
    """Build a network-stream play_local-style Param. The gold_args carry
    `url` (so the oracle relaunches VLC with the URL as positional argv)
    while eval_args use `rule.type=url` mode of upstream `is_vlc_playing`."""
    return Param(
        gold_args={"url": url, "media_kind": "audio"},
        eval_kind="live_state",
        eval_args={"func": "is_vlc_playing", "url": url},
        instr=instr,
    )


def _wallpaper_param(*, snapshot_seconds: int, agent_basename: str,
                     instr: str, file_id: str, task_id: str) -> Param:
    """Build a frame-to-wallpaper Param. The gold png is extracted by
    `_build_wallpaper_extras` from the Desktop-resident source mp4; the
    oracle saves the same gold png as the agent's expected sink AND sets
    it as the GNOME desktop wallpaper. Eval compares the live wallpaper
    image (vm_wallpaper) against the gold via compare_images."""
    template_id = f"{file_id.lower().replace('-', '_')}__{task_id}"
    gold_path = f"/tmp/gold_{template_id}_wp{snapshot_seconds}.png"
    agent_path = f"/home/user/Desktop/{agent_basename}"
    return Param(
        gold_args={"snapshot_seconds": snapshot_seconds,
                   "gold_path": gold_path,
                   "agent_path": agent_path},
        eval_kind="wallpaper_check",
        eval_args={"func": "compare_images",
                   "gold_path": gold_path,
                   "gold_basename":
                       f"gold_{template_id}_wp{snapshot_seconds}.png"},
        instr=instr,
    )


# FILE_TASKS list -----------------------------------------------------------

FILE_TASKS: list[FileTask] = [
    # ---- Loop 1: vlcrc preference shapes ----------------------------------
    # F_VLC_1: bool key. src writes 1 (vlcrc:264) — only target=0 is a real
    # flip; target=1 would be TRIVIAL_PASS (src == target, agent need do
    # nothing). Cap to single non-trivial Param.
    FileTask(F_VLC_1, "set_bgcone", "config_setting", params=[
        _vlcrc_param(vlcrc_key="qt-bgcone", target_value=0,
                     func="check_qt_bgcone",
                     rule_key="expected_qt_bgcone", rule_value=0,
                     instr="I find the skeuomorphic cone artwork in VLC's "
                           "background distracting whenever I pause a video. "
                           "Could you turn off the splash-screen cone icon "
                           "in VLC's preferences so the player background "
                           "stays clean when nothing is playing?"),
    ]),
    # KNOWN ISSUE (validation HARD): the QSpinBox-style controls in
    # Tools → Preferences → Show settings: All accept Ctrl+A but DO NOT
    # select the current value (Qt toolkit quirk — cousin of the gimp
    # GTK Ctrl+A APPEND cluster). Agent's `hotkey(Ctrl,A) → type_text(N)`
    # idiom appends, producing values like `200200` or `5200`. Listed HARD
    # rather than BUG because the only safe instruction-side fix (suggest
    # triple-click / select-all alternative) requires task-specific
    # wording per-Param; left as skill ceiling pending a Qt-aware hint.
    # validation: dropped Param[1] (numeric variant 200→150 on same vlcrc key —
    # same skill axis, just a value rotation; per the mission's "Set max volume
    # to 200/150 → BAD" example). The structurally-distinct alternative (250%)
    # already lives in F_VLC_2X.set_max_volume_alt.
    FileTask(F_VLC_2, "set_max_volume", "config_setting", params=[
        _vlcrc_param(vlcrc_key="qt-max-volume", target_value=200,
                     func="check_qt_max_volume",
                     rule_key="expected_qt_max_volume", rule_value=200,
                     instr="I'm watching a quietly-mixed indie film on my "
                           "laptop and 125% is still not loud enough. Please "
                           "raise VLC's maximum displayed volume cap to 200% "
                           "so I can boost the audio above the default ceiling."),
    ]),
    # validation: dropped Param[1] (different path on same vlcrc key — value
    # rotation; F_VLC_3X.set_recordings_folder_alt already covers /Videos and
    # /tmp/captures destinations).
    FileTask(F_VLC_3, "set_recordings_folder", "config_setting", params=[
        # validation reframe (eval mirror osworld_vlc_8ba5ae7a): tight
        # 1-sentence imperative — "Help me modify the folder used to store
        # my recordings to Desktop" — replacing the verbose
        # context-prefix variant.
        _vlcrc_param(vlcrc_key="input-record-path",
                     target_value="/home/user/Desktop",
                     func="is_vlc_recordings_folder",
                     rule_key="recording_file_path",
                     rule_value="/home/user/Desktop",
                     instr="Help me modify the folder used to store my "
                           "recordings to Desktop."),
    ]),
    # F_VLC_4: bool key. src writes 0 (vlcrc:279) — only target=1 is a real
    # flip; target=0 would be TRIVIAL_PASS. Cap to single non-trivial Param.
    FileTask(F_VLC_4, "set_minimal_view", "config_setting", params=[
        _vlcrc_param(vlcrc_key="qt-minimal-view", target_value=1,
                     func="check_qt_minimal_view",
                     rule_key="expected_qt_minimal_view", rule_value=1,
                     instr="I'm multitasking and the persistent VLC toolbar "
                           "is eating my screen space. Please put VLC into "
                           "its minimal interface mode: open Tools → "
                           "Preferences, and on the Interface page enable "
                           "\"Start in minimal view mode\", then click Save."),
    ]),
    # Loop-1 broaden: 12-RGB list (interface skin colours) — distinct
    # vlcrc value-shape from the bool/int/path/scalar ones above.
    # validation: dropped Param[1] (palette rotation on same key/match rule —
    # value-axis variant). F_VLC_4F.set_slider_colours_alt + F_VLC_4B
    # set_slider_colours_blackish carry structurally distinct palettes.
    FileTask(F_VLC_4B, "set_slider_colours", "config_setting", params=[
        # check_qt_slider_colours (upstream vlc.py:418) requires rule.type
        # ("match" or "blackish"); for "match" the expected value is the
        # raw vlcrc string (line.split('=')[-1].strip()), NOT a list.
        Param(
            gold_args={"vlcrc_key": "qt-slider-colours",
                       "target_value":
                           "153;210;243;255;255;255;20;210;20;0;0;0"},
            eval_kind="vlcrc_kv",
            eval_args={
                "func": "check_qt_slider_colours",
                "rule": {"type": "match",
                         "expected_qt_slider_colours":
                             "153;210;243;255;255;255;20;210;20;0;0;0"},
            },
            instr="I'm theming my media station and want a brighter seek "
                  "bar. Please recolour VLC's slider to a four-stop "
                  "(153, 210, 243), (255, 255, 255), (20, 210, 20), "
                  "(0, 0, 0) gradient so the playback bar pops against the "
                  "darker UI."),
    ]),
    # Loop-1 broaden: hotkey string (different keystroke) — exercises
    # Preferences → Hotkeys nav, not the All-settings tree.
    # Validation PARAM_REDUCIBLE: dropped the rebind-to-Space
    # Param — rebinding is a strictly harder skill than clearing. Kept the
    # clear (target="") variant.
    FileTask(F_VLC_4C, "clear_global_play_pause", "config_setting", params=[
        _vlcrc_param(
            vlcrc_key="global-key-play-pause", target_value="",
            func="check_global_key_play_pause",
            rule_key="expected_global_key_play_pause", rule_value=0,
            instr="I keep triggering pause from another window by accident "
                  "while typing. Please clear VLC's global Play/Pause "
                  "hotkey so the system-wide keyboard shortcut is fully "
                  "unset and only the in-window controls work."),
    ]),
    # F_VLC_4D: bool key. src writes 1 (vlcrc:299) — only target=0 is a real
    # flip; target=1 would be TRIVIAL_PASS. Cap to single non-trivial Param.
    FileTask(F_VLC_4D, "set_play_and_exit", "config_setting", params=[
        _vlcrc_param(
            vlcrc_key="play-and-exit", target_value=0,
            func="check_play_and_exit",
            rule_key="expected_play_and_exit", rule_value=0,
            instr="VLC has been auto-closing on me right after a clip ends "
                  "and I keep losing the window. Please turn off the "
                  "play-and-exit behaviour so VLC stays open after the "
                  "current playlist finishes."),
    ]),
    # F_VLC_4E: bool key. src writes 1 (vlcrc:304) — only target=0 is a real
    # flip; target=1 would be TRIVIAL_PASS. Cap to single non-trivial Param.
    FileTask(F_VLC_4E, "set_one_instance", "config_setting", params=[
        _vlcrc_param(
            vlcrc_key="one-instance-when-started-from-file", target_value=0,
            func="check_one_instance_when_started_from_file",
            rule_key="expected_one_instance_when_started_from_file",
            rule_value=0,
            instr="I want to compare two videos side-by-side and need each "
                  "to open in its own VLC window. Please turn off VLC's "
                  "'allow only one running instance when started from file "
                  "manager' behaviour so double-clicking a video spawns a "
                  "fresh VLC process every time instead of reusing the "
                  "existing window."),
    ]),

    # ---- Loop 1b: broaden prefs_vlcrc ---------------------------
    # Eval has prefs_vlcrc at 47% / 53% (post-infeasibility); synth at 24%.
    # Each FileTask below reuses an EXISTING check_qt_* / is_vlc_recordings_*
    # evaluator func from upstream (verified present in
    # .venv/lib/python3.12/site-packages/desktop_env/evaluators/metrics/vlc.py).
    # New File instances (F_VLC_2X/3X/4F/4G) carry distinct INITIAL values
    # so each Param target is a non-trivial flip from the seed (no
    # trivial-pass guard violation).
    # validation: dropped Param[1] (175→250 value rotation, same key/skill).
    FileTask(F_VLC_2X, "set_max_volume_alt", "config_setting", params=[
        _vlcrc_param(vlcrc_key="qt-max-volume", target_value=175,
                     func="check_qt_max_volume",
                     rule_key="expected_qt_max_volume", rule_value=175,
                     instr="I'm previewing a quietly-mastered audiobook "
                           "tonight and the existing 200% ceiling is "
                           "overkill. Could you set VLC's maximum displayed "
                           "volume to 175% so the slider tops out at a more "
                           "reasonable boost?"),
    ]),
    # validation: dropped Param[1] (path-axis rotation on same vlcrc key).
    FileTask(F_VLC_3X, "set_recordings_folder_alt", "config_setting", params=[
        _vlcrc_param(vlcrc_key="input-record-path",
                     target_value="/home/user/Videos",
                     func="is_vlc_recordings_folder",
                     rule_key="recording_file_path",
                     rule_value="/home/user/Videos",
                     instr="I'm consolidating all my video captures under "
                           "one folder. Please redirect VLC's recordings "
                           "destination to /home/user/Videos so anything I "
                           "capture from a stream lands in the same place "
                           "as my other video files."),
    ]),
    # validation: dropped Param[1] (blackish-rule variant; the blackish rule is
    # already covered by F_VLC_4B.set_slider_colours_blackish and
    # F_VLC_4I.set_slider_colours_blackish_alt — keeping the match-rule
    # palette here preserves the rule-type diversity from F_VLC_4B.
    FileTask(F_VLC_4F, "set_slider_colours_alt", "config_setting", params=[
        Param(
            gold_args={"vlcrc_key": "qt-slider-colours",
                       "target_value":
                           "255;128;64;255;255;200;90;180;255;30;30;30"},
            eval_kind="vlcrc_kv",
            eval_args={
                "func": "check_qt_slider_colours",
                "rule": {"type": "match",
                         "expected_qt_slider_colours":
                             "255;128;64;255;255;200;90;180;255;30;30;30"},
            },
            instr="I'm matching VLC's playback bar to a warm-sunset theme "
                  "I'm using elsewhere on the desktop. Please change VLC's "
                  "slider colours to "
                  "(255, 128, 64), (255, 255, 200), (90, 180, 255), "
                  "(30, 30, 30) so the seek bar picks up the same accent "
                  "palette."),
    ]),
    FileTask(F_VLC_4G, "enable_bgcone", "config_setting", params=[
        _vlcrc_param(vlcrc_key="qt-bgcone", target_value=1,
                     func="check_qt_bgcone",
                     rule_key="expected_qt_bgcone", rule_value=1,
                     instr="I actually like the orange traffic-cone artwork "
                           "VLC shows when nothing is playing — it's a "
                           "friendly visual anchor while I queue the next "
                           "track. Please make sure VLC's background-cone "
                           "icon is enabled so the splash artwork appears "
                           "again on the idle screen."),
    ]),
    # validation (eval mirror osworld_vlc_386dbd0e): bind global play/pause
    # to a real hotkey. Eval = check_global_key_play_pause with
    # expected_global_key_play_pause=1 (any non-empty rebind passes).
    # validation: dropped Param[1] (hotkey-value rotation Space vs Ctrl+Alt+P
    # — same vlcrc key, eval ref is just "non-empty"; both rows scored same
    # rule_value=1, so it was a near-clone on the eval surface).
    FileTask(F_VLC_4H, "bind_global_play_pause", "config_setting", params=[
        _vlcrc_param(
            vlcrc_key="global-key-play-pause", target_value="Space",
            func="check_global_key_play_pause",
            rule_key="expected_global_key_play_pause", rule_value=1,
            instr="I'm reading a lecture PDF while a music video runs in "
                  "VLC, but I keep having to refocus the player every time "
                  "I want to pause. Could you change VLC's settings so I "
                  "can pause/start the video with a keyboard shortcut "
                  "without bringing the player to the front? I want to "
                  "stay focused on the PDF."),
    ]),
    # validation (eval mirror osworld_vlc_d06f0d4d): second blackish-slider
    # row over a different seed so the `type=blackish` rule branch gets
    # broader prefs_vlcrc coverage without duplicating F_VLC_4B's slot-2.
    # validation: dropped Param[1] (palette rotation; same blackish-rule eval).
    FileTask(F_VLC_4I, "set_slider_colours_blackish_alt", "config_setting",
             params=[
        Param(
            gold_args={"vlcrc_key": "qt-slider-colours",
                       "target_value":
                           "30;30;30;25;25;25;20;20;20;10;10;10"},
            eval_kind="vlcrc_kv",
            eval_args={
                "func": "check_qt_slider_colours",
                "rule": {"type": "blackish"},
            },
            instr="Can you change the colour of VLC's playback slider to "
                  "a blackish tone? I often use the player in a low-light "
                  "environment and a darker scheme is easier on my eyes "
                  "at night.",
        ),
    ]),

    # ---- Loop 2: .m3u playlist files --------------------------------------
    # Dropped (synth/vlc.md): eval has 0% file_m3u coverage
    # (`check_list`) while synth was carrying 11% (6 rows). The three m3u
    # FileTasks below (F_VLC_5 save_morning_drive_m3u, F_VLC_6
    # save_study_session_m3u, F_VLC_7 save_podcast_queue_m3u) are dropped
    # to free row budget for prefs_vlcrc + remote-URL playback templates
    # (where eval coverage is actually concentrated). The placeholder File
    # instances (F_VLC_5/6/7) stay defined above for future re-introduction
    # if upstream eval ever adds m3u coverage.
    # ---- Loop 3: synthesized AUDIO source + audio-extract / compare -------
    # validation: dropped Param[1] (only agent_basename differed audio.mp3 vs
    # soundtrack.mp3 — paraphrase clone on identical ffmpeg gold command).
    FileTask(F_VLC_9, "extract_audio_mp3", "media_transform", params=[
        _media_compare_param(
            ext="mp3", file_id="F-VLC-9", task_id="extract_audio_mp3",
            tag="extract",
            gold_ffmpeg_cmd=(
                "rm -f '{gold}' && ffmpeg -y -hide_banner -loglevel error "
                "-i '{src}' -vn -acodec libmp3lame -q:a 4 '{gold}'"
            ),
            agent_basename="audio.mp3", func="compare_audios",
            instr="I want the song from this music video on my phone "
                  "without lugging the full video around. Please use VLC "
                  "to extract just the audio track from the currently-open "
                  "clip and save it on the Desktop as audio.mp3 so I can "
                  "sync it later."),
    ]),
    # validation: dropped Param[1] (agent_basename-axis variant, same gold cmd).
    FileTask(F_VLC_10, "extract_audio_purple", "media_transform", params=[
        _media_compare_param(
            ext="mp3", file_id="F-VLC-10", task_id="extract_audio_purple",
            tag="purple",
            gold_ffmpeg_cmd=(
                "rm -f '{gold}' && ffmpeg -y -hide_banner -loglevel error "
                "-i '{src}' -vn -acodec libmp3lame -q:a 4 '{gold}'"
            ),
            agent_basename="purple_track.mp3", func="compare_audios",
            instr="I'm pulling the audio from a colour-graded test clip "
                  "for a sound-design review. Please use VLC to save the "
                  "audio track of the currently-loaded clip on the Desktop "
                  "as purple_track.mp3 in MP3 format."),
    ]),
    # validation: dropped Param[1] (timestamp + basename rotation on same skill).
    FileTask(F_VLC_10, "snapshot_t1_purple", "media_transform", params=[
        _snapshot_param(snapshot_seconds=1, agent_basename="purple_frame.png",
                        file_id="F-VLC-10", task_id="snapshot_t1_purple",
                        instr="I'm extracting a still for a colour-grading "
                              "moodboard. Please pause the currently-loaded "
                              "video at the 1-second mark and have VLC save "
                              "a snapshot of that frame on the Desktop as "
                              "purple_frame.png."),
    ]),
    # ---- Loop 4: synthesized VIDEO + snapshot / video-filter --------------
    # validation: dropped Param[1] (same snapshot_seconds=2, just different
    # agent_basename interstellar.png vs snap_t2.png — paraphrase clone).
    FileTask(F_VLC_13, "snapshot_t2", "media_transform", params=[
        _snapshot_param(snapshot_seconds=2, agent_basename="interstellar.png",
                        file_id="F-VLC-13", task_id="snapshot_t2",
                        instr="I'm grabbing a hero still for a film blog "
                              "header. Please pause the video on the 2-second "
                              "mark and have VLC save a snapshot of that "
                              "frame on the Desktop as interstellar.png."),
    ]),
    # validation: dropped Param[1] (t=4 vs t=3 timestamp variation, same skill).
    FileTask(F_VLC_14, "snapshot_t4", "media_transform", params=[
        _snapshot_param(snapshot_seconds=4, agent_basename="snap_t4.png",
                        file_id="F-VLC-14", task_id="snapshot_t4",
                        instr="I'm sampling frames every couple of seconds "
                              "for a slideshow contact sheet. Please "
                              "snapshot the video frame at t=4s and save "
                              "it as snap_t4.png on the Desktop."),
    ]),
    # validation: dropped Param[1] (only agent_basename differed — gold cmd
    # identical, eval shape identical).
    FileTask(F_VLC_14, "rotate_90", "media_transform", params=[
        _media_compare_param(
            ext="mp4", file_id="F-VLC-14", task_id="rotate_90",
            tag="rot90",
            gold_ffmpeg_cmd=(
                "rm -f '{gold}' && ffmpeg -y -hide_banner -loglevel error "
                "-i '{src}' -vf rotate=PI/2 -c:v libx264 -pix_fmt yuv420p "
                "-preset ultrafast '{gold}'"
            ),
            agent_basename="rotated.mp4", func="compare_videos",
            instr="My phone recorded a clip on its side and the playback is "
                  "sideways. Please use VLC to rotate this video 90 degrees "
                  "clockwise and export the corrected version to the Desktop "
                  "as rotated.mp4."),
    ]),
    # f_vlc_16__trim_5s: uses `compare_videos_full_duration` (custom helper
    # in eval/metrics.py) instead of upstream `compare_videos`. Upstream's
    # `max_frames_to_check=100 / fps=25 = 4s` sampling cap means any video
    # ≥4 s passes the duration check trivially — defeating the trim task
    # because the agent's no-op full clip and the gold trimmed clip both
    # exit the pHash loop before EOF differences register. The full-duration
    # variant first compares container durations via cv2 then defers to
    # upstream for pHash on the agreed length.
    # validation: dropped Param[1] (-t 5 vs -t 3 duration value rotation,
    # same trim skill).
    FileTask(F_VLC_16, "trim_5s", "media_transform", params=[
        _media_compare_param(
            ext="mp4", file_id="F-VLC-16", task_id="trim_5s",
            tag="trim5",
            gold_ffmpeg_cmd=(
                "rm -f '{gold}' && ffmpeg -y -hide_banner -loglevel error "
                "-ss 0 -t 5 -i '{src}' -c:v libx264 -pix_fmt yuv420p "
                "-preset ultrafast '{gold}'"
            ),
            agent_basename="clip.mp4", func="compare_videos_full_duration",
            instr="I'm pulling a teaser for a social-media post and need "
                  "just the opening of this footage. Please use VLC to trim "
                  "this video down to its first 5 seconds and save the "
                  "shortened clip on the Desktop as clip.mp4."),
    ]),
    # validation: dropped Param[1] (t=6 vs t=8, same snapshot skill).
    FileTask(F_VLC_16, "snapshot_t6_cyan", "media_transform", params=[
        _snapshot_param(snapshot_seconds=6, agent_basename="cyan_t6.png",
                        file_id="F-VLC-16", task_id="snapshot_t6_cyan",
                        instr="I'm capturing a cyan-themed frame for my "
                              "colour palette notes. Please snapshot the "
                              "video frame at t=6s and save it as cyan_t6."
                              "png on the Desktop."),
    ]),

    # ---- Loop 5: live-playback / fullscreen / subtitle gap-filler ---------
    # validation: dropped Param[1] in each play/fullscreen task — sample vs
    # sample_b basename rotation was a paraphrase clone (same eval skill,
    # same kind, just a different filename on Desktop).
    FileTask(F_VLC_17, "play_local_video", "is_vlc_playing", params=[
        # validation reframe (eval mirror osworld_vlc_59f21cfb): tight
        # 1-sentence question voice — "Could you play the music video
        # that's saved on my desktop for me via vlc?" — replacing the
        # multi-sentence first-person explainer.
        _play_local_param(basename="sample.mp4", kind="video",
                          instr="Could you play the sample.mp4 file saved "
                                "on my desktop for me via vlc?"),
    ]),
    FileTask(F_VLC_17, "fullscreen_sample", "is_vlc_fullscreen", params=[
        _fullscreen_param(basename="sample.mp4",
                          instr="I'm presenting a quick sample clip to "
                                "colleagues and the window is tiny. Could "
                                "you open sample.mp4 from the Desktop in VLC "
                                "and switch the player into fullscreen mode "
                                "so the video fills the screen?"),
    ]),
    FileTask(F_VLC_18, "play_local_video_demo", "is_vlc_playing", params=[
        _play_local_param(basename="demo_clip.mp4", kind="video",
                          instr="I'm rehearsing a product demo and need to "
                                "double-check the master clip. Please play "
                                "demo_clip.mp4 from the Desktop in VLC so "
                                "I can review the timing end-to-end."),
    ]),
    FileTask(F_VLC_18, "fullscreen_demo", "is_vlc_fullscreen", params=[
        _fullscreen_param(basename="demo_clip.mp4",
                          instr="I'm about to project the demo on a meeting "
                                "screen and need the whole display. Please "
                                "play demo_clip.mp4 from the Desktop in VLC "
                                "and switch the window to fullscreen so the "
                                "video covers the monitor."),
    ]),
    FileTask(F_VLC_19, "play_local_audio", "is_vlc_playing", params=[
        _play_local_param(basename="song.mp3", kind="audio",
                          instr="I just exported a song.mp3 from my DAW and "
                                "want to hear it on a different player to "
                                "double-check the mix. Please open VLC and "
                                "play song.mp3 from the Desktop."),
    ]),
    FileTask(F_VLC_20, "fullscreen_movie", "is_vlc_fullscreen", params=[
        _fullscreen_param(basename="movie.mp4",
                          instr="I'm settling in for movie night and want "
                                "the full screen, not a tiny window. Please "
                                "switch the VLC window playing movie.mp4 "
                                "into fullscreen mode so the video covers "
                                "the entire monitor."),
    ]),
    # Pruned (rebalance OVER, eval_class=is_vlc_playing):
    # FileTask(F_VLC_20, "play_local_movie", "is_vlc_playing", params=[
        # _play_local_param(basename="movie.mp4", kind="video",
                          # instr="Movie night is starting and the file is "
                                # "already on the Desktop. Could you open "
                                # "movie.mp4 from the Desktop with VLC and "
                                # "start playback so we can settle in?"),
        # _play_local_param(basename="movie_trailer.mp4", kind="video",
                          # instr="Use VLC to play the movie_trailer.mp4 "
                                # "file on the Desktop so the preview clip "
                                # "runs end-to-end before anyone commits to "
                                # "the full feature tonight."),
    # ]),
    # Loop-2 broaden: subtitle (.srt) + media combo. The pre_config plants
    # both lecture.mp4 AND lecture.srt on the Desktop; the agent opens the
    # mp4 in VLC. Eval is `is_vlc_playing` on lecture.mp4 (the .srt is
    # auto-loaded as a sidecar but eval doesn't probe subtitles directly).
    # Pruned (rebalance OVER, eval_class=is_vlc_playing):
    # FileTask(F_VLC_21, "play_lecture_with_srt", "is_vlc_playing", params=[
        # _play_local_param(basename="lecture.mp4", kind="video",
                          # instr="I'm catching up on a recorded lecture and "
                                # "need the captions for the technical bits. "
                                # "Please play lecture.mp4 from the Desktop "
                                # "in VLC — there's a matching lecture.srt "
                                # "sidecar on the Desktop so VLC should "
                                # "auto-load the subtitles."),
        # _play_local_param(basename="lecture_part2.mp4", kind="video",
                          # instr="I'm moving on to the second half of the "
                                # "lecture series tonight. Could you open "
                                # "lecture_part2.mp4 (on the Desktop) with "
                                # "VLC so the video starts playing alongside "
                                # "its sidecar subtitles file?"),
    # ]),
    # Pruned (rebalance OVER, eval_class=is_vlc_fullscreen):
    # FileTask(F_VLC_21, "fullscreen_lecture_with_srt", "is_vlc_fullscreen",
             # params=[
        # _fullscreen_param(basename="lecture.mp4",
                          # instr="I'm watching the lecture on a small monitor "
                                # "and the captions are too tiny. Could you "
                                # "open lecture.mp4 from the Desktop in VLC "
                                # "and switch into fullscreen mode (its "
                                # "lecture.srt sidecar is on the Desktop too) "
                                # "so the subtitles are readable?"),
        # _fullscreen_param(basename="lecture_part2.mp4",
                          # instr="I want a focused view of the lecture's "
                                # "second half on the big screen. Please play "
                                # "lecture_part2.mp4 (Desktop) in VLC and "
                                # "toggle fullscreen so the video covers the "
                                # "screen."),
    # ]),

    # ---- Loop 6 (validation P2): network stream playback --------------------
    # Eval mirror: 59f21cfb / bba3381f-style is_vlc_playing tasks that probe
    # a streamed http URL in VLC's play state. Pre_config builds a local
    # mp3 + spins up a python http.server so the agent has a reachable
    # URL without depending on external network access (F10 mitigation).
    # Pruned (rebalance OVER, eval_class=is_vlc_playing):
    # FileTask(F_VLC_22, "play_http_stream", "is_vlc_playing", params=[
        # _stream_param(url="http://localhost:48080/stream_sample.mp3",
                      # instr="I want to test a podcast URL by streaming it "
                            # "straight into VLC instead of downloading it "
                            # "first. Could you open Media → Open Network "
                            # "Stream in VLC and play "
                            # "http://localhost:48080/stream_sample.mp3 so "
                            # "I can hear the live feed?"),
        # _stream_param(url="http://localhost:48080/stream_alt.mp3",
                      # instr="I'm comparing two network audio feeds for a "
                            # "broadcasting review and want them played live, "
                            # "not as local files. Please use VLC's Media → "
                            # "Open Network Stream to play "
                            # "http://localhost:48080/stream_alt.mp3 from the "
                            # "local test server."),
    # ]),

    # ---- Loop 7 (validation P3): cross-app frame → wallpaper ----------------
    # Eval mirror: efcf0d81 / fba2c100-style compare_images over vm_wallpaper
    # — agent grabs a frame from a VLC video at timestamp T and sets it as
    # the GNOME desktop background (gsettings org.gnome.desktop.background
    # picture-uri). pre_config builds the source mp4 + a gold png frame at
    # the same timestamp; oracle calls gsettings set picture-uri to the
    # gold png; eval reads vm_wallpaper (the current desktop wallpaper) and
    # compare_images it against the gold.
    # validation: dropped Param[1] in wallpaper tasks (timestamp+basename
    # rotation, same skill). Both FileTasks under F_VLC_23 retained so the
    # compare_images bucket keeps two distinct bases.
    FileTask(F_VLC_23, "frame_to_wallpaper", "compare_images", params=[
        # validation reframe (eval mirror osworld_vlc_efcf0d81): tight,
        # 1-sentence eval-voice — "make this part of the video my
        # computer's background picture" — instead of multi-sentence
        # explainer voice. Same gold/oracle/eval mechanics; only the
        # instruction wording changes to match eval distribution.
        _wallpaper_param(snapshot_seconds=5,
                         agent_basename="vlc_frame_wallpaper.png",
                         file_id="F-VLC-23",
                         task_id="frame_to_wallpaper",
                         instr="Make this part of the video my computer's "
                               "background picture."),
    ]),
    FileTask(F_VLC_23, "frame_to_wallpaper_alt", "compare_images", params=[
        _wallpaper_param(snapshot_seconds=18,
                         agent_basename="scenic_wallpaper_mid.png",
                         file_id="F-VLC-23",
                         task_id="frame_to_wallpaper_alt",
                         instr="I want a mid-clip frame from this scenic "
                               "footage as my desktop wallpaper. Please "
                               "have VLC snapshot the video frame at the "
                               "18-second mark, save it on the Desktop as "
                               "scenic_wallpaper_mid.png, then apply that "
                               "image as the GNOME desktop background."),
    ]),
    # validation: dropped Param[1] (timestamp 3→4, basename swap).
    FileTask(F_VLC_13, "extract_frame_to_file", "compare_images", params=[
        _snapshot_param(snapshot_seconds=3,
                        agent_basename="interstellar_frame.png",
                        file_id="F-VLC-13",
                        task_id="extract_frame_to_file",
                        instr="Snap a photo of the current video scene at "
                              "the 3-second mark, save it as "
                              "'interstellar_frame.png', and put it on the "
                              "Desktop, please."),
    ]),
    # Added #3 (config_setting variant, eval mirror d06f0d4d slider
    # colours `blackish` branch). F_VLC_4B slot-2 — same vlcrc key as the
    # slot-1 `set_slider_colours` task but exercises the OTHER rule-shape
    # supported by check_qt_slider_colours (`type=blackish` requires every
    # RGB triplet to have all three channels < 100; see vlc.py:435-454).
    # The target_value packs four all-blackish triplets so both rule.type
    # branches are now structurally represented in the synth set.
    # validation: dropped Param[1] (palette rotation, same blackish-rule eval).
    FileTask(F_VLC_4B, "set_slider_colours_blackish", "config_setting", params=[
        Param(
            gold_args={"vlcrc_key": "qt-slider-colours",
                       "target_value":
                           "10;20;30;40;50;60;70;80;90;0;0;0"},
            eval_kind="vlcrc_kv",
            eval_args={
                "func": "check_qt_slider_colours",
                "rule": {"type": "blackish"},
            },
            instr="I want a fully blacked-out seek bar so the slider "
                  "blends into a dark theme. Please change VLC's slider "
                  "colours to the dim palette "
                  "(10, 20, 30), (40, 50, 60), (70, 80, 90), (0, 0, 0) "
                  "so every channel of every triplet stays below 100."),
    ]),
    # Added #4 (compare_audios variant, eval mirror 8f080098 audio
    # extract). F_VLC_9 slot-2 reuses the red+sine source mp4 already
    # planted by the File's src. Distinct codec/extension (libvorbis ogg)
    # from the slot-1 mp3 task → structurally distinct media-compare gold
    # at the codec axis without requiring a new File.
    # validation: dropped Param[1] (agent_basename rotation on same gold cmd).
    FileTask(F_VLC_9, "extract_audio_ogg", "media_transform", params=[
        _media_compare_param(
            ext="ogg", file_id="F-VLC-9", task_id="extract_audio_ogg",
            tag="ogg",
            gold_ffmpeg_cmd=(
                "rm -f '{gold}' && ffmpeg -y -hide_banner -loglevel error "
                "-i '{src}' -vn -c:a libvorbis -q:a 4 '{gold}'"
            ),
            agent_basename="audio_track.ogg", func="compare_audios",
            instr="I prefer the OGG/Vorbis codec for my offline music "
                  "library because of its open licensing. Could you have "
                  "VLC re-encode the audio from this clip and save the "
                  "result on the Desktop as audio_track.ogg in OGG/Vorbis "
                  "format?"),
    ]),

    # ---- Loop 8 (eval mirror osworld_vlc_aa4b5023):
    # rotate video + Save-As at user-named file path under /home/user/.
    # The Save-As filename is part of the user instruction (mirroring
    # aa4b5023's "save it with the name 1984_Apple_Macintosh_Commercial.mp4"
    # phrasing). Eval = compare_videos on the rotated gold; oracle copies
    # gold to /home/user/<filename>.mp4 so the agent's expected sink lands
    # at the exact path the eval probes.
    # validation: dropped Param[1] (Save-As filename rotation, same rotate
    # gold cmd — paraphrase clone keeping the eval-mirror Param[0]).
    FileTask(F_VLC_28, "rotate_save_as", "media_transform", params=[
        # Param built inline so agent_path lives under /home/user/, not
        # /home/user/Desktop/ (matches eval row aa4b5023 result path).
        Param(
            gold_args={
                "gold_path": "/tmp/gold_f_vlc_28__rotate_save_as_rot.mp4",
                "gold_ffmpeg_cmd": (
                    "rm -f '{gold}' && ffmpeg -y -hide_banner -loglevel error "
                    "-i '{src}' -vf rotate=PI/2 -c:v libx264 -pix_fmt yuv420p "
                    "-preset ultrafast '{gold}'"
                ),
                "agent_path":
                    "/home/user/1984_Apple_Macintosh_Commercial.mp4",
            },
            eval_kind="media_compare",
            eval_args={"func": "compare_videos",
                       "gold_path": "/tmp/gold_f_vlc_28__rotate_save_as_rot.mp4",
                       "gold_basename":
                           "gold_f_vlc_28__rotate_save_as_rot.mp4",
                       "agent_basename":
                           "1984_Apple_Macintosh_Commercial.mp4"},
            instr="Hey, could you turn this video the right way up for me? "
                  "And once it's flipped around, could you save it with "
                  "the name '1984_Apple_Macintosh_Commercial.mp4' in my "
                  "home folder where all my files are?",
        ),
    ]),

    # ---- Loop 6b: remote-URL playback --------------------------
    # Mirrors upstream eval rows (bba3381f HLS, 59f21cfb HTTP). Eval =
    # `is_vlc_playing` with rule.type=url. Oracle pkill+relaunches VLC
    # with the public URL as positional argv and the HTTP iface enabled
    # so the lite runner can curl status.xml. No local asset is built —
    # the URLs hit public CDNs verified reachable at synth-gen time.
    # validation: dropped Param[1] in each remote-stream task (URL rotation
    # within the same CDN family — same is_vlc_playing eval shape; classed
    # as numeric/value-axis variant rather than a distinct skill).
    # F-VLC-24 (Blender MP4 over HTTP) is omitted: the stream takes longer than
    # the post-launch settle window VLC needs before `is_vlc_playing` observes
    # `state=playing`.
    FileTask(F_VLC_25, "play_remote_hls_apple", "is_vlc_playing", params=[
        _stream_param(
            url=("https://devstreaming-cdn.apple.com/videos/streaming/"
                 "examples/img_bipbop_adv_example_fmp4/master.m3u8"),
            instr="Can you start streaming the HLS video from this link "
                  "for me? "
                  "https://devstreaming-cdn.apple.com/videos/streaming/"
                  "examples/img_bipbop_adv_example_fmp4/master.m3u8 — I "
                  "want to see how VLC handles Apple's BipBop reference "
                  "playlist live, without saving it first."),
    ]),
    FileTask(F_VLC_26, "play_remote_hls_mux", "is_vlc_playing", params=[
        _stream_param(
            url="https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
            instr="I'm benchmarking VLC's adaptive-bitrate behaviour "
                  "against Mux's public HLS test stream. Could you open "
                  "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8 in "
                  "VLC as a network stream so the playback runs straight "
                  "from the CDN without any local download?"),
    ]),
    FileTask(F_VLC_27, "play_remote_http_audio", "is_vlc_playing", params=[
        # validation: local http.server URL (see F_VLC_27 File def). Plain MP3
        # over http://localhost:48080 — no external CDN, no 302-redirect race,
        # so VLC reports state=playing well within the eval window.
        _stream_param(
            url="http://localhost:48080/mpthreetest.mp3",
            instr="I'm checking whether a streamed MP3 plays back cleanly "
                  "through VLC before I add it to a podcast playlist. "
                  "Please open Media → Open Network Stream in VLC and play "
                  "http://localhost:48080/mpthreetest.mp3 so the audio plays "
                  "live from the URL rather than a local file."),
    ]),
]


# §I.f — Emission.
TEMPLATES.extend(_emit_templates(FILE_TASKS))
