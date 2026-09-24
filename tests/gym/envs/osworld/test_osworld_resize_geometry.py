"""OSWorld live-step coordinate geometry tests."""

from __future__ import annotations

import pytest

W = 1920
H = 1080


def test_osworld_live_step_path_uses_the_truncating_mode() -> None:
    from lite.gym.envs.osworld.main import to_pyautogui

    assert to_pyautogui("click", {"coordinate": [0, 0]}, W, H) == (
        "pyautogui.click(0, 0, button='left')"
    )
    assert to_pyautogui("click", {"coordinate": [999, 999]}, W, H) == (
        "pyautogui.click(1918, 1078, button='left')"
    )
    assert to_pyautogui("mouse_move", {"coordinate": [333, 667]}, W, H) == (
        "pyautogui.moveTo(639, 720)"
    )
    assert to_pyautogui("click", {"coordinate": [1000, 1000]}, W, H) == (
        "pyautogui.click(1919, 1079, button='left')"
    )


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("click", {"coordinate": [1069, 605]}),
        ("mouse_move", {"coordinate": [-1, 500]}),
        ("scroll", {"coordinate": [500, 1001]}),
        ("drag", {"coordinate": [1000.1, 500]}),
        ("drag", {"coordinate": [500, 500], "start_coordinate": [500, -0.1]}),
    ],
)
def test_osworld_rejects_out_of_range_coordinates(name, arguments) -> None:
    from lite.gym.envs.osworld.actions import to_pyautogui

    with pytest.raises(ValueError, match=r"normalized to \[0, 1000\]"):
        to_pyautogui(name, arguments, W, H)
