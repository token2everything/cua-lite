"""Hand-curated oracle metadata for vlc eval tasks.

VLC media-player tasks.

ORACLES maps OSWorld task UUID → curated entry. Recognised entry keys:
- "actions": list[dict]   — replay-style oracle steps; absence ⇒ this task has no oracle
- "after_postconfig": bool — when True, oracle replays AFTER evaluator.postconfig
- "exclude_reason": str   — canonical reasons not derivable from upstream
                            (manual env-specific blocks; infeasible/google_auth
                            live in __main__.py frozensets)
- "evaluator": dict       — full evaluator replacement (use sparingly: only for
                            principled platform overrides like XFCE-vs-GNOME
                            APIs, or upstream-data corrections like a wrong
                            URL or missing file — applied AFTER _rewrite, so
                            the dict here matches the rewritten form).

Stats at extraction time:
- total entries:           15
- with actions:            15
- after_postconfig=True:   0
- block: exclude_reason:   0
- evaluator override:      0
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {
    # FEASIBLE (validated oracle passes in container): relaunch VLC w/ HTTP iface streaming the Apple CDN HLS URL.
    'bba3381f-b5eb-4439-bd9e-80c22218d5a7': {'actions': [{'type': 'execute', 'parameters': {'command': 'pkill -9 -f vlc 2>/dev/null; sleep 2; true', 'shell': True}}, {'type': 'launch', 'parameters': {'command': "VLC_VERBOSE=-1 vlc --extraintf http --http-password password --no-video-title-show --no-audio 'https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_fmp4/master.m3u8'", 'shell': True}}, {'type': 'sleep', 'parameters': {'seconds': 15}}]},
   '215dfd39-f493-4bc3-a027-8a97d72c61bf': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/vlc',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'timeout 3 vlc --intf '
                                                                                                'dummy --reset-config '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sed -i '
                                                                                                "'s/^#\\?qt-bgcone=.*/qt-bgcone=0/' "
                                                                                                '/home/user/.config/vlc/vlcrc',
                                                                                     'shell': True}}]},
    '386dbd0e-0241-4a0a-b6a2-6704fba26b1c': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/vlc',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'printf '
                                                                                                "'global-key-play-pause=Space\\n' "
                                                                                                '> '
                                                                                                '/home/user/.config/vlc/vlcrc',
                                                                                     'shell': True}}]},
    '59f21cfb-0120-4326-b255-a5b827b38967': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f vlc; '
                                                                                                'sleep 2',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'VLC_VERBOSE=-1 vlc '
                                                                                                '--extraintf http '
                                                                                                '--http-password password '
                                                                                                '--no-video-title-show '
                                                                                                '--no-audio '
                                                                                                "'/home/user/Desktop/Rick "
                                                                                                'Astley - Never Gonna '
                                                                                                'Give You Up (Official '
                                                                                                "Music Video).mp4'",
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '5ac2891a-eacd-4954-b339-98abba077adb': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/vlc',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'test -f '
                                                                                                '/home/user/.config/vlc/vlcrc '
                                                                                                '|| timeout 3 vlc '
                                                                                                '--intf dummy '
                                                                                                '--reset-config '
                                                                                                '2>/dev/null || true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sed -i '
                                                                                                "'s/^#\\?play-and-exit=.*/play-and-exit=0/' "
                                                                                                '/home/user/.config/vlc/vlcrc',
                                                                                     'shell': True}}]},
    '8ba5ae7a-5ae5-4eab-9fcc-5dd4fe3abf89': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/vlc',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'test -f '
                                                                                                '/home/user/.config/vlc/vlcrc '
                                                                                                '|| timeout 5 vlc '
                                                                                                '--intf dummy '
                                                                                                '--play-and-exit '
                                                                                                '/dev/null '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'if grep -q '
                                                                                                "'input-record-path=' "
                                                                                                '/home/user/.config/vlc/vlcrc '
                                                                                                '2>/dev/null; then sed '
                                                                                                '-i '
                                                                                                "'s|^#\\?input-record-path=.*|input-record-path=/home/user/Desktop|' "
                                                                                                '/home/user/.config/vlc/vlcrc; '
                                                                                                'else echo '
                                                                                                "'input-record-path=/home/user/Desktop' "
                                                                                                '>> '
                                                                                                '/home/user/.config/vlc/vlcrc; '
                                                                                                'fi',
                                                                                     'shell': True}}]},
    '8d9fd4e2-6fdb-46b0-b9b9-02f06495c62f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 vlc '
                                                                                                '2>/dev/null; sleep 2\n'
                                                                                                'mkdir -p '
                                                                                                '/home/user/.config/vlc\n'
                                                                                                'cat > '
                                                                                                '/home/user/.config/vlc/vlcrc '
                                                                                                "<< 'VLCEOF'\n"
                                                                                                '[qt]\n'
                                                                                                'qt-privacy-ask=0\n'
                                                                                                'VLCEOF\n'
                                                                                                'echo done',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'DISPLAY=:1 vlc '
                                                                                                '--no-audio '
                                                                                                '--no-video-title-show '
                                                                                                '--start-time=15 '
                                                                                                "'/home/user/Desktop/Gen "
                                                                                                "2.mp4'",
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'WID=$(xdotool search '
                                                                                                '--class vlc '
                                                                                                '2>/dev/null | head '
                                                                                                '-1); if [ -n "$WID" '
                                                                                                ']; then   xdotool '
                                                                                                'windowactivate $WID '
                                                                                                '2>/dev/null; sleep '
                                                                                                '1;   xdotool key f; '
                                                                                                'fi',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '8f080098-ddb1-424c-b438-4e96e5e4786e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Baby "
                                                                                                "Justin Bieber.mp3' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/vlc/8f080098-ddb1-424c-b438-4e96e5e4786e/Baby%20Justin%20Bieber.mp3'",
                                                                                     'shell': True}}]},
    '9195653c-f4aa-453d-aa95-787f6ccfaae9': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/vlc',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'timeout 3 vlc --intf '
                                                                                                'dummy --reset-config '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sed -i '
                                                                                                "'s/^#\\?qt-max-volume=.*/qt-max-volume=200/' "
                                                                                                '/home/user/.config/vlc/vlcrc',
                                                                                     'shell': True}}]},
    'a5bbbcd5-b398-4c91-83d4-55e1e31bbb81': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/vlc',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'timeout 3 vlc --intf '
                                                                                                'dummy --reset-config '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sed -i '
                                                                                                "'s/^#\\?qt-minimal-view=.*/qt-minimal-view=1/' "
                                                                                                '/home/user/.config/vlc/vlcrc',
                                                                                     'shell': True}}]},
    'aa4b5023-aef6-4ed9-bdc9-705f59ab9ad6': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "mkdir -p '/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/1984_Apple_Macintosh_Commercial.mp4' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/vlc/aa4b5023-aef6-4ed9-bdc9-705f59ab9ad6/1984_Apple_Macintosh_Commercial.mp4'",
                                                                                     'shell': True}}]},
    'bba3381f-b5eb-4439-bd9e-80c22218d5a7': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f vlc; '
                                                                                                'sleep 2; true',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'VLC_VERBOSE=-1 vlc '
                                                                                                '--extraintf http '
                                                                                                '--http-password password '
                                                                                                '--no-video-title-show '
                                                                                                '--no-audio '
                                                                                                "'https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_fmp4/master.m3u8'",
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'd06f0d4d-2cd5-4ede-8de9-598629438c6e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/vlc',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'timeout 3 vlc --intf '
                                                                                                'dummy --reset-config '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'if grep -q '
                                                                                                "'qt-slider-colours=' "
                                                                                                '/home/user/.config/vlc/vlcrc '
                                                                                                '2>/dev/null; then sed '
                                                                                                '-i '
                                                                                                "'s|^#\\?qt-slider-colours=.*|qt-slider-colours=20;20;20;15;15;15;10;10;10;5;5;5|' "
                                                                                                '/home/user/.config/vlc/vlcrc; '
                                                                                                'else echo '
                                                                                                "'qt-slider-colours=20;20;20;15;15;15;10;10;10;5;5;5' "
                                                                                                '>> '
                                                                                                '/home/user/.config/vlc/vlcrc; '
                                                                                                'fi',
                                                                                     'shell': True}}]},
    # efcf0d81 (VLC wallpaper): gsettings is the authoritative wallpaper store
    # on the GNOME base (gnome-settings-daemon honours
    # `org.gnome.desktop.background`) — the official VM's `xfconf-query
    # xfce4-desktop /backdrop/...` enumeration doesn't apply here. Eval uses
    # upstream's `vm_wallpaper` getter which is desktop-agnostic.
    'efcf0d81-0835-4880-b2fd-d866e8bc2294': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                '/home/user/Desktop/vlc_snapshot.png '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/vlc/efcf0d81-0835-4880-b2fd-d866e8bc2294/interstellar.png' "
                                                                                                '&& ls -la '
                                                                                                '/home/user/Desktop/vlc_snapshot.png',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'gsettings set '
                                                                                                'org.gnome.desktop.background '
                                                                                                'picture-uri '
                                                                                                "'file:///home/user/Desktop/vlc_snapshot.png' && "
                                                                                                'gsettings set '
                                                                                                'org.gnome.desktop.background '
                                                                                                'picture-uri-dark '
                                                                                                "'file:///home/user/Desktop/vlc_snapshot.png' && "
                                                                                                'gsettings set '
                                                                                                'org.gnome.desktop.background '
                                                                                                'picture-options '
                                                                                                "'zoom'",
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 3}}]},
    'f3977615-2b45-4ac5-8bba-80c17dbe2a37': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/vlc',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'timeout 3 vlc --intf '
                                                                                                'dummy --reset-config '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sed -i '
                                                                                                "'s/^#\\?one-instance-when-started-from-file=.*/one-instance-when-started-from-file=0/' "
                                                                                                '/home/user/.config/vlc/vlcrc',
                                                                                     'shell': True}}]},
    'fba2c100-79e8-42df-ae74-b592418d54f4': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/interstellar.png' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/vlc/fba2c100-79e8-42df-ae74-b592418d54f4/interstellar.png'",
                                                                                     'shell': True}}]}}
