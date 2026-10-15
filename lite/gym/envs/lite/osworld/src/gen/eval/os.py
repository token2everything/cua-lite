"""Hand-curated oracle metadata for os eval tasks.

OS-level / desktop-environment tasks.

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
- total entries:           19
- with actions:            19
- after_postconfig=True:   1
- block: exclude_reason:   2
- evaluator override:      1
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {   '13584542-872b-42d8-b299-866967b5c3ef': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -f '
                                                                                                'gnome-terminal '
                                                                                                '2>/dev/null; sleep 1; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'set -e\n'
                                                                                                'export DISPLAY=:1\n'
                                                                                                'if [ -f '
                                                                                                '/tmp/dbus-session-bus-address '
                                                                                                ']; then export '
                                                                                                'DBUS_SESSION_BUS_ADDRESS="$(cat '
                                                                                                '/tmp/dbus-session-bus-address)"; '
                                                                                                'fi\n'
                                                                                                'profile="$(gsettings '
                                                                                                'get '
                                                                                                'org.gnome.Terminal.ProfilesList '
                                                                                                'default | sed '
                                                                                                '"s/\'//g")"\n'
                                                                                                'schema="org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:${profile}/"\n'
                                                                                                'gsettings set '
                                                                                                '"$schema" '
                                                                                                'default-size-columns '
                                                                                                '132\n'
                                                                                                'gsettings set '
                                                                                                '"$schema" '
                                                                                                'default-size-rows '
                                                                                                '43\n',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '23393935-50c7-4a86-aeea-2b78fd089c5c': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop/cpjpg',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'find '
                                                                                                '/home/user/Desktop/photos '
                                                                                                "-iname '*.jpg' -exec "
                                                                                                'cp {} '
                                                                                                '/home/user/Desktop/cpjpg/ '
                                                                                                '\\;',
                                                                                     'shell': True}}]},
    '28cc3b7e-b194-4bc9-8353-d04c0f4d56d2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'export '
                                                                                                'XDG_RUNTIME_DIR=/run/user/1000; '
                                                                                                '(pulseaudio --check '
                                                                                                '2>/dev/null) || '
                                                                                                '(timeout 8 '
                                                                                                'pulseaudio --start '
                                                                                                '--daemonize=yes '
                                                                                                '--exit-idle-time=-1 '
                                                                                                '|| true)',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'export '
                                                                                                'XDG_RUNTIME_DIR=/run/user/1000; '
                                                                                                'pactl set-sink-volume '
                                                                                                '@DEFAULT_SINK@ 100%',
                                                                                     'shell': True}}],
                                                'config_prepend': [   {   'type': 'execute',
                                                                          'parameters': {   'command': 'export XDG_RUNTIME_DIR=/run/user/1000; (pulseaudio --check 2>/dev/null) || (timeout 8 pulseaudio --start --daemonize=yes --exit-idle-time=-1 || true); pactl set-sink-volume @DEFAULT_SINK@ 50%',
                                                                                            'shell': True}}]},
    '37887e8c-da15-4192-923c-08fa390a176d': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/tmp/test_files/old_files '
                                                                                                '/tmp/test_files/new_files',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'find /tmp/test_files '
                                                                                                '-maxdepth 1 -type f '
                                                                                                '-mtime +29 -exec mv '
                                                                                                '{} '
                                                                                                '/tmp/test_files/old_files/ '
                                                                                                '\\; 2>/dev/null || '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'find /tmp/test_files '
                                                                                                '-maxdepth 1 -type f '
                                                                                                '-exec mv {} '
                                                                                                '/tmp/test_files/new_files/ '
                                                                                                '\\; 2>/dev/null || '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd '
                                                                                                '/tmp/test_files/old_files '
                                                                                                '&& for f in *; do [ '
                                                                                                '-f "$f" ] && gzip '
                                                                                                '"$f"; done '
                                                                                                '2>/dev/null || true',
                                                                                     'shell': True}}]},
    '3ce045a0-877b-42aa-8d2c-b4a863336ab8': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo user | sudo -S '
                                                                                                'apt-get update 2>&1 | '
                                                                                                'tail -1; echo user | '
                                                                                                'sudo -S apt-get '
                                                                                                'install -y bc 2>&1 | '
                                                                                                'tail -1; gsettings '
                                                                                                'set '
                                                                                                'org.gnome.desktop.interface '
                                                                                                'text-scaling-factor '
                                                                                                '1.5; sleep 0.5',
                                                                                     'shell': True}}]},
    '4127319a-8b79-4410-b58a-7a151e15f3d7': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -f '
                                                                                                'gnome-terminal '
                                                                                                '2>/dev/null; sleep '
                                                                                                '0.5',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'bash',
                                                                                                    '-c',
                                                                                                    'gnome-terminal -- '
                                                                                                    "bash -ic 'echo "
                                                                                                    "54; exec bash'"]}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '4d117223-a354-47fb-8b45-62ab1390a95f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'find '
                                                                                                '/home/user/testDir '
                                                                                                '-type f -exec chmod '
                                                                                                '644 {} +',
                                                                                     'shell': True}}]},
    '5812b315-e7bd-4265-b51f-863c02174c28': {   'evaluator': {   'postconfig': [   {   'type': 'execute',
                                                                                       'parameters': {   'command': 'echo '
                                                                                                                    'user '
                                                                                                                    '| '
                                                                                                                    'sudo '
                                                                                                                    '-S '
                                                                                                                    'apt-get '
                                                                                                                    'install '
                                                                                                                    '-y '
                                                                                                                    '--no-install-recommends '
                                                                                                                    'expect '
                                                                                                                    '2>&1 '
                                                                                                                    '| '
                                                                                                                    'tail '
                                                                                                                    '-1',
                                                                                                         'shell': True}},
                                                                                   {   'type': 'download',
                                                                                       'parameters': {   'files': [   {   'url': 'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/os/5812b315-e7bd-4265-b51f-863c02174c28/check_password.sh',
                                                                                                                          'path': 'check_password.sh'}]}},
                                                                                   {   'type': 'execute',
                                                                                       'parameters': {   'command': 'chmod '
                                                                                                                    '+x '
                                                                                                                    'check_password.sh',
                                                                                                         'shell': True}}],
                                                                 'func': 'check_include_exclude',
                                                                 'result': {   'type': 'vm_command_line',
                                                                               'command': 'USERNAME="charles"; '
                                                                                          'HOMEDIR="/home/test1"; '
                                                                                          'PASSWORD="Ex@mpleP@55w0rd!"; '
                                                                                          './check_password.sh '
                                                                                          '"$USERNAME" "$PASSWORD" && '
                                                                                          '[ "$(getent passwd '
                                                                                          '"$USERNAME" | cut -d: -f6)" '
                                                                                          '= "$HOMEDIR" ] && [ $(stat '
                                                                                          '-c "%A" "$HOMEDIR" | cut -b '
                                                                                          '3) = "w" ] && echo '
                                                                                          '"Password, home directory, '
                                                                                          'and write permission check '
                                                                                          'passed" || echo "Check '
                                                                                          'failed"',
                                                                               'shell': True},
                                                                 'expected': {   'type': 'rule',
                                                                                 'rules': {   'include': [   'Password, '
                                                                                                             'home '
                                                                                                             'directory, '
                                                                                                             'and '
                                                                                                             'write '
                                                                                                             'permission '
                                                                                                             'check '
                                                                                                             'passed'],
                                                                                              'exclude': []}}},
                                                'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo user | sudo -S '
                                                                                                'apt-get install -y '
                                                                                                '--no-install-recommends '
                                                                                                'expect 2>&1 | tail -1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo user | sudo -S '
                                                                                                'bash -c "mkdir -p '
                                                                                                '/home/test1; userdel '
                                                                                                '-r charles '
                                                                                                '2>/dev/null; useradd '
                                                                                                '-d /home/test1 -s '
                                                                                                '/bin/bash charles && '
                                                                                                'echo '
                                                                                                'charles:Ex@mpleP@55w0rd! '
                                                                                                '| chpasswd && chown '
                                                                                                '-R charles:charles '
                                                                                                '/home/test1 && chmod '
                                                                                                '755 /home/test1"',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user && curl '
                                                                                                '-sL '
                                                                                                'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/os/5812b315-e7bd-4265-b51f-863c02174c28/check_password.sh '
                                                                                                '-o check_password.sh '
                                                                                                '&& chmod +x '
                                                                                                'check_password.sh',
                                                                                     'shell': True}}]},
    '5c1075ca-bb34-46a3-a7a0-029bd7463e79': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd '
                                                                                                '/home/user/test_environment '
                                                                                                '&& mkdir -p ./fails '
                                                                                                '&& find . -path '
                                                                                                "'./fails' -prune -o "
                                                                                                "-name '*failed.ipynb' "
                                                                                                '-print | while read '
                                                                                                'f; do '
                                                                                                'dest="./fails/$f"; '
                                                                                                'mkdir -p "$(dirname '
                                                                                                '"$dest")"; cp "$f" '
                                                                                                '"$dest"; done',
                                                                                     'shell': True}}]},
    '5ced85fc-fa1a-4217-95fd-0fb530545ce2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'printf '
                                                                                                "'1<br/>\\n2<br/>\\n3<br/>\\n' "
                                                                                                '> '
                                                                                                '/home/user/output.txt',
                                                                                     'shell': True}}]},
    '5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mv '
                                                                                                '/home/user/.local/share/Trash/files/poster_party_night.webp '
                                                                                                '/home/user/Desktop/ '
                                                                                                '2>/dev/null || find '
                                                                                                '/home/user/.local/share/Trash '
                                                                                                '-name '
                                                                                                "'poster_party_night.webp' "
                                                                                                '-exec mv {} '
                                                                                                '/home/user/Desktop/ '
                                                                                                '\\; 2>/dev/null || '
                                                                                                'true',
                                                                                     'shell': True}}],
                                                'config_append': [   {   'type': 'execute',
                                                                         'parameters': {   'command': 'echo user | sudo -S -v 2>/dev/null; '
                                                                                                      'sudo mkdir -p '
                                                                                                      '/home/user/.local/share/Trash/files '
                                                                                                      '&& sudo chown '
                                                                                                      'user:user '
                                                                                                      '/home/user/.local/share/Trash '
                                                                                                      '/home/user/.local/share/Trash/files '
                                                                                                      '&& mv '
                                                                                                      '/home/user/Desktop/poster_party_night.webp '
                                                                                                      '/home/user/.local/share/Trash/files/ '
                                                                                                      '2>/dev/null; '
                                                                                                      'true',
                                                                                           'shell': True}}]},
    '6f56bf42-85b8-4fbb-8e06-6c44960184ba': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user && for '
                                                                                                'd in dir1 dir2 dir3; '
                                                                                                'do cp file1 $d/ '
                                                                                                '2>/dev/null || true; '
                                                                                                'done',
                                                                                     'shell': True}}]},
    '94d95f96-9699-4208-98ba-3c3119edf9c2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "echo user | sudo -S bash -c 'echo "
                                                                                                '"#!/bin/bash" > '
                                                                                                '/usr/local/bin/spotify '
                                                                                                '&& chmod +x '
                                                                                                "/usr/local/bin/spotify'",
                                                                                     'shell': True}}]},
    'a4d98375-215b-4a4d-aee9-3d4370fccc41': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'gsettings set '
                                                                                                'org.gnome.desktop.screensaver '
                                                                                                'lock-enabled true',
                                                                                     'shell': True}}]},
    'b6781586-6346-41cd-935a-a6b1487918fc': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo user | sudo -S -v 2>/dev/null; '
                                                                                                'sudo ln -sf '
                                                                                                '/usr/share/zoneinfo/UTC '
                                                                                                '/etc/localtime && '
                                                                                                'echo UTC | sudo tee '
                                                                                                '/etc/timezone > '
                                                                                                '/dev/null',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo user | sudo -S -v 2>/dev/null; '
                                                                                                'sudo dpkg-reconfigure '
                                                                                                '-f noninteractive '
                                                                                                'tzdata 2>/dev/null; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo user | sudo -S -v 2>/dev/null; '
                                                                                                'printf '
                                                                                                "'#!/bin/bash\\nTZ=$(cat "
                                                                                                '/etc/timezone '
                                                                                                '2>/dev/null || echo '
                                                                                                'UTC)\\nDT=$(TZ=$TZ '
                                                                                                'date "+%%a '
                                                                                                '%%Y-%%m-%%d '
                                                                                                '%%H:%%M:%%S")\\nUDT=$(date '
                                                                                                '-u "+%%a %%Y-%%m-%%d '
                                                                                                '%%H:%%M:%%S")\\nOFF=$(TZ=$TZ '
                                                                                                'date '
                                                                                                '+%%z)\\nOFF=$(echo '
                                                                                                '$OFF | tr -d '
                                                                                                ':)\\necho '
                                                                                                '"               Local '
                                                                                                'time: $DT $TZ"\\necho '
                                                                                                '"           Universal '
                                                                                                'time: $UDT '
                                                                                                'UTC"\\necho '
                                                                                                '"                 RTC '
                                                                                                'time: $UDT"\\necho '
                                                                                                '"                Time '
                                                                                                'zone: $TZ ($TZ, '
                                                                                                '${OFF})"\\necho '
                                                                                                '"System clock '
                                                                                                'synchronized: '
                                                                                                'yes"\\necho '
                                                                                                '"              NTP '
                                                                                                'service: '
                                                                                                'inactive"\\necho '
                                                                                                '"          RTC in '
                                                                                                'local TZ: no"\\n\' | '
                                                                                                'sudo tee '
                                                                                                '/usr/local/bin/timedatectl '
                                                                                                '> /dev/null && sudo '
                                                                                                'chmod +x '
                                                                                                '/usr/local/bin/timedatectl',
                                                                                     'shell': True}}]},
    # bedcedc4: GNOME base ships with idle-dim=false by default, so the
    # OR evaluator (idle-delay==0 OR idle-dim==false) was already satisfied after
    # upstream config (idle-delay=300) → TRIVIAL_PASS. Prepend a step that forces
    # idle-dim=true so the initial state does NOT match gold, requiring the agent
    # to actually toggle the setting.
    'bedcedc4-4d72-425e-ad62-21960b11fe0d': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'export '
                                                                                                'DBUS_SESSION_BUS_ADDRESS=$(cat '
                                                                                                '/tmp/dbus-session-bus-address '
                                                                                                '2>/dev/null || echo '
                                                                                                "'unix:path=/run/user/1000/bus') "
                                                                                                '&& dconf write '
                                                                                                '/org/gnome/settings-daemon/plugins/power/idle-dim '
                                                                                                'false 2>/dev/null; '
                                                                                                'dconf write '
                                                                                                '/org/gnome/desktop/session/idle-delay '
                                                                                                "'uint32 0' "
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}}],
                                                'config_prepend': [   {   'type': 'execute',
                                                                          'parameters': {   'command': [   'gsettings',
                                                                                                           'set',
                                                                                                           'org.gnome.settings-daemon.plugins.power',
                                                                                                           'idle-dim',
                                                                                                           'true']}}]},
    'e0df059f-28a6-4169-924f-b9623e7184cc': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mv '
                                                                                                '/home/user/Desktop/todo_list_Jan_1 '
                                                                                                '/home/user/Desktop/todo_list_Jan_2',
                                                                                     'shell': True}}]},
    'ec4e3f68-9ea4-4c18-a5c9-69f89d1178b3': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'gsettings set '
                                                                                                'org.gnome.shell '
                                                                                                'favorite-apps '
                                                                                                '"[\'thunderbird.desktop\', '
                                                                                                '\'google-chrome.desktop\']"',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sleep 0.5',
                                                                                     'shell': True}}]},
    # f9be0997 (DnD): upstream's `gsettings get
    # org.gnome.desktop.notifications show-banners` is canonical on the GNOME
    # base (gnome-settings-daemon backs gsettings). Oracle: gsettings set
    # show-banners false (DnD on == banners off, matching upstream `expected:
    # "false\n"`).
    'f9be0997-4b7c-45c5-b05c-4612b44a6118': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'gsettings set '
                                                                                                'org.gnome.desktop.notifications '
                                                                                                'show-banners false',
                                                                                     'shell': True}}]}}
