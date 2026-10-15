"""Hand-curated oracle metadata for thunderbird eval tasks.

Thunderbird email-client tasks.

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
- total entries:           14
- with actions:            14
- after_postconfig=True:   2
- block: exclude_reason:   0
- evaluator override:      0
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {   '08c73485-7c6d-4681-999d-919f5c32dcfa': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'grep -q '
                                                                                                "'mail.server.default.applyIncomingFilters' "
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js '
                                                                                                '2>/dev/null || echo '
                                                                                                '\'user_pref("mail.server.default.applyIncomingFilters", '
                                                                                                "true);' >> "
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'grep -q '
                                                                                                "'mail.imap.use_status_for_biff' "
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js '
                                                                                                '2>/dev/null || echo '
                                                                                                '\'user_pref("mail.imap.use_status_for_biff", '
                                                                                                "false);' >> "
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '10a730d5-d414-4b40-b479-684bed1ae522': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f '
                                                                                                'thunderbird '
                                                                                                '2>/dev/null; sleep 3',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release '
                                                                                                '&& touch '
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sed -i '
                                                                                                "'/extensions.activeThemeID/d' "
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js '
                                                                                                '&& echo '
                                                                                                '\'user_pref("extensions.activeThemeID", '
                                                                                                '"thunderbird-compact-dark@mozilla.org");\' '
                                                                                                '>> '
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js',
                                                                                     'shell': True}}]},
    '15c3b339-88f7-4a86-ab16-e71c58dcb01e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'curl -sL '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/thunderbird/dd84e895-72fd-4023-a336-97689ded257c/thunderbird-profile.tar.gz' "
                                                                                                '-o '
                                                                                                '/home/user/tb-regular-profile.tar.gz '
                                                                                                '2>/dev/null && tar '
                                                                                                '-xzf '
                                                                                                '/home/user/tb-regular-profile.tar.gz '
                                                                                                '--recursive-unlink -C '
                                                                                                '/home/user/ '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'DISPLAY=:1 dbus-send '
                                                                                                '--session '
                                                                                                '--print-reply '
                                                                                                '--dest=org.a11y.Bus '
                                                                                                '/org/a11y/bus '
                                                                                                'org.freedesktop.DBus.Properties.Set '
                                                                                                'string:org.a11y.Status '
                                                                                                'string:IsEnabled '
                                                                                                'variant:boolean:true '
                                                                                                '>/dev/null 2>&1; true',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {'command': ['/usr/bin/thunderbird']}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'for i in $(seq 1 20); '
                                                                                                'do DISPLAY=:1 wmctrl '
                                                                                                '-l 2>/dev/null | grep '
                                                                                                '-qi '
                                                                                                "'thunderbird\\|inbox\\|outlook' "
                                                                                                '&& break; sleep 2; '
                                                                                                'done; sleep 5; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'WID=$(DISPLAY=:1 '
                                                                                                'xdotool search --name '
                                                                                                "'Thunderbird' "
                                                                                                '2>/dev/null | head '
                                                                                                '-1); DISPLAY=:1 '
                                                                                                'xdotool '
                                                                                                'windowactivate --sync '
                                                                                                '$WID 2>/dev/null; '
                                                                                                'sleep 1; DISPLAY=:1 '
                                                                                                'xdotool key '
                                                                                                'ctrl+shift+j '
                                                                                                '2>/dev/null; sleep 3; '
                                                                                                'DISPLAY=:1 xdotool '
                                                                                                'type --clearmodifiers '
                                                                                                '--delay 20 '
                                                                                                '\'document.getElementById("tabmail").openTab("contentTab", '
                                                                                                '{url: '
                                                                                                '"about:accountsetup"})\' '
                                                                                                '2>/dev/null; '
                                                                                                'DISPLAY=:1 xdotool '
                                                                                                'key Return '
                                                                                                '2>/dev/null; sleep 4; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'CWID=$(DISPLAY=:1 '
                                                                                                'xdotool search '
                                                                                                '--class Devtools '
                                                                                                '2>/dev/null | head '
                                                                                                '-1); [ -n "$CWID" ] '
                                                                                                '&& DISPLAY=:1 xdotool '
                                                                                                'windowclose $CWID '
                                                                                                '2>/dev/null; sleep 1; '
                                                                                                'WID=$(DISPLAY=:1 '
                                                                                                'xdotool search --name '
                                                                                                "'Thunderbird' "
                                                                                                '2>/dev/null | head '
                                                                                                '-1); DISPLAY=:1 '
                                                                                                'xdotool '
                                                                                                'windowactivate --sync '
                                                                                                '$WID 2>/dev/null; '
                                                                                                'sleep 0.5; DISPLAY=:1 '
                                                                                                'xdotool mousemove 235 '
                                                                                                '262 click 1 '
                                                                                                '2>/dev/null; sleep '
                                                                                                '0.5; DISPLAY=:1 '
                                                                                                'xdotool key Tab '
                                                                                                '2>/dev/null; sleep '
                                                                                                '0.3; DISPLAY=:1 '
                                                                                                'xdotool type '
                                                                                                '--clearmodifiers '
                                                                                                '--delay 30 '
                                                                                                "'anonym-x2024@outlook.com' "
                                                                                                '2>/dev/null; sleep '
                                                                                                '0.3; DISPLAY=:1 '
                                                                                                'xdotool key Tab '
                                                                                                '2>/dev/null; sleep '
                                                                                                '0.3; DISPLAY=:1 '
                                                                                                'xdotool type '
                                                                                                '--clearmodifiers '
                                                                                                "--delay 30 'password' "
                                                                                                '2>/dev/null; sleep 1; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '3f28fe4f-5d9d-4994-a456-efd78cfae1a3': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -f thunderbird; '
                                                                                                'sleep 5',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'path = '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js'\n"
                                                                                                'lines = []\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        lines = '
                                                                                                'f.readlines()\n'
                                                                                                '\n'
                                                                                                'lines = [l for l in '
                                                                                                'lines if '
                                                                                                "'mail.identity.id1.htmlSigText' "
                                                                                                'not in l\n'
                                                                                                '         and '
                                                                                                "'mail.identity.id1.sig_on_reply' "
                                                                                                'not in l\n'
                                                                                                '         and '
                                                                                                "'mail.identity.id1.sig_on_fwd' "
                                                                                                'not in l\n'
                                                                                                '         and '
                                                                                                "'mail.identity.id1.htmlSigFormat' "
                                                                                                'not in l\n'
                                                                                                '         and '
                                                                                                "'mail.identity.id1.attachSignature' "
                                                                                                'not in l]\n'
                                                                                                '\n'
                                                                                                'sig = "Anonym" + '
                                                                                                'chr(10) + "XYZ Lab"\n'
                                                                                                'lines.append(\'user_pref("mail.identity.id1.htmlSigText", '
                                                                                                '"\' + '
                                                                                                'sig.replace(chr(10), '
                                                                                                "'\\\\n') + "
                                                                                                '\'");\\n\')\n'
                                                                                                'lines.append(\'user_pref("mail.identity.id1.htmlSigFormat", '
                                                                                                "false);\\n')\n"
                                                                                                'lines.append(\'user_pref("mail.identity.id1.attachSignature", '
                                                                                                "false);\\n')\n"
                                                                                                'lines.append(\'user_pref("mail.identity.id1.sig_on_reply", '
                                                                                                "true);\\n')\n"
                                                                                                'lines.append(\'user_pref("mail.identity.id1.sig_on_fwd", '
                                                                                                "true);\\n')\n"
                                                                                                '\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    '
                                                                                                'f.writelines(lines)\n'
                                                                                                "print('Signature "
                                                                                                "set')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '3f49d2cc-f400-4e7d-90cc-9b18e401cc31': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -f thunderbird '
                                                                                                '2>/dev/null; sleep 2',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os, '
                                                                                                'glob\n'
                                                                                                '\n'
                                                                                                '# Find the '
                                                                                                'thunderbird profile '
                                                                                                'directory\n'
                                                                                                'profile_dirs = '
                                                                                                "glob.glob('/home/user/.thunderbird/*.default-release')\n"
                                                                                                'if not profile_dirs:\n'
                                                                                                '    profile_dirs = '
                                                                                                "glob.glob('/home/user/.thunderbird/*.default')\n"
                                                                                                'if not profile_dirs:\n'
                                                                                                '    print("No '
                                                                                                'thunderbird profile '
                                                                                                'found!")\n'
                                                                                                '    exit(1)\n'
                                                                                                '\n'
                                                                                                'profile_dir = '
                                                                                                'profile_dirs[0]\n'
                                                                                                'path = '
                                                                                                'os.path.join(profile_dir, '
                                                                                                "'xulstore.json')\n"
                                                                                                '\n'
                                                                                                'data = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        data = '
                                                                                                'json.load(f)\n'
                                                                                                '\n'
                                                                                                '# Set smart/unified '
                                                                                                'folders mode\n'
                                                                                                'messenger_key = '
                                                                                                '"chrome://messenger/content/messenger.xhtml"\n'
                                                                                                'if messenger_key not '
                                                                                                'in data:\n'
                                                                                                '    '
                                                                                                'data[messenger_key] = '
                                                                                                '{}\n'
                                                                                                '\n'
                                                                                                'if "folderTree" not '
                                                                                                'in '
                                                                                                'data[messenger_key]:\n'
                                                                                                '    '
                                                                                                'data[messenger_key]["folderTree"] '
                                                                                                '= {}\n'
                                                                                                '\n'
                                                                                                'data[messenger_key]["folderTree"]["mode"] '
                                                                                                '= "smart"\n'
                                                                                                '\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(data, '
                                                                                                'f, indent=2)\n'
                                                                                                '\n'
                                                                                                "print(f'Set smart "
                                                                                                'folder mode in '
                                                                                                "{path}')\n"
                                                                                                "print(f'Content: "
                                                                                                '{json.dumps(data[messenger_key]["folderTree"])}\')\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '5203d847-2572-4150-912a-03f062254390': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/ImapMail/outlook.office365.com'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cat > '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/ImapMail/outlook.office365.com/msgFilterRules.dat' "
                                                                                                "<< 'FILTEREOF'\n"
                                                                                                'version="9"\n'
                                                                                                'logging="no"\n'
                                                                                                'name="AutoFilter"\n'
                                                                                                'enabled="yes"\n'
                                                                                                'type="17"\n'
                                                                                                'action="Move to '
                                                                                                'folder"\n'
                                                                                                'actionValue="mailbox://nobody@Local%20Folders/Promotions"\n'
                                                                                                'condition="AND '
                                                                                                '(subject,contains,discount)"\n'
                                                                                                'FILTEREOF',
                                                                                     'shell': True}}]},
    '7b1e1ff9-bb85-49be-b01d-d6424be18cd0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'dbus-send --session '
                                                                                                '--print-reply '
                                                                                                '--dest=org.a11y.Bus '
                                                                                                '/org/a11y/bus '
                                                                                                'org.freedesktop.DBus.Properties.Set '
                                                                                                'string:org.a11y.Status '
                                                                                                'string:IsEnabled '
                                                                                                'variant:boolean:true '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {'command': ['/usr/bin/thunderbird']}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'for i in $(seq 1 20); '
                                                                                                'do wmctrl -l '
                                                                                                '2>/dev/null | grep '
                                                                                                '-qi '
                                                                                                "'thunderbird\\|inbox' "
                                                                                                '&& break; sleep 2; '
                                                                                                'done; sleep 5; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'WID=$(xdotool search '
                                                                                                "--name 'Thunderbird' "
                                                                                                '| head -1); xdotool '
                                                                                                'windowactivate --sync '
                                                                                                '$WID; sleep 1; '
                                                                                                'xdotool key '
                                                                                                'ctrl+shift+j; sleep '
                                                                                                '3; xdotool type '
                                                                                                '--delay 20 '
                                                                                                '\'document.getElementById("tabmail").openTab("contentTab", '
                                                                                                '{url: '
                                                                                                '"about:profiles"})\'; '
                                                                                                'xdotool key Return; '
                                                                                                'sleep 3; true',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '9b7bc335-06b5-4cd3-9119-1a649c478509': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/ImapMail/outlook.office365.com'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cat > '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/ImapMail/outlook.office365.com/msgFilterRules.dat' "
                                                                                                "<< 'FILTEREOF'\n"
                                                                                                'version="9"\n'
                                                                                                'logging="no"\n'
                                                                                                'name="AutoFilter"\n'
                                                                                                'enabled="yes"\n'
                                                                                                'type="17"\n'
                                                                                                'action="Forward"\n'
                                                                                                'actionValue="anonym-x2024@gmail.com"\n'
                                                                                                'condition="ALL"\n'
                                                                                                'FILTEREOF',
                                                                                     'shell': True}}]},
    '9bc3cc16-074a-45ac-9bdc-b2a362e1daf3': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/emails.bak',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import os\n'
                                                                                                'import re\n'
                                                                                                'import email\n'
                                                                                                '\n'
                                                                                                'outdir = '
                                                                                                "'/home/user/emails.bak'\n"
                                                                                                'profile_dir = '
                                                                                                "'/home/user/.thunderbird'\n"
                                                                                                '\n'
                                                                                                '# Find ALL potential '
                                                                                                'inbox files (both '
                                                                                                'Inbox and INBOX)\n'
                                                                                                'inbox_candidates = '
                                                                                                '[]\n'
                                                                                                'for root, dirs, files '
                                                                                                'in '
                                                                                                'os.walk(profile_dir):\n'
                                                                                                '    for f in files:\n'
                                                                                                '        fname_lower = '
                                                                                                'f.lower()\n'
                                                                                                '        if '
                                                                                                'fname_lower == '
                                                                                                "'inbox' and not "
                                                                                                "f.endswith('.msf'):\n"
                                                                                                '            '
                                                                                                'inbox_candidates.append(os.path.join(root, '
                                                                                                'f))\n'
                                                                                                '\n'
                                                                                                'print(f"Found inbox '
                                                                                                'candidates: '
                                                                                                '{inbox_candidates}")\n'
                                                                                                '\n'
                                                                                                'count = 0\n'
                                                                                                'for inbox_path in '
                                                                                                'inbox_candidates:\n'
                                                                                                '    if not '
                                                                                                'os.path.exists(inbox_path):\n'
                                                                                                '        continue\n'
                                                                                                '    '
                                                                                                'print(f"Processing: '
                                                                                                '{inbox_path}")\n'
                                                                                                '    try:\n'
                                                                                                '        with '
                                                                                                "open(inbox_path, 'r', "
                                                                                                "errors='replace') as "
                                                                                                'f:\n'
                                                                                                '            content = '
                                                                                                'f.read()\n'
                                                                                                '\n'
                                                                                                '        # Split by '
                                                                                                '"From - " or "From " '
                                                                                                'at line start (mbox '
                                                                                                'format)\n'
                                                                                                '        # Thunderbird '
                                                                                                'uses "From - " as '
                                                                                                'separator for IMAP '
                                                                                                'offline files\n'
                                                                                                '        messages = '
                                                                                                "re.split(r'^From "
                                                                                                "[-\\w]', content, "
                                                                                                'flags=re.MULTILINE)\n'
                                                                                                '\n'
                                                                                                '        for i, '
                                                                                                'msg_text in '
                                                                                                'enumerate(messages):\n'
                                                                                                '            if not '
                                                                                                'msg_text.strip():\n'
                                                                                                '                '
                                                                                                'continue\n'
                                                                                                '\n'
                                                                                                '            # '
                                                                                                'Reconstruct the "From '
                                                                                                '" line\n'
                                                                                                '            if i > '
                                                                                                '0:\n'
                                                                                                '                '
                                                                                                'msg_text = "From " + '
                                                                                                'msg_text\n'
                                                                                                '\n'
                                                                                                '            try:\n'
                                                                                                '                msg = '
                                                                                                'email.message_from_string(msg_text)\n'
                                                                                                '                '
                                                                                                'subject = '
                                                                                                "msg.get('Subject', "
                                                                                                "f'message_{count}')\n"
                                                                                                '                if '
                                                                                                'not subject:\n'
                                                                                                '                    '
                                                                                                'subject = '
                                                                                                "f'message_{count}'\n"
                                                                                                '                # '
                                                                                                'Decode subject if '
                                                                                                'encoded\n'
                                                                                                '                from '
                                                                                                'email.header import '
                                                                                                'decode_header\n'
                                                                                                '                '
                                                                                                'decoded_parts = '
                                                                                                'decode_header(subject)\n'
                                                                                                '                '
                                                                                                "decoded_subject = ''\n"
                                                                                                '                for '
                                                                                                'part, charset in '
                                                                                                'decoded_parts:\n'
                                                                                                '                    '
                                                                                                'if isinstance(part, '
                                                                                                'bytes):\n'
                                                                                                '                        '
                                                                                                'decoded_subject += '
                                                                                                'part.decode(charset '
                                                                                                "or 'utf-8', "
                                                                                                "errors='replace')\n"
                                                                                                '                    '
                                                                                                'else:\n'
                                                                                                '                        '
                                                                                                'decoded_subject += '
                                                                                                'part\n'
                                                                                                '                '
                                                                                                'subject = '
                                                                                                'decoded_subject\n'
                                                                                                '\n'
                                                                                                '                # '
                                                                                                'Clean subject for '
                                                                                                'filename\n'
                                                                                                '                '
                                                                                                'subject = '
                                                                                                're.sub(r\'[/\\\\:*?"<>|]\', '
                                                                                                "'_', subject)\n"
                                                                                                '                '
                                                                                                'subject = '
                                                                                                'subject.strip()[:100]\n'
                                                                                                '                if '
                                                                                                'not subject:\n'
                                                                                                '                    '
                                                                                                'subject = '
                                                                                                "f'message_{count}'\n"
                                                                                                '\n'
                                                                                                '                '
                                                                                                'eml_path = '
                                                                                                'os.path.join(outdir, '
                                                                                                "f'{subject}.eml')\n"
                                                                                                '                with '
                                                                                                "open(eml_path, 'w', "
                                                                                                "errors='replace') as "
                                                                                                'f:\n'
                                                                                                '                    '
                                                                                                'f.write(msg.as_string())\n'
                                                                                                '                count '
                                                                                                '+= 1\n'
                                                                                                '                '
                                                                                                "print(f'Saved: "
                                                                                                "{subject}.eml')\n"
                                                                                                '            except '
                                                                                                'Exception as e:\n'
                                                                                                '                '
                                                                                                "print(f'Error parsing "
                                                                                                "message {i}: {e}')\n"
                                                                                                '    except Exception '
                                                                                                'as e:\n'
                                                                                                "        print(f'Error "
                                                                                                'processing '
                                                                                                "{inbox_path}: {e}')\n"
                                                                                                '\n'
                                                                                                "print(f'Total: "
                                                                                                '{count} emails backed '
                                                                                                "up')\n"
                                                                                                '\n'
                                                                                                '# List what we saved\n'
                                                                                                'for f in '
                                                                                                'sorted(os.listdir(outdir)):\n'
                                                                                                "    print(f'  {f}')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'a10b69e1-6034-4a2b-93e1-571d45194f75': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -f thunderbird '
                                                                                                '2>/dev/null; sleep 2',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/Mail/Local "
                                                                                                "Folders'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'touch '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/Mail/Local "
                                                                                                "Folders/COMPANY' && "
                                                                                                'touch '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/Mail/Local "
                                                                                                "Folders/COMPANY.msf' "
                                                                                                '&& touch '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/Mail/Local "
                                                                                                "Folders/UNIVERSITY' "
                                                                                                '&& touch '
                                                                                                "'/home/user/.thunderbird/t5q2a5hp.default-release/Mail/Local "
                                                                                                "Folders/UNIVERSITY.msf'",
                                                                                     'shell': True}}]},
    'd38192b0-17dc-4e1d-99c3-786d0117de77': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'DISPLAY=:1 dbus-send '
                                                                                                '--session '
                                                                                                '--print-reply '
                                                                                                '--dest=org.a11y.Bus '
                                                                                                '/org/a11y/bus '
                                                                                                'org.freedesktop.DBus.Properties.Set '
                                                                                                'string:org.a11y.Status '
                                                                                                'string:IsEnabled '
                                                                                                'variant:boolean:true '
                                                                                                '>/dev/null 2>&1; true',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   '/usr/bin/thunderbird',
                                                                                                    '-compose',
                                                                                                    "from='Anonym "
                                                                                                    'Tester '
                                                                                                    "<anonym-x2024@outlook.com>',to=assistant@outlook.com,subject='New-month "
                                                                                                    'AWS '
                                                                                                    "Bill',body='test',attachment='file:///home/user/aws-bill.pdf'"]}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'dd84e895-72fd-4023-a336-97689ded257c': {   'actions': [   {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 thunderbird; '
                                                                                                'sleep 2',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': [   'python3',
                                                                                                    '-c',
                                                                                                    'import sqlite3\n'
                                                                                                    "db='/home/user/.thunderbird/t5q2a5hp.default-release/global-messages-db.sqlite'\n"
                                                                                                    'conn=sqlite3.connect(db)\n'
                                                                                                    'c=conn.cursor()\n'
                                                                                                    "c.execute('SELECT "
                                                                                                    'id, '
                                                                                                    'conversationID '
                                                                                                    'FROM messages '
                                                                                                    'WHERE '
                                                                                                    "folderID=13')\n"
                                                                                                    'rows=c.fetchall()\n'
                                                                                                    "print(f'Messages "
                                                                                                    'in Bills: '
                                                                                                    "{len(rows)}')\n"
                                                                                                    'for mid, cid in '
                                                                                                    'rows:\n'
                                                                                                    '    '
                                                                                                    "c.execute('DELETE "
                                                                                                    'FROM '
                                                                                                    'messageAttributes '
                                                                                                    'WHERE messageID=? '
                                                                                                    'AND '
                                                                                                    "attributeID=58',(mid,))\n"
                                                                                                    '    '
                                                                                                    "c.execute('INSERT "
                                                                                                    'INTO '
                                                                                                    'messageAttributes(conversationID,messageID,attributeID,value) '
                                                                                                    "VALUES(?,?,58,1)',(cid,mid))\n"
                                                                                                    'conn.commit()\n'
                                                                                                    'conn.close()\n'
                                                                                                    "print('Done')"]}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'dfac9ee8-9bc4-4cdc-b465-4a4bfcd2f397': {   'actions': [   {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 thunderbird; '
                                                                                                'sleep 2',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': [   'python3',
                                                                                                    '-c',
                                                                                                    'import json, os, '
                                                                                                    're\n'
                                                                                                    "profile='/home/user/.thunderbird/t5q2a5hp.default-release'\n"
                                                                                                    '# Clean '
                                                                                                    'logins.json\n'
                                                                                                    "lp=os.path.join(profile,'logins.json')\n"
                                                                                                    'if '
                                                                                                    'os.path.exists(lp):\n'
                                                                                                    '    '
                                                                                                    'd=json.load(open(lp))\n'
                                                                                                    '    '
                                                                                                    "orig=len(d.get('logins',[]))\n"
                                                                                                    '    '
                                                                                                    "d['logins']=[l "
                                                                                                    'for l in '
                                                                                                    "d.get('logins',[]) "
                                                                                                    "if 'outlook' not "
                                                                                                    'in '
                                                                                                    "l.get('hostname','').lower() "
                                                                                                    'and '
                                                                                                    "'anonym-x2024' "
                                                                                                    'not in '
                                                                                                    "l.get('encryptedUsername','').lower()]\n"
                                                                                                    '    '
                                                                                                    "json.dump(d,open(lp,'w'))\n"
                                                                                                    '    '
                                                                                                    "print(f'Removed "
                                                                                                    '{orig-len(d["logins"])} '
                                                                                                    "login(s)')\n"
                                                                                                    'else:\n'
                                                                                                    "    print('No "
                                                                                                    "logins.json')\n"
                                                                                                    '# Clean prefs.js\n'
                                                                                                    "pp=os.path.join(profile,'prefs.js')\n"
                                                                                                    'lines=open(pp).readlines()\n'
                                                                                                    'srv=None\n'
                                                                                                    'for l in lines:\n'
                                                                                                    '    '
                                                                                                    'm=re.match(r\'user_pref\\("mail\\.server\\.(server\\d+)\\.hostname",\\s*"([^"]+)"\\);\',l)\n'
                                                                                                    '    if m and '
                                                                                                    "'outlook' in "
                                                                                                    'm.group(2).lower(): '
                                                                                                    'srv=m.group(1)\n'
                                                                                                    '    '
                                                                                                    'm=re.match(r\'user_pref\\("mail\\.server\\.(server\\d+)\\.userName",\\s*"([^"]+)"\\);\',l)\n'
                                                                                                    '    if m and '
                                                                                                    "'anonym-x2024' in "
                                                                                                    'm.group(2).lower(): '
                                                                                                    'srv=m.group(1)\n'
                                                                                                    'acct=ident=None\n'
                                                                                                    'if srv:\n'
                                                                                                    '    for l in '
                                                                                                    'lines:\n'
                                                                                                    '        '
                                                                                                    'm=re.match(r\'user_pref\\("mail\\.account\\.(account\\d+)\\.server",\\s*"\'+srv+r\'"\\);\',l)\n'
                                                                                                    '        if m: '
                                                                                                    'acct=m.group(1)\n'
                                                                                                    '    for l in '
                                                                                                    'lines:\n'
                                                                                                    '        if acct:\n'
                                                                                                    '            '
                                                                                                    'm=re.match(r\'user_pref\\("mail\\.account\\.\'+acct+r\'\\.identities",\\s*"([^"]+)"\\);\',l)\n'
                                                                                                    '            if m: '
                                                                                                    'ident=m.group(1)\n'
                                                                                                    "print(f'srv={srv} "
                                                                                                    'acct={acct} '
                                                                                                    "ident={ident}')\n"
                                                                                                    'new=[]\n'
                                                                                                    'for l in lines:\n'
                                                                                                    '    skip=False\n'
                                                                                                    '    if srv and '
                                                                                                    "f'mail.server.{srv}.' "
                                                                                                    'in l: skip=True\n'
                                                                                                    '    if acct and '
                                                                                                    "f'mail.account.{acct}.' "
                                                                                                    'in l: skip=True\n'
                                                                                                    '    if ident and '
                                                                                                    "f'mail.identity.{ident}.' "
                                                                                                    'in l: skip=True\n'
                                                                                                    '    if not skip: '
                                                                                                    'new.append(l)\n'
                                                                                                    'final=[]\n'
                                                                                                    'for l in new:\n'
                                                                                                    '    if acct and '
                                                                                                    "'mail.accountmanager.accounts' "
                                                                                                    'in l:\n'
                                                                                                    '        '
                                                                                                    'm=re.match(r\'user_pref\\("mail\\.accountmanager\\.accounts",\\s*"([^"]+)"\\);\',l)\n'
                                                                                                    '        if m:\n'
                                                                                                    '            '
                                                                                                    'accts=[a.strip() '
                                                                                                    'for a in '
                                                                                                    "m.group(1).split(',') "
                                                                                                    'if '
                                                                                                    'a.strip()!=acct]\n'
                                                                                                    '            '
                                                                                                    'l=f\'user_pref("mail.accountmanager.accounts", '
                                                                                                    '"{",".join(accts)}");\\n\'\n'
                                                                                                    '    '
                                                                                                    'final.append(l)\n'
                                                                                                    "open(pp,'w').writelines(final)\n"
                                                                                                    "print(f'Removed "
                                                                                                    '{len(lines)-len(final)} '
                                                                                                    "lines')"]}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'f201fbc3-44e6-46fc-bcaa-432f9815454c': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'grep -q '
                                                                                                "'mail.identity.id1.auto_quote' "
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js '
                                                                                                '2>/dev/null || echo '
                                                                                                '\'user_pref("mail.identity.id1.auto_quote", '
                                                                                                "false);' >> "
                                                                                                '/home/user/.thunderbird/t5q2a5hp.default-release/prefs.js',
                                                                                     'shell': True}}],
                                                'after_postconfig': True}}
