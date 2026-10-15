"""Hand-curated oracle metadata for vs_code eval tasks.

VS Code editor tasks.

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
- total entries:           18
- with actions:            18
- after_postconfig=True:   0
- block: exclude_reason:   0
- evaluator override:      0
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {   '0512bb38-d531-4acf-9e7e-0add90816068': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'code '
                                                                                                '--no-sandbox '
                                                                                                '--install-extension '
                                                                                                '/home/user/test.vsix '
                                                                                                '--force '
                                                                                                '2>/dev/null '
                                                                                                '|| true',
                                                                                     'shell': True}}]},
    '0ed39f63-6049-43d4-ba4d-5fa2fe04a951': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/vscode_replace_text.txt' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/vs_code/0ed39f63-6049-43d4-ba4d-5fa2fe04a951/vscode_replace_text_gold.txt'",
                                                                                     'shell': True}}]},
    '276cc624-87ea-4f08-ab93-f770e3790175': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/settings.json'\n"
                                                                                                'existing = '
                                                                                                '{}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.update({'editor.wordWrapColumn': "
                                                                                                '50})\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '4e60007a-f5be-4bfc-9723-c39affa0a6d3': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'code '
                                                                                                '--no-sandbox '
                                                                                                '--install-extension '
                                                                                                'njpwerner.autodocstring '
                                                                                                '--force '
                                                                                                '2>/dev/null '
                                                                                                '|| true',
                                                                                     'shell': True}}]},
    '53ad5833-3455-407b-bbc6-45b4c79ab8fb': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User '
                                                                                                '&& python3 '
                                                                                                '-c "import '
                                                                                                'json, '
                                                                                                "os;p='/home/user/.config/Code/User/settings.json';d=json.load(open(p)) "
                                                                                                'if '
                                                                                                'os.path.exists(p) '
                                                                                                'else '
                                                                                                '{};d[\'security.workspace.trust.enabled\']=False;json.dump(d,open(p,\'w\'))"',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -f '
                                                                                                "'code "
                                                                                                "--no-sandbox' "
                                                                                                '2>/dev/null; '
                                                                                                'sleep 3',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'code',
                                                                                                    '--no-sandbox',
                                                                                                    '/home/user/project'],
                                                                                     'shell': False}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'export '
                                                                                                'DISPLAY=:1 '
                                                                                                '&& python3 '
                                                                                                '-c "import '
                                                                                                'pyautogui, '
                                                                                                'time; '
                                                                                                'time.sleep(0.5); '
                                                                                                "pyautogui.press('escape'); "
                                                                                                'time.sleep(1)"',
                                                                                     'shell': True}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}}]},
    '57242fad-77ca-454f-b71b-f187181a9f23': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'touch '
                                                                                                '/home/user/Desktop/test.py',
                                                                                     'shell': True}}]},
    '5e2d93d8-8ad0-4435-b150-1692aacaa994': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/project',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo '
                                                                                                '\'{"folders": '
                                                                                                '[{"path": '
                                                                                                '"."}], '
                                                                                                '"settings": '
                                                                                                "{}}' > "
                                                                                                '/home/user/project/project.code-workspace',
                                                                                     'shell': True}}]},
    '6ed0a554-cbee-4b44-84ea-fd6c042f4fe1': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json\n'
                                                                                                'path = '
                                                                                                "'/home/user/project.code-workspace'\n"
                                                                                                'with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '    data = '
                                                                                                'json.load(f)\n'
                                                                                                "data['folders'] "
                                                                                                "= [{'path': "
                                                                                                "'project'}, "
                                                                                                "{'path': "
                                                                                                "'data1'}, "
                                                                                                "{'path': "
                                                                                                "'data2'}]\n"
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(data, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '70745df8-f2f5-42bd-8074-fbc10334fcc5': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/settings.json'\n"
                                                                                                'existing = '
                                                                                                '{}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.update({'files.autoSave': "
                                                                                                "'afterDelay', "
                                                                                                "'files.autoSaveDelay': "
                                                                                                '500})\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '930fdb3b-11a8-46fe-9bac-577332e2640e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/keybindings.json'\n"
                                                                                                'existing = '
                                                                                                '[]\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.append({'key': "
                                                                                                "'ctrl+j', "
                                                                                                "'command': "
                                                                                                "'workbench.action.focusActiveEditorGroup', "
                                                                                                "'when': "
                                                                                                "'terminalFocus'})\n"
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '9439a27b-18ae-42d8-9778-5f68f891805e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/settings.json'\n"
                                                                                                'existing = '
                                                                                                '{}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.update({'debug.focusEditorOnBreak': "
                                                                                                'False})\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '982d12a5-beab-424f-8d38-d2a48429e511': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/settings.json'\n"
                                                                                                'existing = '
                                                                                                '{}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.update({'workbench.colorTheme': "
                                                                                                "'Visual "
                                                                                                'Studio '
                                                                                                "Dark'})\n"
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '9d425400-e9b2-4424-9a4b-d4c7abac4140': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/settings.json'\n"
                                                                                                'existing = '
                                                                                                '{}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.update({'workbench.editor.wrapTabs': "
                                                                                                'True})\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'c6bf789c-ba3a-4209-971d-b63abf0ab733': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/settings.json'\n"
                                                                                                'existing = '
                                                                                                '{}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.update({'files.exclude': "
                                                                                                "{'**/__pycache__': "
                                                                                                'True}})\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'e2b5e914-ffe1-44d2-8e92-58f8c5d92bb2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/settings.json'\n"
                                                                                                'existing = '
                                                                                                '{}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.update({'python.analysis.diagnosticSeverityOverrides': "
                                                                                                "{'reportMissingImports': "
                                                                                                "'none'}})\n"
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'ea98c5d7-3cf9-4f9b-8ad3-366b58e0fcae': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/Code/User',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/Code/User/keybindings.json'\n"
                                                                                                'existing = '
                                                                                                '[]\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'existing = '
                                                                                                'json.load(f)\n'
                                                                                                "existing.append({'key': "
                                                                                                "'ctrl+f', "
                                                                                                "'command': "
                                                                                                "'-list.find', "
                                                                                                "'when': "
                                                                                                "'listFocus "
                                                                                                '&& '
                                                                                                "listSupportsFind'})\n"
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(existing, '
                                                                                                'f, '
                                                                                                'indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'eabc805a-bfcf-4460-b250-ac92135819f6': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'code '
                                                                                                '--no-sandbox '
                                                                                                '--install-extension '
                                                                                                'ms-python.python '
                                                                                                '--force '
                                                                                                '2>/dev/null '
                                                                                                '|| true',
                                                                                     'shell': True}}]},
    'ec71221e-ac43-46f9-89b8-ee7d80f7e1c5': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/test.py' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/vs_code/ec71221e-ac43-46f9-89b8-ee7d80f7e1c5/test_gold.py'",
                                                                                     'shell': True}}]}}
