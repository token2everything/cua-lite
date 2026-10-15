"""Hand-curated oracle metadata for libreoffice_impress eval tasks.

LibreOffice Impress presentation tasks.

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
- total entries:           47
- with actions:            47
- after_postconfig=True:   1
- block: exclude_reason:   0
- evaluator override:      0
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {   '04578141-1d42-4146-b9cf-6fab4ce5fd74': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/45_2.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/04578141-1d42-4146-b9cf-6fab4ce5fd74/45_2_Gold.pptx'",
                                                                                     'shell': True}}]},
    '05dd4c1d-c489-4c85-8389-a7836c4f0567': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/38_1.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/05dd4c1d-c489-4c85-8389-a7836c4f0567/38_1_Gold.pptx'",
                                                                                     'shell': True}}]},
    '08aced46-45a2-48d7-993b-ed3fb5b32302': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/22_6.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/08aced46-45a2-48d7-993b-ed3fb5b32302/22_6_Gold.pptx'",
                                                                                     'shell': True}}]},
    '0a211154-fda0-48d0-9274-eaac4ce5486d': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/13_0.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/0a211154-fda0-48d0-9274-eaac4ce5486d/13_0_Gold.pptx'",
                                                                                     'shell': True}}]},
    '0f84bef9-9790-432e-92b7-eece357603fb': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1; '
                                                                                                'mkdir -p '
                                                                                                '/home/user/.config/libreoffice/4/user',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/libreoffice/4/user/registrymodifications.xcu'\n"
                                                                                                '\n'
                                                                                                '# Read '
                                                                                                'existing '
                                                                                                'file or '
                                                                                                'create new\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path, '
                                                                                                "'r') as f:\n"
                                                                                                '        '
                                                                                                'content = '
                                                                                                'f.read()\n'
                                                                                                'else:\n'
                                                                                                '    content '
                                                                                                "= ''\n"
                                                                                                '\n'
                                                                                                '# Check if '
                                                                                                'it already '
                                                                                                'has the '
                                                                                                'presenter '
                                                                                                'screen '
                                                                                                'setting\n'
                                                                                                'if '
                                                                                                "'EnablePresenterScreen' "
                                                                                                'in '
                                                                                                'content:\n'
                                                                                                '    # '
                                                                                                'Replace '
                                                                                                'existing '
                                                                                                'value\n'
                                                                                                '    import '
                                                                                                're\n'
                                                                                                '    content '
                                                                                                '= re.sub(\n'
                                                                                                '        '
                                                                                                "r'(<item "
                                                                                                'oor:path="/org\\.openoffice\\.Office\\.Impress/Misc/Start">.*?<prop '
                                                                                                'oor:name="EnablePresenterScreen".*?<value>).*?(</value>)\',\n'
                                                                                                '        '
                                                                                                "r'\\1false\\2',\n"
                                                                                                '        '
                                                                                                'content,\n'
                                                                                                '        '
                                                                                                'flags=re.DOTALL\n'
                                                                                                '    )\n'
                                                                                                'else:\n'
                                                                                                '    # Add '
                                                                                                'it before '
                                                                                                'closing '
                                                                                                'tag\n'
                                                                                                '    '
                                                                                                'new_item = '
                                                                                                "'<item "
                                                                                                'oor:path="/org.openoffice.Office.Impress/Misc/Start"><prop '
                                                                                                'oor:name="EnablePresenterScreen" '
                                                                                                'oor:op="fuse"><value>false</value></prop></item>\'\n'
                                                                                                '    if '
                                                                                                "'</oor:items>' "
                                                                                                'in '
                                                                                                'content:\n'
                                                                                                '        '
                                                                                                'content = '
                                                                                                "content.replace('</oor:items>', "
                                                                                                'new_item + '
                                                                                                "'\\n</oor:items>')\n"
                                                                                                '    elif '
                                                                                                'not '
                                                                                                'content.strip():\n'
                                                                                                '        '
                                                                                                'content = '
                                                                                                "'<?xml "
                                                                                                'version="1.0" '
                                                                                                'encoding="UTF-8"?>\\n<oor:items '
                                                                                                'xmlns:oor="http://openoffice.org/2001/registry" '
                                                                                                'xmlns:xs="http://www.w3.org/2001/XMLSchema" '
                                                                                                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\\n\' '
                                                                                                '+ new_item '
                                                                                                '+ '
                                                                                                "'\\n</oor:items>\\n'\n"
                                                                                                '    else:\n'
                                                                                                '        # '
                                                                                                'Append if '
                                                                                                'no proper '
                                                                                                'structure\n'
                                                                                                '        '
                                                                                                'content = '
                                                                                                "'<?xml "
                                                                                                'version="1.0" '
                                                                                                'encoding="UTF-8"?>\\n<oor:items '
                                                                                                'xmlns:oor="http://openoffice.org/2001/registry" '
                                                                                                'xmlns:xs="http://www.w3.org/2001/XMLSchema" '
                                                                                                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\\n\' '
                                                                                                '+ new_item '
                                                                                                '+ '
                                                                                                "'\\n</oor:items>\\n'\n"
                                                                                                '\n'
                                                                                                'os.makedirs(os.path.dirname(path), '
                                                                                                'exist_ok=True)\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'f.write(content)\n'
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '15aece23-a215-4579-91b4-69eec72e18da': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from pptx '
                                                                                                'import '
                                                                                                'Presentation\n'
                                                                                                'from '
                                                                                                'pptx.enum.text '
                                                                                                'import '
                                                                                                'PP_ALIGN\n'
                                                                                                '\n'
                                                                                                'pptx_path = '
                                                                                                "'/home/user/Desktop/134_2.pptx'\n"
                                                                                                'prs = '
                                                                                                'Presentation(pptx_path)\n'
                                                                                                'slide_height '
                                                                                                '= '
                                                                                                'prs.slide_height\n'
                                                                                                '\n'
                                                                                                '# Fix 1: '
                                                                                                'Set Slide 1 '
                                                                                                'title '
                                                                                                'placeholder '
                                                                                                'alignment '
                                                                                                'to CENTER\n'
                                                                                                'slide1 = '
                                                                                                'prs.slides[0]\n'
                                                                                                'title_shape '
                                                                                                '= '
                                                                                                'slide1.shapes[0]\n'
                                                                                                'if '
                                                                                                'hasattr(title_shape, '
                                                                                                "'text_frame'):\n"
                                                                                                '    for '
                                                                                                'para in '
                                                                                                'title_shape.text_frame.paragraphs:\n'
                                                                                                '        '
                                                                                                'para.alignment '
                                                                                                '= '
                                                                                                'PP_ALIGN.CENTER\n'
                                                                                                '    '
                                                                                                'print("Set '
                                                                                                'Slide 1 '
                                                                                                'title '
                                                                                                'alignment '
                                                                                                'to '
                                                                                                'CENTER")\n'
                                                                                                '\n'
                                                                                                '# Fix 2: '
                                                                                                'Move Slide '
                                                                                                '2 title to '
                                                                                                'bottom\n'
                                                                                                'slide2 = '
                                                                                                'prs.slides[1]\n'
                                                                                                'for shape '
                                                                                                'in '
                                                                                                'slide2.shapes:\n'
                                                                                                '    if '
                                                                                                'hasattr(shape, '
                                                                                                "'text') and "
                                                                                                'shape.text.strip() '
                                                                                                "== 'Product "
                                                                                                "Comparison':\n"
                                                                                                '        '
                                                                                                'new_top = '
                                                                                                'slide_height '
                                                                                                '- '
                                                                                                'shape.height\n'
                                                                                                '        '
                                                                                                'print(f"Moving '
                                                                                                'Slide 2 '
                                                                                                'title from '
                                                                                                'top={shape.top} '
                                                                                                'to '
                                                                                                'top={new_top}")\n'
                                                                                                '        '
                                                                                                'shape.top = '
                                                                                                'new_top\n'
                                                                                                '        '
                                                                                                'break\n'
                                                                                                '\n'
                                                                                                '# Fix 3: '
                                                                                                'Adjust '
                                                                                                'table '
                                                                                                'height to '
                                                                                                'match gold\n'
                                                                                                'for shape '
                                                                                                'in '
                                                                                                'slide2.shapes:\n'
                                                                                                '    if '
                                                                                                'shape.shape_type '
                                                                                                '== 19:  # '
                                                                                                'TABLE\n'
                                                                                                '        '
                                                                                                'print(f"Adjusting '
                                                                                                'table '
                                                                                                'height from '
                                                                                                '{shape.height} '
                                                                                                'to '
                                                                                                '2353680")\n'
                                                                                                '        '
                                                                                                'shape.height '
                                                                                                '= 2353680\n'
                                                                                                '        '
                                                                                                'break\n'
                                                                                                '\n'
                                                                                                'prs.save(pptx_path)\n'
                                                                                                'print("Done")\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '21760ecb-8f62-40d2-8d85-0cee5725cb72': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from pptx '
                                                                                                'import '
                                                                                                'Presentation\n'
                                                                                                'from '
                                                                                                'pptx.oxml.ns '
                                                                                                'import qn\n'
                                                                                                '\n'
                                                                                                'prs = '
                                                                                                "Presentation('/home/user/Desktop/Ch4 "
                                                                                                'Video '
                                                                                                "Effect.pptx')\n"
                                                                                                'slide = '
                                                                                                'prs.slides[0]\n'
                                                                                                '\n'
                                                                                                '# Remove '
                                                                                                'existing '
                                                                                                'transition '
                                                                                                'if any\n'
                                                                                                'existing = '
                                                                                                "slide._element.findall(qn('p:transition'))\n"
                                                                                                'for e in '
                                                                                                'existing:\n'
                                                                                                '    '
                                                                                                'slide._element.remove(e)\n'
                                                                                                '\n'
                                                                                                '# Add '
                                                                                                'dissolve '
                                                                                                'transition '
                                                                                                'to first '
                                                                                                'slide\n'
                                                                                                'transition '
                                                                                                '= '
                                                                                                "slide._element.makeelement(qn('p:transition'), "
                                                                                                '{\n'
                                                                                                "    'spd': "
                                                                                                "'med',\n"
                                                                                                '})\n'
                                                                                                'dissolve = '
                                                                                                "transition.makeelement(qn('p:dissolve'), "
                                                                                                '{})\n'
                                                                                                'transition.append(dissolve)\n'
                                                                                                'slide._element.append(transition)\n'
                                                                                                "prs.save('/home/user/Desktop/Ch4 "
                                                                                                'Video '
                                                                                                "Effect.pptx')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '2b94c692-6abb-48ae-ab0b-b3e8a19cb340': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from pptx '
                                                                                                'import '
                                                                                                'Presentation\n'
                                                                                                'from '
                                                                                                'pptx.util '
                                                                                                'import Emu\n'
                                                                                                '\n'
                                                                                                'prs = '
                                                                                                "Presentation('/home/user/Desktop/201_6.pptx')\n"
                                                                                                'slide_width '
                                                                                                '= '
                                                                                                'prs.slide_width\n'
                                                                                                '\n'
                                                                                                '# Slide 2 '
                                                                                                '(index 1) - '
                                                                                                'move image '
                                                                                                'to the '
                                                                                                'right side\n'
                                                                                                'slide = '
                                                                                                'prs.slides[1]\n'
                                                                                                'for shape '
                                                                                                'in '
                                                                                                'slide.shapes:\n'
                                                                                                '    # Find '
                                                                                                'image '
                                                                                                'shapes '
                                                                                                '(shape_type '
                                                                                                '13 = '
                                                                                                'picture)\n'
                                                                                                '    if '
                                                                                                'shape.shape_type '
                                                                                                '== 13:\n'
                                                                                                '        # '
                                                                                                'Move to '
                                                                                                'right side: '
                                                                                                'set left so '
                                                                                                'the right '
                                                                                                'edge aligns '
                                                                                                'with slide '
                                                                                                'width\n'
                                                                                                '        '
                                                                                                'shape.left '
                                                                                                '= '
                                                                                                'slide_width '
                                                                                                '- '
                                                                                                'shape.width\n'
                                                                                                '        '
                                                                                                'break\n'
                                                                                                '\n'
                                                                                                "prs.save('/home/user/Desktop/201_6.pptx')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '2cd43775-7085-45d8-89fa-9e35c0a915cf': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/libreoffice/4/user',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/libreoffice/4/user/registrymodifications.xcu'\n"
                                                                                                'lines = []\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'lines = '
                                                                                                'f.readlines()\n'
                                                                                                '# '
                                                                                                'Add/update '
                                                                                                'auto-save '
                                                                                                'settings\n'
                                                                                                'save_line = '
                                                                                                "'<item "
                                                                                                'oor:path="/org.openoffice.Office.Common/Save/Document"><prop '
                                                                                                'oor:name="AutoSave" '
                                                                                                'oor:op="fuse"><value>true</value></prop></item>\\n\'\n'
                                                                                                'interval_line '
                                                                                                "= '<item "
                                                                                                'oor:path="/org.openoffice.Office.Common/Save/Document"><prop '
                                                                                                'oor:name="AutoSaveTimeIntervall" '
                                                                                                'oor:op="fuse"><value>3</value></prop></item>\\n\'\n'
                                                                                                '# Remove '
                                                                                                'old '
                                                                                                'entries\n'
                                                                                                'lines = [l '
                                                                                                'for l in '
                                                                                                'lines if '
                                                                                                "'AutoSave' "
                                                                                                'not in l '
                                                                                                'and '
                                                                                                "'AutoSaveTimeIntervall' "
                                                                                                'not in l]\n'
                                                                                                '# Insert '
                                                                                                'before '
                                                                                                'closing '
                                                                                                'tag\n'
                                                                                                'if lines '
                                                                                                'and '
                                                                                                "'</oor:items>' "
                                                                                                'in '
                                                                                                'lines[-1]:\n'
                                                                                                '    '
                                                                                                'lines.insert(-1, '
                                                                                                'save_line)\n'
                                                                                                '    '
                                                                                                'lines.insert(-1, '
                                                                                                'interval_line)\n'
                                                                                                'else:\n'
                                                                                                '    '
                                                                                                'lines.append(save_line)\n'
                                                                                                '    '
                                                                                                'lines.append(interval_line)\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'f.writelines(lines)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '3161d64e-3120-47b4-aaad-6a764a92493b': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/45_1.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/3161d64e-3120-47b4-aaad-6a764a92493b/45_1_Gold.pptx'",
                                                                                     'shell': True}}]},
    '358aa0a7-6677-453f-ae35-e440f004c31e': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/note-taking-strategies.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/358aa0a7-6677-453f-ae35-e440f004c31e/note-taking-strategies_Gold.pptx'",
                                                                                     'shell': True}}]},
    '39be0d19-634d-4475-8768-09c130f5425d': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/41_3.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/39be0d19-634d-4475-8768-09c130f5425d/41_3_Gold.pptx'",
                                                                                     'shell': True}}]},
    '3b27600c-3668-4abd-8f84-7bcdebbccbdb': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from pptx '
                                                                                                'import '
                                                                                                'Presentation\n'
                                                                                                'from '
                                                                                                'pptx.dml.color '
                                                                                                'import '
                                                                                                'RGBColor\n'
                                                                                                '\n'
                                                                                                'prs = '
                                                                                                "Presentation('/home/user/Desktop/lec17-gui-events.pptx')\n"
                                                                                                'for slide '
                                                                                                'in '
                                                                                                'prs.slides:\n'
                                                                                                '    bg = '
                                                                                                'slide.background\n'
                                                                                                '    fill = '
                                                                                                'bg.fill\n'
                                                                                                '    '
                                                                                                'fill.solid()\n'
                                                                                                '    '
                                                                                                'fill.fore_color.rgb '
                                                                                                '= '
                                                                                                'RGBColor(0, '
                                                                                                '0, 255)\n'
                                                                                                "prs.save('/home/user/Desktop/lec17-gui-events.pptx')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '455d3c66-7dc6-4537-a39a-36d3e9119df7': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/res.png' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/455d3c66-7dc6-4537-a39a-36d3e9119df7/wssf-project-plan-on-a-page.png'",
                                                                                     'shell': True}}]},
    '4ed5abd0-8b5d-47bd-839f-cacfa15ca37a': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/4_1.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/4ed5abd0-8b5d-47bd-839f-cacfa15ca37a/4_1_Gold.pptx'",
                                                                                     'shell': True}}]},
    '550ce7e7-747b-495f-b122-acdc4d0b8e54': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/New_Club_Spring_2018_Training.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/550ce7e7-747b-495f-b122-acdc4d0b8e54/New_Club_Spring_2018_Training_with_strike.data'",
                                                                                     'shell': True}}], 'evaluator_options': [{'examine_shape': False}, {'examine_shape': False}]},
    '57667013-ea97-417c-9dce-2713091e6e2a': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/1_2.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/57667013-ea97-417c-9dce-2713091e6e2a/1_2_Gold.pptx'",
                                                                                     'shell': True}}]},
    '5c1a6c3d-c1b3-47cb-9b01-8d1b7544ffa1': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/39_2.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/5c1a6c3d-c1b3-47cb-9b01-8d1b7544ffa1/39_2_Gold.pptx'",
                                                                                     'shell': True}}]},
    '5cfb9197-e72b-454b-900e-c06b0c802b40': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/33_1.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/5cfb9197-e72b-454b-900e-c06b0c802b40/33_1_Gold.pptx'",
                                                                                     'shell': True}}]},
    '5d901039-a89c-4bfb-967b-bf66f4df075e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from pptx '
                                                                                                'import '
                                                                                                'Presentation\n'
                                                                                                'from '
                                                                                                'pptx.util '
                                                                                                'import Emu\n'
                                                                                                '\n'
                                                                                                'prs = '
                                                                                                "Presentation('/home/user/Desktop/CPD_Background_Investigation_Process.pptx')\n"
                                                                                                'slide = '
                                                                                                'prs.slides[0]\n'
                                                                                                'slide_width '
                                                                                                '= '
                                                                                                'prs.slide_width\n'
                                                                                                'slide_height '
                                                                                                '= '
                                                                                                'prs.slide_height\n'
                                                                                                '\n'
                                                                                                '# Find the '
                                                                                                'image on '
                                                                                                'the first '
                                                                                                'slide\n'
                                                                                                'for shape '
                                                                                                'in '
                                                                                                'slide.shapes:\n'
                                                                                                '    if '
                                                                                                'shape.shape_type '
                                                                                                '== 13:  # '
                                                                                                'Picture\n'
                                                                                                '        '
                                                                                                'img_width = '
                                                                                                'shape.width\n'
                                                                                                '        '
                                                                                                'img_height '
                                                                                                '= '
                                                                                                'shape.height\n'
                                                                                                '        # '
                                                                                                'Calculate '
                                                                                                'aspect '
                                                                                                'ratio\n'
                                                                                                '        '
                                                                                                'aspect = '
                                                                                                'img_width / '
                                                                                                'img_height\n'
                                                                                                '        '
                                                                                                'slide_aspect '
                                                                                                '= '
                                                                                                'slide_width '
                                                                                                '/ '
                                                                                                'slide_height\n'
                                                                                                '        \n'
                                                                                                '        if '
                                                                                                'aspect > '
                                                                                                'slide_aspect:\n'
                                                                                                '            '
                                                                                                '# Image is '
                                                                                                'wider '
                                                                                                'relative to '
                                                                                                'slide - fit '
                                                                                                'to width, '
                                                                                                'may exceed '
                                                                                                'height\n'
                                                                                                '            '
                                                                                                'new_width = '
                                                                                                'slide_width\n'
                                                                                                '            '
                                                                                                'new_height '
                                                                                                '= '
                                                                                                'int(slide_width '
                                                                                                '/ aspect)\n'
                                                                                                '        '
                                                                                                'else:\n'
                                                                                                '            '
                                                                                                '# Image is '
                                                                                                'taller '
                                                                                                'relative to '
                                                                                                'slide - fit '
                                                                                                'to height, '
                                                                                                'may exceed '
                                                                                                'width  \n'
                                                                                                '            '
                                                                                                'new_height '
                                                                                                '= '
                                                                                                'slide_height\n'
                                                                                                '            '
                                                                                                'new_width = '
                                                                                                'int(slide_height '
                                                                                                '* aspect)\n'
                                                                                                '        \n'
                                                                                                '        # '
                                                                                                'But the '
                                                                                                'task says '
                                                                                                '"stretch to '
                                                                                                'fill" '
                                                                                                'keeping '
                                                                                                'proportion, '
                                                                                                'so use the '
                                                                                                'larger '
                                                                                                'dimension\n'
                                                                                                '        # '
                                                                                                'to fill the '
                                                                                                'entire '
                                                                                                'page\n'
                                                                                                '        if '
                                                                                                'aspect > '
                                                                                                'slide_aspect:\n'
                                                                                                '            '
                                                                                                'new_height '
                                                                                                '= '
                                                                                                'slide_height\n'
                                                                                                '            '
                                                                                                'new_width = '
                                                                                                'int(slide_height '
                                                                                                '* aspect)\n'
                                                                                                '        '
                                                                                                'else:\n'
                                                                                                '            '
                                                                                                'new_width = '
                                                                                                'slide_width\n'
                                                                                                '            '
                                                                                                'new_height '
                                                                                                '= '
                                                                                                'int(slide_width '
                                                                                                '/ aspect)\n'
                                                                                                '        \n'
                                                                                                '        '
                                                                                                'shape.width '
                                                                                                '= '
                                                                                                'new_width\n'
                                                                                                '        '
                                                                                                'shape.height '
                                                                                                '= '
                                                                                                'new_height\n'
                                                                                                '        # '
                                                                                                'Center\n'
                                                                                                '        '
                                                                                                'shape.left '
                                                                                                '= '
                                                                                                '(slide_width '
                                                                                                '- '
                                                                                                'new_width) '
                                                                                                '// 2\n'
                                                                                                '        '
                                                                                                'shape.top = '
                                                                                                '(slide_height '
                                                                                                '- '
                                                                                                'new_height) '
                                                                                                '// 2\n'
                                                                                                '        '
                                                                                                'break\n'
                                                                                                '\n'
                                                                                                "prs.save('/home/user/Desktop/CPD_Background_Investigation_Process.pptx')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '70bca0cc-c117-427e-b0be-4df7299ebeb6': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/71_6.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/70bca0cc-c117-427e-b0be-4df7299ebeb6/71_6_Gold.pptx'",
                                                                                     'shell': True}}]},
    '73c99fb9-f828-43ce-b87a-01dc07faa224': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/109_4.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/73c99fb9-f828-43ce-b87a-01dc07faa224/109_4_Gold.pptx'",
                                                                                     'shell': True}}]},
    '7ae48c60-f143-4119-b659-15b8f485eb9a': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/30_1.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/7ae48c60-f143-4119-b659-15b8f485eb9a/30_1_Gold.pptx'",
                                                                                     'shell': True}}]},
    '7dbc52a6-11e0-4c9a-a2cb-1e36cfda80d8': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/164_3.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/7dbc52a6-11e0-4c9a-a2cb-1e36cfda80d8/164_3_Gold.pptx'",
                                                                                     'shell': True}}]},
    '841b50aa-df53-47bd-a73a-22d3a9f73160': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/181_2.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/841b50aa-df53-47bd-a73a-22d3a9f73160/181_2_Gold.pptx'",
                                                                                     'shell': True}}]},
    '8979838c-54a5-4454-a2b8-3d135a1a5c8f': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/186_3.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/8979838c-54a5-4454-a2b8-3d135a1a5c8f/186_3_Gold.pptx'",
                                                                                     'shell': True}}]},
    '986fc832-6af2-417c-8845-9272b3a1528b': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/154_3.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/986fc832-6af2-417c-8845-9272b3a1528b/154_3_Gold.pptx'",
                                                                                     'shell': True}}]},
    '9cf05d24-6bd9-4dae-8967-f67d88f5d38a': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/214_9.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/9cf05d24-6bd9-4dae-8967-f67d88f5d38a/214_9_Gold.pptx'",
                                                                                     'shell': True}}]},
    '9ec204e4-f0a3-42f8-8458-b772a6797cab': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/MLA_Workshop_061X_Works_Cited.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/9ec204e4-f0a3-42f8-8458-b772a6797cab/MLA_Workshop_061X_Works_Cited_Gold.pptx'",
                                                                                     'shell': True}}]},
    'a097acff-6266-4291-9fbd-137af7ecd439': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/pre.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/a097acff-6266-4291-9fbd-137af7ecd439/Secrets-of-Monetizing-Video.pptx'",
                                                                                     'shell': True}}]},
    'a434992a-89df-4577-925c-0c58b747f0f4': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/16_2.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/a434992a-89df-4577-925c-0c58b747f0f4/16_2_Gold.pptx'",
                                                                                     'shell': True}}]},
    'a53f80cd-4a90-4490-8310-097b011433f6': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/21_0.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/a53f80cd-4a90-4490-8310-097b011433f6/21_0_Gold.pptx'",
                                                                                     'shell': True}}]},
    'a669ef01-ded5-4099-9ea9-25e99b569840': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Writing-Outlines.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/a669ef01-ded5-4099-9ea9-25e99b569840/Writing-Outlines_Gold.pptx'",
                                                                                     'shell': True}}]},
    'ac1b39ff-ee4d-4483-abce-c117e98942f0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall -9 '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'killall -9 '
                                                                                                'soffice '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 2; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 -c '
                                                                                                "'from pptx "
                                                                                                'import '
                                                                                                'Presentation\n'
                                                                                                'p = '
                                                                                                'Presentation("/home/user/Desktop/55_10.pptx")\n'
                                                                                                'slide = '
                                                                                                'p.slides[2]\n'
                                                                                                'slide_h = '
                                                                                                'p.slide_height\n'
                                                                                                'for shp in '
                                                                                                'slide.shapes:\n'
                                                                                                '    if '
                                                                                                'shp.shape_type '
                                                                                                '== 19:\n'
                                                                                                '        '
                                                                                                'shp.top = '
                                                                                                'slide_h - '
                                                                                                'shp.height\n'
                                                                                                'p.save("/home/user/Desktop/55_10.pptx")\n'
                                                                                                'print("saved")\' '
                                                                                                '2>&1 | tee '
                                                                                                '/tmp/pptx.log; '
                                                                                                'grep -q '
                                                                                                'saved '
                                                                                                '/tmp/pptx.log',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    'ac9bb6cb-1888-43ab-81e4-a98a547918cd': {   'actions': [   {   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True},
                                                                   'type': 'execute'},
                                                               {   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import zipfile\n'
                                                                                                'import shutil\n'
                                                                                                'import os\n'
                                                                                                'import '
                                                                                                'xml.etree.ElementTree '
                                                                                                'as ET\n'
                                                                                                '\n'
                                                                                                'pptx_path = '
                                                                                                "'/home/user/Desktop/saa-format-guide.pptx'\n"
                                                                                                'tmp_dir = '
                                                                                                "'/tmp/pptx_edit_ac9bb6cb'\n"
                                                                                                "TARGET = 'FF0000'\n"
                                                                                                '\n'
                                                                                                'if '
                                                                                                'os.path.exists(tmp_dir):\n'
                                                                                                '    '
                                                                                                'shutil.rmtree(tmp_dir)\n'
                                                                                                'os.makedirs(tmp_dir)\n'
                                                                                                '\n'
                                                                                                'with '
                                                                                                'zipfile.ZipFile(pptx_path, '
                                                                                                "'r') as z:\n"
                                                                                                '    '
                                                                                                'z.extractall(tmp_dir)\n'
                                                                                                '\n'
                                                                                                'A_NS = '
                                                                                                "'http://schemas.openxmlformats.org/drawingml/2006/main'\n"
                                                                                                'P_NS = '
                                                                                                "'http://schemas.openxmlformats.org/presentationml/2006/main'\n"
                                                                                                "ns = {'a': A_NS, 'p': "
                                                                                                'P_NS}\n'
                                                                                                'for prefix, uri in '
                                                                                                'ns.items():\n'
                                                                                                '    '
                                                                                                'ET.register_namespace(prefix, '
                                                                                                'uri)\n'
                                                                                                "ET.register_namespace('r', "
                                                                                                "'http://schemas.openxmlformats.org/officeDocument/2006/relationships')\n"
                                                                                                "ET.register_namespace('mc', "
                                                                                                "'http://schemas.openxmlformats.org/markup-compatibility/2006')\n"
                                                                                                '\n'
                                                                                                'sm_path = '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'ppt', "
                                                                                                "'slideMasters', "
                                                                                                "'slideMaster1.xml')\n"
                                                                                                'tree = '
                                                                                                'ET.parse(sm_path)\n'
                                                                                                'root = '
                                                                                                'tree.getroot()\n'
                                                                                                '\n'
                                                                                                'A = lambda t: '
                                                                                                "'{%s}%s' % (A_NS, t)\n"
                                                                                                '\n'
                                                                                                'def '
                                                                                                'force_solid_fill(rpr, '
                                                                                                'color_hex):\n'
                                                                                                '    for sf in '
                                                                                                "list(rpr.findall(A('solidFill'))):\n"
                                                                                                '        '
                                                                                                'rpr.remove(sf)\n'
                                                                                                '    sf = '
                                                                                                "ET.Element(A('solidFill'))\n"
                                                                                                '    clr = '
                                                                                                'ET.SubElement(sf, '
                                                                                                "A('srgbClr'))\n"
                                                                                                "    clr.set('val', "
                                                                                                'color_hex)\n'
                                                                                                '    rpr.insert(0, '
                                                                                                'sf)\n'
                                                                                                '\n'
                                                                                                '# Find every p:sp '
                                                                                                'whose ph type == '
                                                                                                '"sldNum".\n'
                                                                                                'sldnum_shapes = []\n'
                                                                                                'for sp in '
                                                                                                "root.findall('.//p:sp', "
                                                                                                'ns):\n'
                                                                                                '    ph = '
                                                                                                "sp.find('./p:nvSpPr/p:nvPr/p:ph', "
                                                                                                'ns)\n'
                                                                                                '    if ph is not None '
                                                                                                "and ph.get('type') == "
                                                                                                "'sldNum':\n"
                                                                                                '        '
                                                                                                'sldnum_shapes.append(sp)\n'
                                                                                                '\n'
                                                                                                "print('sldNum shapes "
                                                                                                "found:', "
                                                                                                'len(sldnum_shapes))\n'
                                                                                                '\n'
                                                                                                'for sp in '
                                                                                                'sldnum_shapes:\n'
                                                                                                '    # Patch all '
                                                                                                'run-property nodes '
                                                                                                'inside this shape.\n'
                                                                                                '    for tag in '
                                                                                                "('rPr', 'defRPr', "
                                                                                                "'endParaRPr'):\n"
                                                                                                '        for rpr in '
                                                                                                "sp.findall('.//' + "
                                                                                                'A(tag)):\n'
                                                                                                '            '
                                                                                                'force_solid_fill(rpr, '
                                                                                                'TARGET)\n'
                                                                                                '    # Ensure each '
                                                                                                'a:fld has an a:rPr '
                                                                                                'with red fill.\n'
                                                                                                '    for fld in '
                                                                                                "sp.findall('.//a:fld', "
                                                                                                'ns):\n'
                                                                                                '        rpr = '
                                                                                                "fld.find('./a:rPr', "
                                                                                                'ns)\n'
                                                                                                '        if rpr is '
                                                                                                'None:\n'
                                                                                                '            rpr = '
                                                                                                "ET.Element(A('rPr'))\n"
                                                                                                '            '
                                                                                                'fld.insert(0, rpr)\n'
                                                                                                '        '
                                                                                                'force_solid_fill(rpr, '
                                                                                                'TARGET)\n'
                                                                                                '\n'
                                                                                                '# '
                                                                                                'Belt-and-suspenders: '
                                                                                                'rewrite every srgbClr '
                                                                                                'outside the sldNum '
                                                                                                'shapes\n'
                                                                                                '# to 000000 so the '
                                                                                                'broken upstream '
                                                                                                'reverse-scan also '
                                                                                                'lands on FF0000.\n'
                                                                                                'sldnum_ids = set()\n'
                                                                                                'for sp in '
                                                                                                'sldnum_shapes:\n'
                                                                                                '    for el in '
                                                                                                'sp.iter():\n'
                                                                                                '        '
                                                                                                'sldnum_ids.add(id(el))\n'
                                                                                                '\n'
                                                                                                'for clr in '
                                                                                                "root.iter(A('srgbClr')):\n"
                                                                                                '    if id(clr) in '
                                                                                                'sldnum_ids:\n'
                                                                                                '        continue\n'
                                                                                                '    val = '
                                                                                                "clr.get('val')\n"
                                                                                                '    if val and '
                                                                                                'val.upper() != '
                                                                                                'TARGET:\n'
                                                                                                '        '
                                                                                                "clr.set('val', "
                                                                                                "'000000')\n"
                                                                                                '\n'
                                                                                                'tree.write(sm_path, '
                                                                                                'xml_declaration=True, '
                                                                                                "encoding='UTF-8')\n"
                                                                                                '\n'
                                                                                                'os.remove(pptx_path)\n'
                                                                                                'with '
                                                                                                'zipfile.ZipFile(pptx_path, '
                                                                                                "'w', "
                                                                                                'zipfile.ZIP_DEFLATED) '
                                                                                                'as z:\n'
                                                                                                '    for r_dir, _dirs, '
                                                                                                'files in '
                                                                                                'os.walk(tmp_dir):\n'
                                                                                                '        for f in '
                                                                                                'files:\n'
                                                                                                '            full_path '
                                                                                                '= os.path.join(r_dir, '
                                                                                                'f)\n'
                                                                                                '            arcname = '
                                                                                                'os.path.relpath(full_path, '
                                                                                                'tmp_dir)\n'
                                                                                                '            '
                                                                                                'z.write(full_path, '
                                                                                                'arcname)\n'
                                                                                                '\n'
                                                                                                'shutil.rmtree(tmp_dir)\n'
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True},
                                                                   'type': 'execute'}]},
    'af23762e-2bfd-4a1d-aada-20fa8de9ce07': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Forests.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/af23762e-2bfd-4a1d-aada-20fa8de9ce07/Forests_Gold.pptx'",
                                                                                     'shell': True}}]},
    'af2d657a-e6b3-4c6a-9f67-9e3ed015974c': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/9_1.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/af2d657a-e6b3-4c6a-9f67-9e3ed015974c/9_1_Gold.pptx'",
                                                                                     'shell': True}}]},
    'b8adbc24-cef2-4b15-99d5-ecbe7ff445eb': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/189_4.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/b8adbc24-cef2-4b15-99d5-ecbe7ff445eb/189_4_Gold.pptx'",
                                                                                     'shell': True}}]},
    'bf4e9888-f10f-47af-8dba-76413038b73c': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/4.3-Template_4.29.2016.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/bf4e9888-f10f-47af-8dba-76413038b73c/4.3-Template_4.29.2016_Gold.pptx'",
                                                                                     'shell': True}}]},
    'c59742c0-4323-4b9d-8a02-723c251deaa0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'zipfile, '
                                                                                                'shutil, os\n'
                                                                                                'import '
                                                                                                'xml.etree.ElementTree '
                                                                                                'as ET\n'
                                                                                                '\n'
                                                                                                'pptx_path = '
                                                                                                "'/home/user/Desktop/Mady_and_Mia_Baseball.pptx'\n"
                                                                                                'mp3_path = '
                                                                                                "'/home/user/Desktop/Baseball.mp3'\n"
                                                                                                'tmp_dir = '
                                                                                                "'/tmp/pptx_edit_c59742c0'\n"
                                                                                                '\n'
                                                                                                'if '
                                                                                                'os.path.exists(tmp_dir):\n'
                                                                                                '    '
                                                                                                'shutil.rmtree(tmp_dir)\n'
                                                                                                'os.makedirs(tmp_dir)\n'
                                                                                                '\n'
                                                                                                'with '
                                                                                                'zipfile.ZipFile(pptx_path, '
                                                                                                "'r') as z:\n"
                                                                                                '    '
                                                                                                'z.extractall(tmp_dir)\n'
                                                                                                '\n'
                                                                                                '# Copy the '
                                                                                                'mp3 into '
                                                                                                'ppt/media/\n'
                                                                                                'media_dir = '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'ppt/media')\n"
                                                                                                'os.makedirs(media_dir, '
                                                                                                'exist_ok=True)\n'
                                                                                                'shutil.copy2(mp3_path, '
                                                                                                'os.path.join(media_dir, '
                                                                                                "'Baseball.mp3'))\n"
                                                                                                '\n'
                                                                                                '# Add '
                                                                                                'relationship '
                                                                                                'to slide1\n'
                                                                                                'slide_rels_path '
                                                                                                '= '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'ppt/slides/_rels/slide1.xml.rels')\n"
                                                                                                'if '
                                                                                                'os.path.exists(slide_rels_path):\n'
                                                                                                '    '
                                                                                                'rels_tree = '
                                                                                                'ET.parse(slide_rels_path)\n'
                                                                                                '    '
                                                                                                'rels_root = '
                                                                                                'rels_tree.getroot()\n'
                                                                                                '    # Find '
                                                                                                'next rId\n'
                                                                                                '    max_id '
                                                                                                '= 0\n'
                                                                                                '    for rel '
                                                                                                'in '
                                                                                                'rels_root:\n'
                                                                                                '        rid '
                                                                                                '= '
                                                                                                "rel.get('Id', "
                                                                                                "'')\n"
                                                                                                '        if '
                                                                                                "rid.startswith('rId'):\n"
                                                                                                '            '
                                                                                                'try:\n'
                                                                                                '                '
                                                                                                'max_id = '
                                                                                                'max(max_id, '
                                                                                                'int(rid[3:]))\n'
                                                                                                '            '
                                                                                                'except '
                                                                                                'ValueError:\n'
                                                                                                '                '
                                                                                                'pass\n'
                                                                                                '    new_rid '
                                                                                                '= '
                                                                                                "f'rId{max_id "
                                                                                                "+ 1}'\n"
                                                                                                '\n'
                                                                                                '    '
                                                                                                "ET.register_namespace('', "
                                                                                                "'http://schemas.openxmlformats.org/package/2006/relationships')\n"
                                                                                                '    new_rel '
                                                                                                '= '
                                                                                                'ET.SubElement(rels_root, '
                                                                                                "'Relationship')\n"
                                                                                                '    '
                                                                                                "new_rel.set('Id', "
                                                                                                'new_rid)\n'
                                                                                                '    '
                                                                                                "new_rel.set('Type', "
                                                                                                "'http://schemas.openxmlformats.org/officeDocument/2006/relationships/audio')\n"
                                                                                                '    '
                                                                                                "new_rel.set('Target', "
                                                                                                "'../media/Baseball.mp3')\n"
                                                                                                '    '
                                                                                                'rels_tree.write(slide_rels_path, '
                                                                                                'xml_declaration=True, '
                                                                                                "encoding='UTF-8')\n"
                                                                                                '\n'
                                                                                                '# Update '
                                                                                                '[Content_Types].xml '
                                                                                                'to include '
                                                                                                'mp3\n'
                                                                                                'ct_path = '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'[Content_Types].xml')\n"
                                                                                                'ct_tree = '
                                                                                                'ET.parse(ct_path)\n'
                                                                                                'ct_root = '
                                                                                                'ct_tree.getroot()\n'
                                                                                                "ET.register_namespace('', "
                                                                                                "'http://schemas.openxmlformats.org/package/2006/content-types')\n"
                                                                                                '# Check if '
                                                                                                'mp3 '
                                                                                                'extension '
                                                                                                'already '
                                                                                                'exists\n'
                                                                                                'has_mp3 = '
                                                                                                'False\n'
                                                                                                'for elem in '
                                                                                                'ct_root:\n'
                                                                                                '    if '
                                                                                                "elem.get('Extension') "
                                                                                                "== 'mp3':\n"
                                                                                                '        '
                                                                                                'has_mp3 = '
                                                                                                'True\n'
                                                                                                '        '
                                                                                                'break\n'
                                                                                                'if not '
                                                                                                'has_mp3:\n'
                                                                                                '    '
                                                                                                'ext_elem = '
                                                                                                'ET.SubElement(ct_root, '
                                                                                                "'Default')\n"
                                                                                                '    '
                                                                                                "ext_elem.set('Extension', "
                                                                                                "'mp3')\n"
                                                                                                '    '
                                                                                                "ext_elem.set('ContentType', "
                                                                                                "'audio/mpeg')\n"
                                                                                                'ct_tree.write(ct_path, '
                                                                                                'xml_declaration=True, '
                                                                                                "encoding='UTF-8')\n"
                                                                                                '\n'
                                                                                                '# '
                                                                                                'Repackage\n'
                                                                                                'os.remove(pptx_path)\n'
                                                                                                'with '
                                                                                                'zipfile.ZipFile(pptx_path, '
                                                                                                "'w', "
                                                                                                'zipfile.ZIP_DEFLATED) '
                                                                                                'as zf:\n'
                                                                                                '    for '
                                                                                                'root_dir, '
                                                                                                'dirs, files '
                                                                                                'in '
                                                                                                'os.walk(tmp_dir):\n'
                                                                                                '        for '
                                                                                                'file in '
                                                                                                'files:\n'
                                                                                                '            '
                                                                                                'file_path = '
                                                                                                'os.path.join(root_dir, '
                                                                                                'file)\n'
                                                                                                '            '
                                                                                                'arcname = '
                                                                                                'os.path.relpath(file_path, '
                                                                                                'tmp_dir)\n'
                                                                                                '            '
                                                                                                'zf.write(file_path, '
                                                                                                'arcname)\n'
                                                                                                "print('Done "
                                                                                                '- audio '
                                                                                                "added')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'c82632a4-56b6-4db4-9dd1-3820ee3388e4': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/31_2.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/c82632a4-56b6-4db4-9dd1-3820ee3388e4/31_2_Gold.pptx'",
                                                                                     'shell': True}}]},
    'ce88f674-ab7a-43da-9201-468d38539e4a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'zipfile, '
                                                                                                'shutil, os\n'
                                                                                                'import '
                                                                                                'xml.etree.ElementTree '
                                                                                                'as ET\n'
                                                                                                '\n'
                                                                                                'pptx_path = '
                                                                                                "'/home/user/Desktop/AM_Last_Page_Template.pptx'\n"
                                                                                                'tmp_dir = '
                                                                                                "'/tmp/pptx_edit_ce88f674'\n"
                                                                                                '\n'
                                                                                                'if '
                                                                                                'os.path.exists(tmp_dir):\n'
                                                                                                '    '
                                                                                                'shutil.rmtree(tmp_dir)\n'
                                                                                                'os.makedirs(tmp_dir)\n'
                                                                                                '\n'
                                                                                                'with '
                                                                                                'zipfile.ZipFile(pptx_path, '
                                                                                                "'r') as z:\n"
                                                                                                '    '
                                                                                                'z.extractall(tmp_dir)\n'
                                                                                                '\n'
                                                                                                '# Modify '
                                                                                                'presentation.xml '
                                                                                                'to swap '
                                                                                                'slide '
                                                                                                'dimensions\n'
                                                                                                'pres_path = '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'ppt/presentation.xml')\n"
                                                                                                '\n'
                                                                                                '# Register '
                                                                                                'all '
                                                                                                'namespaces\n'
                                                                                                'namespaces '
                                                                                                '= {}\n'
                                                                                                'for event, '
                                                                                                '(prefix, '
                                                                                                'uri) in '
                                                                                                'ET.iterparse(pres_path, '
                                                                                                "events=['start-ns']):\n"
                                                                                                '    '
                                                                                                'namespaces[prefix] '
                                                                                                '= uri\n'
                                                                                                'for prefix, '
                                                                                                'uri in '
                                                                                                'namespaces.items():\n'
                                                                                                '    '
                                                                                                'ET.register_namespace(prefix, '
                                                                                                'uri)\n'
                                                                                                '\n'
                                                                                                'tree = '
                                                                                                'ET.parse(pres_path)\n'
                                                                                                'root = '
                                                                                                'tree.getroot()\n'
                                                                                                '\n'
                                                                                                'sldSz = '
                                                                                                "root.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}sldSz')\n"
                                                                                                'if sldSz is '
                                                                                                'not None:\n'
                                                                                                '    cx = '
                                                                                                "sldSz.get('cx')\n"
                                                                                                '    cy = '
                                                                                                "sldSz.get('cy')\n"
                                                                                                '    # Swap '
                                                                                                'width and '
                                                                                                'height for '
                                                                                                'portrait\n'
                                                                                                '    '
                                                                                                "sldSz.set('cx', "
                                                                                                'cy)\n'
                                                                                                '    '
                                                                                                "sldSz.set('cy', "
                                                                                                'cx)\n'
                                                                                                '    '
                                                                                                "sldSz.set('type', "
                                                                                                "'custom')\n"
                                                                                                '    '
                                                                                                "print(f'Swapped: "
                                                                                                '{cx}x{cy} '
                                                                                                '-> '
                                                                                                "{cy}x{cx}')\n"
                                                                                                '\n'
                                                                                                'tree.write(pres_path, '
                                                                                                'xml_declaration=True, '
                                                                                                "encoding='UTF-8')\n"
                                                                                                '\n'
                                                                                                'os.remove(pptx_path)\n'
                                                                                                'with '
                                                                                                'zipfile.ZipFile(pptx_path, '
                                                                                                "'w', "
                                                                                                'zipfile.ZIP_DEFLATED) '
                                                                                                'as zf:\n'
                                                                                                '    for '
                                                                                                'root_dir, '
                                                                                                'dirs, files '
                                                                                                'in '
                                                                                                'os.walk(tmp_dir):\n'
                                                                                                '        for '
                                                                                                'file in '
                                                                                                'files:\n'
                                                                                                '            '
                                                                                                'file_path = '
                                                                                                'os.path.join(root_dir, '
                                                                                                'file)\n'
                                                                                                '            '
                                                                                                'arcname = '
                                                                                                'os.path.relpath(file_path, '
                                                                                                'tmp_dir)\n'
                                                                                                '            '
                                                                                                'zf.write(file_path, '
                                                                                                'arcname)\n'
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'e4ef0baf-4b52-4590-a47e-d4d464cca2d7': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/42_2.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/e4ef0baf-4b52-4590-a47e-d4d464cca2d7/42_2_Gold.pptx'",
                                                                                     'shell': True}}]},
    'ed43c15f-00cb-4054-9c95-62c880865d68': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from pptx '
                                                                                                'import '
                                                                                                'Presentation\n'
                                                                                                'from '
                                                                                                'pptx.util '
                                                                                                'import Emu\n'
                                                                                                '\n'
                                                                                                'prs = '
                                                                                                "Presentation('/home/user/Desktop/43_1.pptx')\n"
                                                                                                '\n'
                                                                                                '# Task: '
                                                                                                'Move '
                                                                                                'picture on '
                                                                                                'page 2 to '
                                                                                                'slide top, '
                                                                                                'make '
                                                                                                'textboxes '
                                                                                                'underlined '
                                                                                                'on slide 1 '
                                                                                                'and 2\n'
                                                                                                '\n'
                                                                                                '# 1. Move '
                                                                                                'picture on '
                                                                                                'slide 2 '
                                                                                                '(index 1) '
                                                                                                'to top\n'
                                                                                                'slide2 = '
                                                                                                'prs.slides[1]\n'
                                                                                                'for shape '
                                                                                                'in '
                                                                                                'slide2.shapes:\n'
                                                                                                '    if '
                                                                                                'shape.shape_type '
                                                                                                '== 13:  # '
                                                                                                'Picture\n'
                                                                                                '        '
                                                                                                'shape.top = '
                                                                                                '0\n'
                                                                                                '        '
                                                                                                'break\n'
                                                                                                '\n'
                                                                                                '# 2. Make '
                                                                                                'textboxes '
                                                                                                'underlined '
                                                                                                'on slides 1 '
                                                                                                'and 2\n'
                                                                                                'for '
                                                                                                'slide_idx '
                                                                                                'in [0, 1]:\n'
                                                                                                '    slide = '
                                                                                                'prs.slides[slide_idx]\n'
                                                                                                '    for '
                                                                                                'shape in '
                                                                                                'slide.shapes:\n'
                                                                                                '        if '
                                                                                                'shape.has_text_frame:\n'
                                                                                                '            '
                                                                                                'for para in '
                                                                                                'shape.text_frame.paragraphs:\n'
                                                                                                '                '
                                                                                                'for run in '
                                                                                                'para.runs:\n'
                                                                                                '                    '
                                                                                                'run.font.underline '
                                                                                                '= True\n'
                                                                                                '\n'
                                                                                                "prs.save('/home/user/Desktop/43_1.pptx')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'edb61b14-a854-4bf5-a075-c8075c11293a': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/24_8.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/edb61b14-a854-4bf5-a075-c8075c11293a/24_8_Gold.pptx'",
                                                                                     'shell': True}}]},
    # ef9d12bd (P6 check_left_panel): kill LO, clear panel-hide xcu keys + crash
    # recovery state (otherwise the relaunch shows a "Document Recovery" dialog
    # instead of an empty Impress doc, and check_left_panel finds no Slides View
    # frame), then relaunch with --norestore --nologo so no recovery dialog.
    'ef9d12bd-bcee-4ba0-a40e-918400f43ddf': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'dbus-send '
                                                                                                '--session '
                                                                                                '--print-reply '
                                                                                                '--dest=org.a11y.Bus '
                                                                                                '/org/a11y/bus '
                                                                                                'org.freedesktop.DBus.Properties.Set '
                                                                                                'string:org.a11y.Status '
                                                                                                'string:IsEnabled '
                                                                                                'variant:boolean:true '
                                                                                                '2>/dev/null; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                '-9 -q '
                                                                                                'soffice.bin '
                                                                                                'oosplash '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1; '
                                                                                                'sed -i '
                                                                                                "'/SlideSorterBar/d' "
                                                                                                '/home/user/.config/libreoffice/4/user/registrymodifications.xcu '
                                                                                                '2>/dev/null; '
                                                                                                'sed -i '
                                                                                                "'/LeftPaneVisible/d' "
                                                                                                '/home/user/.config/libreoffice/4/user/registrymodifications.xcu '
                                                                                                '2>/dev/null; '
                                                                                                'sed -i '
                                                                                                "'/SlideSorter/d' "
                                                                                                '/home/user/.config/libreoffice/4/user/registrymodifications.xcu '
                                                                                                '2>/dev/null; '
                                                                                                'rm -rf '
                                                                                                '/home/user/.config/libreoffice/4/user/backup '
                                                                                                '2>/dev/null; '
                                                                                                'rm -rf '
                                                                                                '/home/user/.config/libreoffice/4/user/temp '
                                                                                                '2>/dev/null; '
                                                                                                "find /home/user/.config/libreoffice -name '*.lock' -delete "
                                                                                                '2>/dev/null; '
                                                                                                "find /home/user -name '.~lock.*' -delete "
                                                                                                '2>/dev/null; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'libreoffice',
                                                                                                    '--impress',
                                                                                                    '--norestore',
                                                                                                    '--nologo']}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'for i in '
                                                                                                '$(seq 1 '
                                                                                                '30); do '
                                                                                                'wmctrl -l '
                                                                                                '2>/dev/null '
                                                                                                "| grep -qi 'Impress' "
                                                                                                '&& break; '
                                                                                                'sleep 1; '
                                                                                                'done; sleep '
                                                                                                '5; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sleep 3; '
                                                                                                'true',
                                                                                     'shell': True}}]},
    'f23acfd2-c485-4b7c-a1e7-d4303ddfe864': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/69_4.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_impress/f23acfd2-c485-4b7c-a1e7-d4303ddfe864/69_4_Gold.pptx'",
                                                                                     'shell': True}}]}}
