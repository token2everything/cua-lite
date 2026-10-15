"""Hand-curated oracle metadata for gimp eval tasks.

GIMP image-editing tasks.

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
- total entries:           16
- with actions:            16
- after_postconfig=True:   1
- block: exclude_reason:   0
- evaluator override:      0
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {   '06ca5602-62ca-47f6-ad4f-da151cde54cc': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from PIL '
                                                                                                'import '
                                                                                                'Image\n'
                                                                                                'img = '
                                                                                                "Image.open('/home/user/Desktop/computer.png')\n"
                                                                                                '# Convert '
                                                                                                'to palette '
                                                                                                'with max '
                                                                                                'colors and '
                                                                                                'no '
                                                                                                'dithering '
                                                                                                'for best '
                                                                                                'structure '
                                                                                                'preservation\n'
                                                                                                'img_p = '
                                                                                                'img.quantize(colors=256, '
                                                                                                'method=Image.Quantize.MEDIANCUT, '
                                                                                                'dither=Image.Dither.NONE)\n'
                                                                                                '# Save as '
                                                                                                'palette-based '
                                                                                                'PNG\n'
                                                                                                "img_p.save('/home/user/Desktop/computer.png')\n"
                                                                                                "img_p.save('/home/user/Desktop/palette_computer.png')\n"
                                                                                                "print(f'Converted "
                                                                                                'to '
                                                                                                'palette-based: '
                                                                                                "mode={img_p.mode}')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}], 'evaluator': {'func': 'check_palette_and_structure_sim', 'expected': {'type': 'cloud_file', 'path': 'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/gimp/06ca5602-62ca-47f6-ad4f-da151cde54cc/computer.png', 'dest': 'computer.png'}, 'result': {'type': 'vm_file', 'path': '/home/user/Desktop/computer.png', 'dest': 'computer.png'}}},
    '2a729ded-3296-423d-aec4-7dd55ed5fbb3': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/dog_without_background.png' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/gimp/2a729ded-3296-423d-aec4-7dd55ed5fbb3/dog_cutout_gold.png'",
                                                                                     'shell': True}}]},
    '554785e9-4523-4e7a-b8e1-8016f565f56a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from PIL '
                                                                                                'import '
                                                                                                'Image, '
                                                                                                'ImageEnhance\n'
                                                                                                '\n'
                                                                                                'img = '
                                                                                                "Image.open('/home/user/Desktop/woman_sitting_by_the_tree2.png')\n"
                                                                                                '# Increase '
                                                                                                'saturation\n'
                                                                                                'enhancer = '
                                                                                                'ImageEnhance.Color(img)\n'
                                                                                                'img_enhanced '
                                                                                                '= '
                                                                                                'enhancer.enhance(1.5)  '
                                                                                                '# 1.5x '
                                                                                                'saturation\n'
                                                                                                "img_enhanced.save('/home/user/Desktop/edited_colorful.png')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '72f83cdc-bf76-4531-9a1b-eb893a13f8aa': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from PIL '
                                                                                                'import '
                                                                                                'Image\n'
                                                                                                '\n'
                                                                                                'img = '
                                                                                                "Image.open('/home/user/Desktop/berry.png')\n"
                                                                                                '# Mirror '
                                                                                                'horizontally '
                                                                                                '(flip '
                                                                                                'left-right)\n'
                                                                                                'img_mirror '
                                                                                                '= '
                                                                                                'img.transpose(Image.FLIP_LEFT_RIGHT)\n'
                                                                                                "img_mirror.save('/home/user/Desktop/berry_mirror.png')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '734d6579-c07d-47a8-9ae2-13339795476b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from PIL '
                                                                                                'import '
                                                                                                'Image\n'
                                                                                                'import '
                                                                                                'numpy as '
                                                                                                'np\n'
                                                                                                '\n'
                                                                                                '# Load the '
                                                                                                'original '
                                                                                                'white '
                                                                                                'background '
                                                                                                'image (the '
                                                                                                'flattened '
                                                                                                'PNG '
                                                                                                'version)\n'
                                                                                                'img = '
                                                                                                "Image.open('/home/user/Desktop/white_background_with_object.png')\n"
                                                                                                'pixels = '
                                                                                                'np.array(img)\n'
                                                                                                '\n'
                                                                                                '# The '
                                                                                                'expected '
                                                                                                '(tgt) image '
                                                                                                'is '
                                                                                                'white_background_with_object.png\n'
                                                                                                '# '
                                                                                                'check_green_background '
                                                                                                'iterates '
                                                                                                'over tgt '
                                                                                                'pixels, '
                                                                                                'finds '
                                                                                                'non-black '
                                                                                                'ones,\n'
                                                                                                '# then '
                                                                                                'checks if '
                                                                                                'corresponding '
                                                                                                'src pixel '
                                                                                                'has g > r '
                                                                                                'and g > b\n'
                                                                                                '# So we '
                                                                                                'need to '
                                                                                                'make all '
                                                                                                'non-black '
                                                                                                'background '
                                                                                                'pixels '
                                                                                                'green\n'
                                                                                                'for x in '
                                                                                                'range(pixels.shape[0]):\n'
                                                                                                '    for y '
                                                                                                'in '
                                                                                                'range(pixels.shape[1]):\n'
                                                                                                '        r, '
                                                                                                'g, b = '
                                                                                                'pixels[x, '
                                                                                                'y][:3]\n'
                                                                                                '        # '
                                                                                                'If the '
                                                                                                'pixel is '
                                                                                                'black '
                                                                                                '(object), '
                                                                                                'keep it; '
                                                                                                'otherwise '
                                                                                                'make it '
                                                                                                'green\n'
                                                                                                '        if '
                                                                                                '(r, g, b) '
                                                                                                '!= (0, 0, '
                                                                                                '0):\n'
                                                                                                '            '
                                                                                                'pixels[x, '
                                                                                                'y][0] = '
                                                                                                '0    # R\n'
                                                                                                '            '
                                                                                                'pixels[x, '
                                                                                                'y][1] = '
                                                                                                '255  # G\n'
                                                                                                '            '
                                                                                                'pixels[x, '
                                                                                                'y][2] = '
                                                                                                '0    # B\n'
                                                                                                '\n'
                                                                                                'result = '
                                                                                                'Image.fromarray(pixels)\n'
                                                                                                "result.save('/home/user/Desktop/green_background_with_object.png')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '7767eef2-56a3-4cea-8c9f-48c070c7d65b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/GIMP/2.10',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/GIMP/2.10/gimprc'\n"
                                                                                                'lines = []\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'lines = '
                                                                                                'f.readlines()\n'
                                                                                                '\n'
                                                                                                '# Remove '
                                                                                                'existing '
                                                                                                'theme line '
                                                                                                'if present\n'
                                                                                                'lines = [l '
                                                                                                'for l in '
                                                                                                'lines if '
                                                                                                'not '
                                                                                                "l.strip().startswith('(theme "
                                                                                                "')]\n"
                                                                                                '\n'
                                                                                                '# Add the '
                                                                                                'Light theme '
                                                                                                'setting\n'
                                                                                                "lines.append('(theme "
                                                                                                '"Light")\\n\')\n'
                                                                                                '\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'f.writelines(lines)\n'
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '77b8ab4d-994f-43ac-8930-8ca087d7c4b4': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/export.jpg' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/gimp/77b8ab4d-994f-43ac-8930-8ca087d7c4b4/The_Lost_River_Of_Dreams.jpg'",
                                                                                     'shell': True}}]},
    '7a4deb26-d57d-4ea9-9a73-630f66a7b568': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from PIL '
                                                                                                'import '
                                                                                                'Image, '
                                                                                                'ImageEnhance\n'
                                                                                                '\n'
                                                                                                'img = '
                                                                                                "Image.open('/home/user/Desktop/woman_sitting_by_the_tree.png')\n"
                                                                                                '# Decrease '
                                                                                                'brightness\n'
                                                                                                'enhancer = '
                                                                                                'ImageEnhance.Brightness(img)\n'
                                                                                                'img_darker '
                                                                                                '= '
                                                                                                'enhancer.enhance(0.6)  '
                                                                                                '# 60% '
                                                                                                'brightness\n'
                                                                                                "img_darker.save('/home/user/Desktop/edited_darker.png')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '7b7617bd-57cc-468e-9c91-40c4ec2bcb3d': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/GIMP/2.10',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/GIMP/2.10/gimprc'\n"
                                                                                                'lines = []\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'lines = '
                                                                                                'f.readlines()\n'
                                                                                                '\n'
                                                                                                '# Remove '
                                                                                                'existing '
                                                                                                'undo-levels '
                                                                                                'line if '
                                                                                                'present\n'
                                                                                                'lines = [l '
                                                                                                'for l in '
                                                                                                'lines if '
                                                                                                'not '
                                                                                                "l.strip().startswith('(undo-levels "
                                                                                                "')]\n"
                                                                                                '\n'
                                                                                                '# Add the '
                                                                                                'undo-levels '
                                                                                                'setting\n'
                                                                                                "lines.append('(undo-levels "
                                                                                                "100)\\n')\n"
                                                                                                '\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'f.writelines(lines)\n'
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'a746add2-cab0-4740-ac36-c3769d9bfb46': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/GIMP/2.10',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'printf '
                                                                                                "'(history\\n "
                                                                                                '("filters-vignette" '
                                                                                                "1)\\n)\\n' "
                                                                                                '> '
                                                                                                '/home/user/.config/GIMP/2.10/action-history',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    'b148e375-fe0b-4bec-90e7-38632b0d73c2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/GIMP/2.10',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/GIMP/2.10/gimprc'\n"
                                                                                                'lines = []\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'lines = '
                                                                                                'f.readlines()\n'
                                                                                                '\n'
                                                                                                '# Remove '
                                                                                                'existing '
                                                                                                'layer-new-name '
                                                                                                'line if '
                                                                                                'present\n'
                                                                                                'lines = [l '
                                                                                                'for l in '
                                                                                                'lines if '
                                                                                                'not '
                                                                                                "l.strip().startswith('(layer-new-name "
                                                                                                "')]\n"
                                                                                                '\n'
                                                                                                '# Add the '
                                                                                                'layer-new-name '
                                                                                                'setting\n'
                                                                                                "lines.append('(layer-new-name "
                                                                                                '"Square")\\n\')\n'
                                                                                                '\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'f.writelines(lines)\n'
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'd16c99dc-2a1e-46f2-b350-d97c86c85c15': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from PIL '
                                                                                                'import '
                                                                                                'Image\n'
                                                                                                '\n'
                                                                                                '# Load the '
                                                                                                'dog image '
                                                                                                '(the '
                                                                                                'reference '
                                                                                                'for '
                                                                                                'structure '
                                                                                                'comparison)\n'
                                                                                                'img = '
                                                                                                "Image.open('/home/user/Desktop/dog_with_background.png')\n"
                                                                                                'w, h = '
                                                                                                'img.size\n'
                                                                                                '\n'
                                                                                                '# Resize to '
                                                                                                'height=512, '
                                                                                                'maintaining '
                                                                                                'aspect '
                                                                                                'ratio\n'
                                                                                                'new_h = '
                                                                                                '512\n'
                                                                                                'new_w = '
                                                                                                'int(w * '
                                                                                                'new_h / h)\n'
                                                                                                'img_resized '
                                                                                                '= '
                                                                                                'img.resize((new_w, '
                                                                                                'new_h), '
                                                                                                'Image.LANCZOS)\n'
                                                                                                "img_resized.save('/home/user/Desktop/resized.png')\n"
                                                                                                "print(f'Resized "
                                                                                                'from '
                                                                                                '{w}x{h} to '
                                                                                                "{new_w}x{new_h}')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'd52d6308-ec58-42b7-a2c9-de80e4837b2b': {   'actions': [   {   'type': 'close_window',
                                                                   'parameters': {'window_name': 'GIMP'}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'kill -9 '
                                                                                                '$(pgrep -if '
                                                                                                'gimp) '
                                                                                                '2>/dev/null; '
                                                                                                'kill -9 '
                                                                                                '$(pgrep -if '
                                                                                                'script-fu) '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 3; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/.config/GIMP/2.10',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/GIMP/2.10/sessionrc'\n"
                                                                                                'lines = []\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with '
                                                                                                'open(path) '
                                                                                                'as f:\n'
                                                                                                '        '
                                                                                                'lines = '
                                                                                                'f.readlines()\n'
                                                                                                '\n'
                                                                                                '# Remove '
                                                                                                'existing '
                                                                                                'hide-docks '
                                                                                                'line\n'
                                                                                                'lines = [l '
                                                                                                'for l in '
                                                                                                'lines if '
                                                                                                "'hide-docks' "
                                                                                                'not in l]\n'
                                                                                                '\n'
                                                                                                '# Add '
                                                                                                'hide-docks '
                                                                                                'yes in GIMP '
                                                                                                'format\n'
                                                                                                "lines.append('(hide-docks "
                                                                                                "yes)\\n')\n"
                                                                                                '\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'f.writelines(lines)\n'
                                                                                                "print('Set "
                                                                                                'hide-docks '
                                                                                                "to yes')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'e2dd0213-26db-4349-abe5-d5667bfd725c': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import '
                                                                                                'struct\n'
                                                                                                'path = '
                                                                                                "'/home/user/Desktop/orange_background.xcf'\n"
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'rb') as "
                                                                                                'f:\n'
                                                                                                '    data = '
                                                                                                'f.read()\n'
                                                                                                'target_x = '
                                                                                                "struct.pack('>i', "
                                                                                                '976)\n'
                                                                                                'target_y = '
                                                                                                "struct.pack('>i', "
                                                                                                '702)\n'
                                                                                                'pos = 0\n'
                                                                                                'while '
                                                                                                'True:\n'
                                                                                                '    idx = '
                                                                                                'data.find(target_x, '
                                                                                                'pos)\n'
                                                                                                '    if idx '
                                                                                                '== -1:\n'
                                                                                                '        '
                                                                                                'break\n'
                                                                                                '    if idx '
                                                                                                '+ 8 <= '
                                                                                                'len(data) '
                                                                                                'and '
                                                                                                'data[idx+4:idx+8] '
                                                                                                '== '
                                                                                                'target_y:\n'
                                                                                                '        '
                                                                                                'data = '
                                                                                                'data[:idx] '
                                                                                                '+ '
                                                                                                "struct.pack('>i', "
                                                                                                '0) + '
                                                                                                'data[idx+4:]\n'
                                                                                                '        '
                                                                                                "print(f'Moved "
                                                                                                'text layer '
                                                                                                'from x=976 '
                                                                                                'to x=0 at '
                                                                                                'offset '
                                                                                                "{idx}')\n"
                                                                                                '        '
                                                                                                'break\n'
                                                                                                '    pos = '
                                                                                                'idx + 1\n'
                                                                                                'with '
                                                                                                'open(path, '
                                                                                                "'wb') as "
                                                                                                'f:\n'
                                                                                                '    '
                                                                                                'f.write(data)\n'
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'gimp',
                                                                                                    '/home/user/Desktop/orange_background.xcf']}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}}]},
    'f4aec372-4fb0-4df5-a52b-79e0e2a5d6ce': {   'actions': [   {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -f '
                                                                                                'gimp; sleep '
                                                                                                '3',
                                                                                     'shell': True}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'export '
                                                                                                'DISPLAY=:1 '
                                                                                                '&& timeout -s KILL 90 gimp -i '
                                                                                                "-b '(let* "
                                                                                                '((image '
                                                                                                '(car '
                                                                                                '(gimp-file-load '
                                                                                                'RUN-NONINTERACTIVE '
                                                                                                '"/home/user/Desktop/Triangle_On_The_Side.xcf" '
                                                                                                '"Triangle_On_The_Side.xcf"))) '
                                                                                                '(img-w (car '
                                                                                                '(gimp-image-width '
                                                                                                'image))) '
                                                                                                '(img-h (car '
                                                                                                '(gimp-image-height '
                                                                                                'image))) '
                                                                                                '(layers '
                                                                                                '(gimp-image-get-layers '
                                                                                                'image)) '
                                                                                                '(num-layers '
                                                                                                '(car '
                                                                                                'layers)) '
                                                                                                '(layer-array '
                                                                                                '(cadr '
                                                                                                'layers))) '
                                                                                                '(let loop '
                                                                                                '((i 0)) '
                                                                                                '(when (< i '
                                                                                                'num-layers) '
                                                                                                '(let* '
                                                                                                '((layer '
                                                                                                '(vector-ref '
                                                                                                'layer-array '
                                                                                                'i)) (name '
                                                                                                '(car '
                                                                                                '(gimp-item-get-name '
                                                                                                'layer))) '
                                                                                                '(lw (car '
                                                                                                '(gimp-drawable-width '
                                                                                                'layer))) '
                                                                                                '(lh (car '
                                                                                                '(gimp-drawable-height '
                                                                                                'layer)))) '
                                                                                                '(unless '
                                                                                                '(string=? '
                                                                                                'name '
                                                                                                '"Background") '
                                                                                                '(gimp-layer-set-offsets '
                                                                                                'layer '
                                                                                                '(quotient '
                                                                                                '(- img-w '
                                                                                                'lw) 2) '
                                                                                                '(quotient '
                                                                                                '(- img-h '
                                                                                                'lh) 2))) '
                                                                                                '(loop (+ i '
                                                                                                '1))))) '
                                                                                                '(gimp-image-flatten '
                                                                                                'image) '
                                                                                                '(file-png-save '
                                                                                                'RUN-NONINTERACTIVE '
                                                                                                'image (car '
                                                                                                '(gimp-image-get-active-drawable '
                                                                                                'image)) '
                                                                                                '"/home/user/Desktop/Triangle_In_The_Middle.png" '
                                                                                                '"Triangle_In_The_Middle.png" '
                                                                                                '0 9 1 1 1 1 '
                                                                                                '1) '
                                                                                                '(file-xcf-save '
                                                                                                'RUN-NONINTERACTIVE '
                                                                                                'image (car '
                                                                                                '(gimp-image-get-active-drawable '
                                                                                                'image)) '
                                                                                                '"/home/user/Desktop/Triangle_On_The_Side.xcf" '
                                                                                                '"Triangle_On_The_Side.xcf") '
                                                                                                '(gimp-quit '
                                                                                                "0))' 2>&1 | "
                                                                                                'tail -5; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'gimp',
                                                                                                    '/home/user/Desktop/Triangle_On_The_Side.xcf']}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}}]},
    'f723c744-e62c-4ae6-98d1-750d3cd7d79d': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'gimp '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from PIL '
                                                                                                'import '
                                                                                                'Image, '
                                                                                                'ImageEnhance\n'
                                                                                                '\n'
                                                                                                'img = '
                                                                                                "Image.open('/home/user/Desktop/berries.png')\n"
                                                                                                '# Increase '
                                                                                                'contrast\n'
                                                                                                'enhancer = '
                                                                                                'ImageEnhance.Contrast(img)\n'
                                                                                                'img_contrast '
                                                                                                '= '
                                                                                                'enhancer.enhance(1.5)  '
                                                                                                '# 1.5x '
                                                                                                'contrast\n'
                                                                                                "img_contrast.save('/home/user/Desktop/berries_contrast.png')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]}}
