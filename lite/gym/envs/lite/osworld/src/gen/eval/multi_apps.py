"""Hand-curated oracle metadata for multi_apps eval tasks.

Multi-application interop tasks.

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
- total entries:           92
- with actions:            92
- after_postconfig=True:   6
- block: exclude_reason:   2
- evaluator override:      2
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {
    # FEASIBLE (validated oracle passes): download the 7 gold reference PDFs to the Desktop output paths (compare_pdfs).
    '185f29bd-5da0-40a6-b69c-ba7f4e0324ef': {'actions': [{'type': 'execute', 'parameters': {'shell': True, 'command': 'mkdir -p /home/user/Desktop && python3 -c "import urllib.request as r; r.urlretrieve(\'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Alex%20Lee.pdf\', \'/home/user/Desktop/Alex Lee.pdf\')" && python3 -c "import urllib.request as r; r.urlretrieve(\'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/David%20Wilson.pdf\', \'/home/user/Desktop/David Wilson.pdf\')" && python3 -c "import urllib.request as r; r.urlretrieve(\'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Emily%20Johnson.pdf\', \'/home/user/Desktop/Emily Johnson.pdf\')" && python3 -c "import urllib.request as r; r.urlretrieve(\'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/John%20Doe.pdf\', \'/home/user/Desktop/John Doe.pdf\')" && python3 -c "import urllib.request as r; r.urlretrieve(\'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Linda%20Green.pdf\', \'/home/user/Desktop/Linda Green.pdf\')" && python3 -c "import urllib.request as r; r.urlretrieve(\'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Michael%20Brown.pdf\', \'/home/user/Desktop/Michael Brown.pdf\')" && python3 -c "import urllib.request as r; r.urlretrieve(\'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Sophia%20Carter.pdf\', \'/home/user/Desktop/Sophia Carter.pdf\')"'}}]},
   '00fa164e-2612-4439-992e-157d019a8436': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Documents/awesome-desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Documents/awesome-desktop/awe_desk_env.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/00fa164e-2612-4439-992e-157d019a8436/awe_desk_env_gt.docx'",
                                                                                     'shell': True}}]},
    '02ce9a50-7af2-47ed-8596-af0c230501f8': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import subprocess\n'
                                                                                                'from PIL import '
                                                                                                'Image, ImageDraw, '
                                                                                                'ImageFont\n'
                                                                                                'result = '
                                                                                                "subprocess.run(['ls', "
                                                                                                "'/home/user'], "
                                                                                                'capture_output=True, '
                                                                                                'text=True)\n'
                                                                                                'output = '
                                                                                                'result.stdout\n'
                                                                                                'img = '
                                                                                                "Image.new('RGB', "
                                                                                                '(1200, 800), '
                                                                                                'color=(0, 0, 0))\n'
                                                                                                'draw = '
                                                                                                'ImageDraw.Draw(img)\n'
                                                                                                'try:\n'
                                                                                                '    font = '
                                                                                                "ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', "
                                                                                                '32)\n'
                                                                                                'except Exception:\n'
                                                                                                '    try:\n'
                                                                                                '        font = '
                                                                                                "ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', "
                                                                                                '32)\n'
                                                                                                '    except Exception:\n'
                                                                                                '        font = '
                                                                                                'ImageFont.load_default()\n'
                                                                                                'draw.text((20, 20), '
                                                                                                "'user@desktop:~$ ls', "
                                                                                                'fill=(0, 255, 0), '
                                                                                                'font=font)\n'
                                                                                                'y = 70\n'
                                                                                                'for line in '
                                                                                                "output.split('\\n'):\n"
                                                                                                '    if line.strip():\n'
                                                                                                '        '
                                                                                                'draw.text((20, y), '
                                                                                                'line, fill=(255, 255, '
                                                                                                '255), font=font)\n'
                                                                                                '        y += 38\n'
                                                                                                "img.save('/home/user/Desktop/ls.png')\n"
                                                                                                'PYEOF\n'
                                                                                                'ls -la '
                                                                                                '/home/user/Desktop/ls.png',
                                                                                     'shell': True}}]},
    '09a37c51-e625-49f4-a514-20a773797a8a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/pic.jpg' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/09a37c51-e625-49f4-a514-20a773797a8a/pic.jpg'",
                                                                                     'shell': True}}]},
    '0e5303d4-8820-42f6-b18d-daf7e633de21': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "mkdir -p '/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/lecture_slides.zip' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/0e5303d4-8820-42f6-b18d-daf7e633de21/lecture_slides.zip'",
                                                                                     'shell': True}}]},
    '185f29bd-5da0-40a6-b69c-ba7f4e0324ef': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Alex "
                                                                                                "Lee.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Alex%20Lee.pdf'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/David "
                                                                                                "Wilson.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/David%20Wilson.pdf'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Emily "
                                                                                                "Johnson.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Emily%20Johnson.pdf'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/John "
                                                                                                "Doe.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/John%20Doe.pdf'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Linda "
                                                                                                "Green.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Linda%20Green.pdf'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Michael "
                                                                                                "Brown.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Michael%20Brown.pdf'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Sophia "
                                                                                                "Carter.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/185f29bd-5da0-40a6-b69c-ba7f4e0324ef/Sophia%20Carter.pdf'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '1f18aa87-af6f-41ef-9853-cdb8f32ebdea': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Answer.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/1f18aa87-af6f-41ef-9853-cdb8f32ebdea/Answer_Gold.docx'",
                                                                                     'shell': True}}]},
    '20236825-b5df-46e7-89bf-62e1d640a897': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/res.txt' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/20236825-b5df-46e7-89bf-62e1d640a897/res.txt'",
                                                                                     'shell': True}}]},
    '227d2f97-562b-4ccb-ae47-a5ec9e142fbb': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/image.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/227d2f97-562b-4ccb-ae47-a5ec9e142fbb/image.docx'",
                                                                                     'shell': True}}]},
    '236833a3-5704-47fc-888c-4f298f09f799': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/paper_reading_2024_03_01.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/236833a3-5704-47fc-888c-4f298f09f799/paper_reading_2024_03_01.docx'",
                                                                                     'shell': True}}]},
    '2373b66a-092d-44cb-bfd7-82e86e7a3b4d': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import datetime\n'
                                                                                                'lines = []\n'
                                                                                                'now = '
                                                                                                'datetime.datetime.now()\n'
                                                                                                'lines.append("Linux '
                                                                                                '5.15.0 (cua-vm)  " + '
                                                                                                'now.strftime("%m/%d/%Y") '
                                                                                                '+ "  _x86_64_  (4 '
                                                                                                'CPU)")\n'
                                                                                                'lines.append("")\n'
                                                                                                'hdr = "{:>12s}    '
                                                                                                'CPU     %user     '
                                                                                                '%nice   %system   '
                                                                                                '%iowait    %steal     '
                                                                                                '%idle".format(now.strftime("%I:%M:%S '
                                                                                                '%p"))\n'
                                                                                                'lines.append(hdr)\n'
                                                                                                'for i in range(30):\n'
                                                                                                '    t = (now + '
                                                                                                'datetime.timedelta(seconds=i+1)).strftime("%I:%M:%S '
                                                                                                '%p")\n'
                                                                                                '    '
                                                                                                'lines.append("{:>12s}    '
                                                                                                'all      1.00      '
                                                                                                '0.00      0.50      '
                                                                                                '0.00      0.00     '
                                                                                                '98.50".format(t))\n'
                                                                                                'with '
                                                                                                "open('/home/user/Desktop/System_Resources_Report.txt', "
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'f.write("\\n".join(lines) '
                                                                                                '+ "\\n")\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '26150609-0da3-4a7d-8868-0faf9c5f01bb': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'cat > '
                                                                                                '/home/user/Desktop/snake/food.py '
                                                                                                "<< 'PYFILE'\n"
                                                                                                '# food.py\n'
                                                                                                'import pygame\n'
                                                                                                'import random\n'
                                                                                                'from settings import '
                                                                                                '*\n'
                                                                                                '\n'
                                                                                                'class Food:\n'
                                                                                                '    def '
                                                                                                '__init__(self):\n'
                                                                                                '        self.position '
                                                                                                '= (random.randint(0, '
                                                                                                '(WIDTH - SNAKE_SIZE) '
                                                                                                '// SNAKE_SIZE) * '
                                                                                                'SNAKE_SIZE,\n'
                                                                                                '                         '
                                                                                                'random.randint(0, '
                                                                                                '(HEIGHT - SNAKE_SIZE) '
                                                                                                '// SNAKE_SIZE) * '
                                                                                                'SNAKE_SIZE)\n'
                                                                                                '        self.color = '
                                                                                                'RED\n'
                                                                                                '\n'
                                                                                                '    def draw(self, '
                                                                                                'surface):\n'
                                                                                                '        rect = '
                                                                                                'pygame.Rect((self.position[0], '
                                                                                                'self.position[1]), '
                                                                                                '(SNAKE_SIZE, '
                                                                                                'SNAKE_SIZE))\n'
                                                                                                '        '
                                                                                                'pygame.draw.rect(surface, '
                                                                                                'self.color, rect)\n'
                                                                                                '\n'
                                                                                                '    def '
                                                                                                'respawn(self):\n'
                                                                                                '        self.position '
                                                                                                '= (random.randint(0, '
                                                                                                '(WIDTH - SNAKE_SIZE) '
                                                                                                '// SNAKE_SIZE) * '
                                                                                                'SNAKE_SIZE,\n'
                                                                                                '                         '
                                                                                                'random.randint(0, '
                                                                                                '(HEIGHT - SNAKE_SIZE) '
                                                                                                '// SNAKE_SIZE) * '
                                                                                                'SNAKE_SIZE)\n'
                                                                                                'PYFILE',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cat > '
                                                                                                '/home/user/Desktop/snake/pygame.py '
                                                                                                "<< 'PYFILE'\n"
                                                                                                '"""Minimal pygame '
                                                                                                'stub for headless '
                                                                                                'test execution."""\n'
                                                                                                'K_UP = 1073741906\n'
                                                                                                'K_DOWN = 1073741905\n'
                                                                                                'K_LEFT = 1073741904\n'
                                                                                                'K_RIGHT = 1073741903\n'
                                                                                                'QUIT = 256\n'
                                                                                                'KEYDOWN = 768\n'
                                                                                                '\n'
                                                                                                'class Rect:\n'
                                                                                                '    def '
                                                                                                '__init__(self, pos, '
                                                                                                'size):\n'
                                                                                                '        self.x, '
                                                                                                'self.y = pos\n'
                                                                                                '        self.w, '
                                                                                                'self.h = size\n'
                                                                                                '\n'
                                                                                                'class _Draw:\n'
                                                                                                '    @staticmethod\n'
                                                                                                '    def rect(surface, '
                                                                                                'color, rect):\n'
                                                                                                '        pass\n'
                                                                                                '\n'
                                                                                                'draw = _Draw()\n'
                                                                                                '\n'
                                                                                                'def init():\n'
                                                                                                '    pass\n'
                                                                                                'PYFILE',
                                                                                     'shell': True}}],
                                                'evaluator': {   'postconfig': [],
                                                                 'func': 'check_python_file_by_test_suite',
                                                                 'result': {   'type': 'vm_file',
                                                                               'path': [   '/home/user/Desktop/snake/food.py',
                                                                                           '/home/user/Desktop/snake/main.py',
                                                                                           '/home/user/Desktop/snake/settings.py',
                                                                                           '/home/user/Desktop/snake/snake.py'],
                                                                               'dest': [   'food.py',
                                                                                           'main.py',
                                                                                           'settings.py',
                                                                                           'snake.py'],
                                                                               'multi': True},
                                                                 'expected': {   'type': 'cloud_file',
                                                                                 'path': 'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/26150609-0da3-4a7d-8868-0faf9c5f01bb/test.py',
                                                                                 'dest': 'test_suite.py'}}},
    '26660ad1-6ebb-4f59-8cba-a8432dfe8d38': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Test/Speed',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cat > '
                                                                                                '/home/user/Test/Speed/results.txt '
                                                                                                "<< 'EOF'\n"
                                                                                                'Ping 15ms\n'
                                                                                                'Download 95.5Mbps\n'
                                                                                                'Upload 45.2Mbps\n'
                                                                                                'EOF',
                                                                                     'shell': True}}]},
    '2b9493d7-49b8-493a-a71b-56cd1f4d6908': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "echo 'kill $(pgrep "
                                                                                                "soffice)' >> "
                                                                                                '/home/user/.bash_history',
                                                                                     'shell': True}}]},
    '2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop/students "
                                                                                                "work'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/students "
                                                                                                "work/case study.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e/case%20study%20gold.docx'",
                                                                                     'shell': True}}],
                                                'exclude_reason': 'upstream_generated_eval_bug',
                                                'evaluator': {   'postconfig': [   {   'type': 'activate_window',
                                                                                       'parameters': {   'window_name': 'case '
                                                                                                                        'study.docx '
                                                                                                                        '- '
                                                                                                                        'LibreOffice '
                                                                                                                        'Writer',
                                                                                                         'strict': True}},
                                                                                   {   'type': 'sleep',
                                                                                       'parameters': {'seconds': 0.5}},
                                                                                   {   'type': 'execute',
                                                                                       'parameters': {   'command': [   'python',
                                                                                                                        '-c',
                                                                                                                        'import '
                                                                                                                        'pyautogui; '
                                                                                                                        'import '
                                                                                                                        'time; '
                                                                                                                        "pyautogui.hotkey('ctrl', "
                                                                                                                        "'s'); "
                                                                                                                        'time.sleep(0.5); ']}}],
                                                                 'func': 'compare_references',
                                                                 'expected': {   'type': 'cloud_file',
                                                                                 'path': 'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e/case%20study%20gold.docx',
                                                                                 'dest': 'case study gold.docx'},
                                                                 'result': {   'type': 'vm_file',
                                                                               'path': '/home/user/Desktop/students '
                                                                                       'work/case study.docx',
                                                                               'dest': 'case study.docx'},
                                                                 'options': {   'content_only': True,
                                                                                'reference_base_result': 0.6}}},
    '2c9fc0de-3ee7-45e1-a5df-c86206ad78b5': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd '
                                                                                                '/home/user/projects/binder '
                                                                                                '&& git add -A && git '
                                                                                                "commit -m 'daily "
                                                                                                "update'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd '
                                                                                                '/home/user/projects/binder '
                                                                                                '&& git push origin '
                                                                                                'main',
                                                                                     'shell': True}}]},
    '2fe4b718-3bd7-46ec-bdce-b184f5653624': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/src_clip.gif' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/2fe4b718-3bd7-46ec-bdce-b184f5653624/src_clip.gif'",
                                                                                     'shell': True}}]},
    '337d318b-aa07-4f4f-b763-89d9a2dd013f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop/problematic'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/problematic/Invoice "
                                                                                                "# 243729.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/337d318b-aa07-4f4f-b763-89d9a2dd013f/Invoice%20%23%20243729.pdf'",
                                                                                     'shell': True}}]},
    '36037439-2044-4b50-b9d1-875b5a332143': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    'https://scholar.google.com/citations?hl=en&user=qRAQ5BsAAAAJ']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                  'exclude_reason': 'upstream_live_site_drift'},
    '3680a5ee-6870-426a-a997-eba929a0d25c': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'curl -sL -o '
                                                                                                '/home/user/Desktop/output.csv '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/3680a5ee-6870-426a-a997-eba929a0d25c/output.csv'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'export DISPLAY=:1 && '
                                                                                                'gnome-terminal -- '
                                                                                                'bash -c \'soffice '
                                                                                                '--calc '
                                                                                                '/home/user/Desktop/output.csv\' '
                                                                                                '&',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '3a93cae4-ad3e-403e-8c12-65303b271818': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Course "
                                                                                                "Timetable.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/3a93cae4-ad3e-403e-8c12-65303b271818/Course%20Timetable%20Gold.xlsx'",
                                                                                     'shell': True}}]},
    '3c8f201a-009d-4bbe-8b65-a6f8b35bb57f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                '/tmp/kingbird_orig.jpeg '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/3c8f201a-009d-4bbe-8b65-a6f8b35bb57f/kingbird.jpeg'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'from PIL import '
                                                                                                'Image\n'
                                                                                                'img = '
                                                                                                "Image.open('/tmp/kingbird_orig.jpeg')\n"
                                                                                                '# Resize to reduce '
                                                                                                'file size if needed\n'
                                                                                                'w, h = img.size\n'
                                                                                                'ratio = min(1.0, '
                                                                                                '(600000 / (w * h * '
                                                                                                '3)) ** 0.5)\n'
                                                                                                'if ratio < 1.0:\n'
                                                                                                '    new_w = int(w * '
                                                                                                'ratio)\n'
                                                                                                '    new_h = int(h * '
                                                                                                'ratio)\n'
                                                                                                '    img = '
                                                                                                'img.resize((new_w, '
                                                                                                'new_h), '
                                                                                                'Image.LANCZOS)\n'
                                                                                                "img.save('/home/user/Desktop/compressed.jpeg', "
                                                                                                "'JPEG', quality=60, "
                                                                                                'optimize=True)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '3e3fc409-bff3-4905-bf16-c968eee3f807': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/movies.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/3e3fc409-bff3-4905-bf16-c968eee3f807/gold_movies.xlsx'",
                                                                                     'shell': True}}]},
    '3f05f3b9-29ba-4b6b-95aa-2204697ffc06': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install mutagen '
                                                                                                '2>&1 | tail -1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import os, glob\n'
                                                                                                'from mutagen.id3 '
                                                                                                'import ID3, TIT2, '
                                                                                                'TPE1, '
                                                                                                'ID3NoHeaderError\n'
                                                                                                'from mutagen.mp3 '
                                                                                                'import MP3\n'
                                                                                                '\n'
                                                                                                'music_dir = '
                                                                                                "'/home/user/Music'\n"
                                                                                                'for mp3_file in '
                                                                                                'glob.glob(os.path.join(music_dir, '
                                                                                                "'*.mp3')):\n"
                                                                                                '    basename = '
                                                                                                'os.path.splitext(os.path.basename(mp3_file))[0]\n'
                                                                                                '    parts = '
                                                                                                "basename.split(' - ', "
                                                                                                '1)\n'
                                                                                                '    if len(parts) == '
                                                                                                '2:\n'
                                                                                                '        artist, title '
                                                                                                '= parts[0].strip(), '
                                                                                                'parts[1].strip()\n'
                                                                                                '    else:\n'
                                                                                                '        artist, title '
                                                                                                "= '', basename\n"
                                                                                                '\n'
                                                                                                '    # Create ID3 tag '
                                                                                                'from scratch if '
                                                                                                'needed\n'
                                                                                                '    try:\n'
                                                                                                '        tags = '
                                                                                                'ID3(mp3_file)\n'
                                                                                                '    except '
                                                                                                'ID3NoHeaderError:\n'
                                                                                                '        tags = ID3()\n'
                                                                                                '\n'
                                                                                                '    '
                                                                                                'tags.add(TPE1(encoding=3, '
                                                                                                'text=[artist]))\n'
                                                                                                '    '
                                                                                                'tags.add(TIT2(encoding=3, '
                                                                                                'text=[title]))\n'
                                                                                                '    '
                                                                                                'tags.save(mp3_file)\n'
                                                                                                "    print(f'Set "
                                                                                                '{basename}: '
                                                                                                'artist={artist}, '
                                                                                                "title={title}')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '415ef462-bed3-493a-ac36-ca8c6d23bf1b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Documents/Finance'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Documents/Finance/tally_book.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/415ef462-bed3-493a-ac36-ca8c6d23bf1b/tally_book_gt.xlsx'",
                                                                                     'shell': True}}]},
    '42d25c08-fb87-4927-8b65-93631280a26f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Documents/Novels/Pass "
                                                                                                "Through'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Documents/Novels/Pass "
                                                                                                'Through/Pass '
                                                                                                "Through.epub' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/42d25c08-fb87-4927-8b65-93631280a26f/Pass%20Through.epub'",
                                                                                     'shell': True}}]},
    '42f4d1c7-4521-4161-b646-0a8934e36081': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'code --no-sandbox '
                                                                                                '--install-extension '
                                                                                                'mattn.lisp --force '
                                                                                                '2>/dev/null || true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PY'\n"
                                                                                                'from PIL import '
                                                                                                'Image\n'
                                                                                                'img = '
                                                                                                "Image.open('/home/user/Desktop/character.png')\n"
                                                                                                'img_resized = '
                                                                                                'img.resize((128, '
                                                                                                '128), Image.LANCZOS)\n'
                                                                                                "img_resized.save('/home/user/Desktop/resized.png')\n"
                                                                                                'PY',
                                                                                     'shell': True}}]},
    '47f7c0ce-a5fb-4100-a5e6-65cd0e7429e5': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                '/tmp/frame_008.png '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/47f7c0ce-a5fb-4100-a5e6-65cd0e7429e5/landscape.png' "
                                                                                                '|| true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PY'\n"
                                                                                                'from pptx import '
                                                                                                'Presentation\n'
                                                                                                'from pptx.util import '
                                                                                                'Inches, Emu\n'
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'pptx_path = '
                                                                                                "'/home/user/Desktop/Robotic_Workshop_Infographics.pptx'\n"
                                                                                                'frame_path = '
                                                                                                "'/tmp/frame_008.png'\n"
                                                                                                '\n'
                                                                                                'if not '
                                                                                                'os.path.exists(frame_path):\n'
                                                                                                '    # Download gold '
                                                                                                'image as fallback\n'
                                                                                                '    import '
                                                                                                'urllib.request\n'
                                                                                                '    '
                                                                                                "urllib.request.urlretrieve('https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/47f7c0ce-a5fb-4100-a5e6-65cd0e7429e5/landscape.png', "
                                                                                                'frame_path)\n'
                                                                                                '\n'
                                                                                                'prs = '
                                                                                                'Presentation(pptx_path)\n'
                                                                                                'slide = '
                                                                                                'prs.slides[1]  # '
                                                                                                'second slide '
                                                                                                '(0-indexed)\n'
                                                                                                '\n'
                                                                                                '# Set background '
                                                                                                'image\n'
                                                                                                'from pptx.oxml.ns '
                                                                                                'import qn\n'
                                                                                                'import lxml.etree as '
                                                                                                'etree\n'
                                                                                                '\n'
                                                                                                '# Add image as '
                                                                                                'background\n'
                                                                                                'bg = '
                                                                                                'slide.background\n'
                                                                                                'fill = bg.fill\n'
                                                                                                'fill.background()\n'
                                                                                                '\n'
                                                                                                '# Use slide '
                                                                                                'relationship to add '
                                                                                                'image\n'
                                                                                                'from '
                                                                                                'pptx.opc.constants '
                                                                                                'import '
                                                                                                'RELATIONSHIP_TYPE as '
                                                                                                'RT\n'
                                                                                                'image_part, rId = '
                                                                                                'slide.part.get_or_add_image_part(frame_path)\n'
                                                                                                '\n'
                                                                                                '# Set the background '
                                                                                                'fill to use the '
                                                                                                'image\n'
                                                                                                'bgPr = bg._element\n'
                                                                                                'for child in '
                                                                                                'list(bgPr):\n'
                                                                                                '    '
                                                                                                'bgPr.remove(child)\n'
                                                                                                '\n'
                                                                                                '# Create proper '
                                                                                                'background XML\n'
                                                                                                "bg_xml = f'''<p:bgPr "
                                                                                                'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"\n'
                                                                                                '               '
                                                                                                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"\n'
                                                                                                '               '
                                                                                                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
                                                                                                '  <a:blipFill>\n'
                                                                                                '    <a:blip '
                                                                                                'r:embed="{rId}"/>\n'
                                                                                                '    <a:stretch>\n'
                                                                                                '      <a:fillRect/>\n'
                                                                                                '    </a:stretch>\n'
                                                                                                '  </a:blipFill>\n'
                                                                                                "</p:bgPr>'''\n"
                                                                                                '\n'
                                                                                                'bgPr_elem = '
                                                                                                'etree.fromstring(bg_xml)\n'
                                                                                                'bgPr.append(bgPr_elem)\n'
                                                                                                '\n'
                                                                                                'prs.save(pptx_path)\n'
                                                                                                "print('Done')\n"
                                                                                                'PY',
                                                                                     'shell': True}}]},
    '48c46dc7-fe04-4505-ade7-723cba1aa6f6': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -f chrome '
                                                                                                '2>/dev/null; pkill -f '
                                                                                                'gnome-terminal '
                                                                                                '2>/dev/null; pkill -f '
                                                                                                'nautilus 2>/dev/null; '
                                                                                                'sleep 1; true',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'bash',
                                                                                                    '-c',
                                                                                                    'nautilus '
                                                                                                    '/home/user/Documents/Projects/OSWorld '
                                                                                                    '&']}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'bash',
                                                                                                    '-c',
                                                                                                    'gnome-terminal '
                                                                                                    '--working-directory=/home/user/Documents/Projects/OSWorld '
                                                                                                    '-- bash -ic '
                                                                                                    "'printf "
                                                                                                    '"\\033]0;~/Documents/Projects/OSWorld\\007"; '
                                                                                                    "exec bash' &"]}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    'https://github.com',
                                                                                                    'https://docs.python.org/3/']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '48d05431-6cd5-4e76-82eb-12b60d823f7d': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'cat >> '
                                                                                                '/home/user/.bashrc << '
                                                                                                "'CONDAEOF'\n"
                                                                                                '\n'
                                                                                                '# >>> conda '
                                                                                                'initialize >>>\n'
                                                                                                '__conda_setup="$(\'/home/user/anaconda3/bin/conda\' '
                                                                                                "'shell.bash' 'hook' "
                                                                                                '2> /dev/null)"\n'
                                                                                                'if [ $? -eq 0 ]; '
                                                                                                'then\n'
                                                                                                '    eval '
                                                                                                '"$__conda_setup"\n'
                                                                                                'else\n'
                                                                                                '    if [ -f '
                                                                                                '"/home/user/anaconda3/etc/profile.d/conda.sh" '
                                                                                                ']; then\n'
                                                                                                '        . '
                                                                                                '"/home/user/anaconda3/etc/profile.d/conda.sh"\n'
                                                                                                '    fi\n'
                                                                                                'fi\n'
                                                                                                'unset __conda_setup\n'
                                                                                                '# <<< conda '
                                                                                                'initialize <<<\n'
                                                                                                'CONDAEOF',
                                                                                     'shell': True}}]},
    '4c26e3f3-3a14-4d86-b44a-d3cedebbb487': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install Pillow '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'from PIL import '
                                                                                                'Image, ImageEnhance\n'
                                                                                                'import '
                                                                                                'urllib.request\n'
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'url = '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/4c26e3f3-3a14-4d86-b44a-d3cedebbb487/back.png'\n"
                                                                                                'out_path = '
                                                                                                "'/home/user/Desktop/background.png'\n"
                                                                                                'tmp_path = '
                                                                                                "'/tmp/back_original.png'\n"
                                                                                                '\n'
                                                                                                "os.makedirs('/home/user/Desktop', "
                                                                                                'exist_ok=True)\n'
                                                                                                'urllib.request.urlretrieve(url, '
                                                                                                'tmp_path)\n'
                                                                                                'img = '
                                                                                                'Image.open(tmp_path)\n'
                                                                                                "print(f'Original: "
                                                                                                '{img.size} '
                                                                                                "{img.mode}')\n"
                                                                                                'enhancer = '
                                                                                                'ImageEnhance.Brightness(img)\n'
                                                                                                'bright_img = '
                                                                                                'enhancer.enhance(1.5)\n'
                                                                                                'bright_img.save(out_path, '
                                                                                                "'PNG')\n"
                                                                                                "print(f'Saved to "
                                                                                                "{out_path}')\n"
                                                                                                'v = '
                                                                                                'Image.open(out_path)\n'
                                                                                                "print(f'Verify: "
                                                                                                "{v.size} {v.mode}')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '510f64c8-9bcc-4be1-8d30-638705850618': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': [   '/bin/bash',
                                                                                                    '-c',
                                                                                                    "echo 'code "
                                                                                                    "~/Desktop/project' "
                                                                                                    '> '
                                                                                                    '/home/user/.bash_history '
                                                                                                    '&& chown '
                                                                                                    'user:user '
                                                                                                    '/home/user/.bash_history '
                                                                                                    '&& echo -n '
                                                                                                    "'project' > "
                                                                                                    '/home/user/OpenProject.txt '
                                                                                                    '&& chown '
                                                                                                    'user:user '
                                                                                                    '/home/user/OpenProject.txt']}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json\n'
                                                                                                'from pathlib import Path\n'
                                                                                                '\n'
                                                                                                "home = Path('/home/user')\n"
                                                                                                "code_user = home / '.config/Code/User'\n"
                                                                                                "folder_uri = 'file:///home/user/Desktop/project'\n"
                                                                                                '\n'
                                                                                                "ws_dir = code_user / 'workspaceStorage/oracle-open-project'\n"
                                                                                                'ws_dir.mkdir(parents=True, exist_ok=True)\n'
                                                                                                "(ws_dir / 'workspace.json').write_text(json.dumps({'folder': folder_uri}), encoding='utf-8')\n"
                                                                                                '\n'
                                                                                                "storage = code_user / 'globalStorage/storage.json'\n"
                                                                                                'storage.parent.mkdir(parents=True, exist_ok=True)\n'
                                                                                                'try:\n'
                                                                                                '    data = json.loads(storage.read_text(encoding=\'utf-8\'))\n'
                                                                                                'except Exception:\n'
                                                                                                '    data = {}\n'
                                                                                                "data.setdefault('windowsState', {})['lastActiveWindow'] = {'folder': folder_uri}\n"
                                                                                                "recent = data.setdefault('history', {}).setdefault('recentlyOpenedPathsList', {})\n"
                                                                                                "entries = recent.setdefault('entries', [])\n"
                                                                                                "entries[:] = [e for e in entries if e.get('folderUri') != folder_uri]\n"
                                                                                                "entries.insert(0, {'folderUri': folder_uri})\n"
                                                                                                "storage.write_text(json.dumps(data), encoding='utf-8')\n"
                                                                                                'PYEOF\n'
                                                                                                'chown -R user:user /home/user/.config/Code',
                                                                                     'shell': True}}]},
    '51f5801c-18b3-4f25-b0c3-02f85507a078': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/notes.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/51f5801c-18b3-4f25-b0c3-02f85507a078/notes_gold.docx'",
                                                                                     'shell': True}}]},
    '58565672-7bfe-48ab-b828-db349231de6b': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    'https://www.apple.com/',
                                                                                                    'https://scholar.google.com/',
                                                                                                    'https://www.amazon.com/']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '5990457f-2adb-467b-a4af-5c857c92d762': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install openpyxl '
                                                                                                '2>&1 | tail -1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PY'\n"
                                                                                                'import openpyxl\n'
                                                                                                'path = '
                                                                                                "'/home/user/Desktop/researchers.xlsx'\n"
                                                                                                'wb = '
                                                                                                'openpyxl.load_workbook(path)\n'
                                                                                                'ws = wb.active\n'
                                                                                                'next_row = ws.max_row '
                                                                                                '+ 1\n'
                                                                                                "data = ['Yann LeCun', "
                                                                                                "'345074', '147', "
                                                                                                "'372', 'Deep "
                                                                                                "learning', "
                                                                                                "'https://hal.science/hal-04206682/document']\n"
                                                                                                'for col_idx, val in '
                                                                                                'enumerate(data, 1):\n'
                                                                                                '    '
                                                                                                'ws.cell(row=next_row, '
                                                                                                'column=col_idx, '
                                                                                                'value=val)\n'
                                                                                                'wb.save(path)\n'
                                                                                                "print('Done')\n"
                                                                                                'PY',
                                                                                     'shell': True}}]},
    '5bc63fb9-276a-4439-a7c1-9dc76401737f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/gemini_results.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/5bc63fb9-276a-4439-a7c1-9dc76401737f/gemini_results_Gold.docx'",
                                                                                     'shell': True}}]},
    '5df7b33a-9f77-4101-823e-02f863e1c1ae': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop/book',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/book/book.zip' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/5df7b33a-9f77-4101-823e-02f863e1c1ae/book.zip'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '67890eb6-6ce5-4c00-9e3d-fb4972699b06': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/best_awards_acl.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/67890eb6-6ce5-4c00-9e3d-fb4972699b06/gold_best_awards_acl.xlsx'",
                                                                                     'shell': True}}]},
    '68a25bd4-59c7-4f4d-975e-da0c8509c848': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "mkdir -p '/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/paper01.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/68a25bd4-59c7-4f4d-975e-da0c8509c848/paper01.pdf'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/ans.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/68a25bd4-59c7-4f4d-975e-da0c8509c848/ans.docx'",
                                                                                     'shell': True}}]},
    # FEASIBLE but oracle left BLANK: the ML stack is absent on both the official VM
    # and our container (task IS to install it); the install-based oracle imports cleanly
    # standalone but doesn't reproduce in the validate harness's exec context — left
    # unverified (runnable, not excluded) rather than shipping a non-replaying oracle.
    '69acbb55-d945-4927-a87b-8480e1a5bb7e': {},
    '6f4073b8-d8ea-4ade-8a18-c5d1d5d5aa9a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PY'\n"
                                                                                                'import openpyxl\n'
                                                                                                'path = '
                                                                                                "'/home/user/Desktop/ConferenceCity.xlsx'\n"
                                                                                                'wb = '
                                                                                                'openpyxl.load_workbook(path)\n'
                                                                                                'ws = wb.active\n'
                                                                                                '\n'
                                                                                                '# The expected cities '
                                                                                                'in order (from '
                                                                                                'evaluator)\n'
                                                                                                'cities = [\n'
                                                                                                '    "Scottsdale", '
                                                                                                '"Atlanta", "Lake '
                                                                                                'Tahoe", "Banff", '
                                                                                                '"Beijing", '
                                                                                                '"Montreal", "San '
                                                                                                'Diego",\n'
                                                                                                '    "Lille", '
                                                                                                '"Montreal", "San '
                                                                                                'Juan", "New York", '
                                                                                                '"Barcelona", '
                                                                                                '"Toulon", "Sydney",\n'
                                                                                                '    "Long Beach", '
                                                                                                '"Vancouver", '
                                                                                                '"Stockholm", '
                                                                                                '"Montreal", "New '
                                                                                                'Orleans", "Long '
                                                                                                'Beach", "Vancouver"\n'
                                                                                                ']\n'
                                                                                                '\n'
                                                                                                '# Find the column for '
                                                                                                'cities (likely column '
                                                                                                'C or the Location '
                                                                                                'column)\n'
                                                                                                '# Scan for empty '
                                                                                                'cells that need '
                                                                                                'filling\n'
                                                                                                'row = 2  # start from '
                                                                                                'row 2 (skip header)\n'
                                                                                                'city_idx = 0\n'
                                                                                                'while row <= '
                                                                                                'ws.max_row and '
                                                                                                'city_idx < '
                                                                                                'len(cities):\n'
                                                                                                '    # Check if the '
                                                                                                'location cell is '
                                                                                                'empty\n'
                                                                                                '    loc_cell = '
                                                                                                'ws.cell(row=row, '
                                                                                                'column=3)  # column '
                                                                                                'C\n'
                                                                                                '    if loc_cell.value '
                                                                                                'is None or '
                                                                                                'str(loc_cell.value).strip() '
                                                                                                "== '':\n"
                                                                                                '        '
                                                                                                'loc_cell.value = '
                                                                                                'cities[city_idx]\n'
                                                                                                '    city_idx += 1\n'
                                                                                                '    row += 1\n'
                                                                                                '\n'
                                                                                                'wb.save(path)\n'
                                                                                                "print('Done')\n"
                                                                                                'PY',
                                                                                     'shell': True}}]},
    '716a6079-22da-47f1-ba73-c9d58f986a38': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo -n '
                                                                                                "'/home/user/Data3/List3/secret.docx' "
                                                                                                '| DISPLAY=:1 xclip '
                                                                                                '-selection clipboard '
                                                                                                '-i && sleep 1',
                                                                                     'shell': True}}]},
    '74d5859f-ed66-4d3e-aa0e-93d7a592ce41': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Projects/happy-extension/browserAction '
                                                                                                '/home/user/Projects/happy-extension/icons',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json\n'
                                                                                                'manifest = {\n'
                                                                                                '    '
                                                                                                '"manifest_version": '
                                                                                                '2,\n'
                                                                                                '    "name": '
                                                                                                '"happy-extension",\n'
                                                                                                '    "version": '
                                                                                                '"0.0.1",\n'
                                                                                                '    "description": '
                                                                                                '"",\n'
                                                                                                '    "background": '
                                                                                                '{"scripts": '
                                                                                                '["background_script.js"]},\n'
                                                                                                '    "browser_action": '
                                                                                                '{\n'
                                                                                                '        '
                                                                                                '"default_icon": '
                                                                                                '{"64": '
                                                                                                '"icons/icon.png"},\n'
                                                                                                '        '
                                                                                                '"default_popup": '
                                                                                                '"browserAction/index.html",\n'
                                                                                                '        '
                                                                                                '"default_title": '
                                                                                                '"happy-extension"\n'
                                                                                                '    }\n'
                                                                                                '}\n'
                                                                                                'with '
                                                                                                "open('/home/user/Projects/happy-extension/manifest.json', "
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'json.dump(manifest, '
                                                                                                'f, indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'touch '
                                                                                                '/home/user/Projects/happy-extension/background_script.js',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'touch '
                                                                                                '/home/user/Projects/happy-extension/browserAction/index.html',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'touch '
                                                                                                '/home/user/Projects/happy-extension/browserAction/style.css',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'touch '
                                                                                                '/home/user/Projects/happy-extension/browserAction/script.js',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Projects/happy-extension/background_script.js' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/74d5859f-ed66-4d3e-aa0e-93d7a592ce41/file_1t5Llhn6seDUXVs-eILu6CjwFEQL9Z5Qm.bin'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Projects/happy-extension/browserAction/index.html' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/74d5859f-ed66-4d3e-aa0e-93d7a592ce41/index.html'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Projects/happy-extension/browserAction/style.css' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/74d5859f-ed66-4d3e-aa0e-93d7a592ce41/style.css'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Projects/happy-extension/browserAction/script.js' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/74d5859f-ed66-4d3e-aa0e-93d7a592ce41/file_14YYnhCfRtHQNk8M4fBPaUQeteoFMGBsA.bin'",
                                                                                     'shell': True}}]},
    '778efd0a-153f-4842-9214-f05fc176b877': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                '/home/user/Desktop/planet.wav '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/778efd0a-153f-4842-9214-f05fc176b877/planet.wav'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PY'\n"
                                                                                                'import zipfile\n'
                                                                                                'import shutil\n'
                                                                                                'import os\n'
                                                                                                'import '
                                                                                                'xml.etree.ElementTree '
                                                                                                'as ET\n'
                                                                                                '\n'
                                                                                                'pptx_path = '
                                                                                                "'/home/user/Desktop/Minimalist_Business_Slides.pptx'\n"
                                                                                                'wav_path = '
                                                                                                "'/home/user/Desktop/planet.wav'\n"
                                                                                                'tmp_dir = '
                                                                                                "'/tmp/pptx_edit'\n"
                                                                                                '\n'
                                                                                                '# Extract pptx\n'
                                                                                                'if '
                                                                                                'os.path.exists(tmp_dir):\n'
                                                                                                '    '
                                                                                                'shutil.rmtree(tmp_dir)\n'
                                                                                                'os.makedirs(tmp_dir)\n'
                                                                                                'with '
                                                                                                'zipfile.ZipFile(pptx_path, '
                                                                                                "'r') as z:\n"
                                                                                                '    '
                                                                                                'z.extractall(tmp_dir)\n'
                                                                                                '\n'
                                                                                                '# Copy audio file '
                                                                                                'into pptx media '
                                                                                                'folder\n'
                                                                                                'media_dir = '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'ppt', 'media')\n"
                                                                                                'os.makedirs(media_dir, '
                                                                                                'exist_ok=True)\n'
                                                                                                'shutil.copy2(wav_path, '
                                                                                                'os.path.join(media_dir, '
                                                                                                "'planet.wav'))\n"
                                                                                                '\n'
                                                                                                '# Add relationship to '
                                                                                                'slide1\n'
                                                                                                'rels_path = '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'ppt', 'slides', "
                                                                                                "'_rels', "
                                                                                                "'slide1.xml.rels')\n"
                                                                                                "ET.register_namespace('', "
                                                                                                "'http://schemas.openxmlformats.org/package/2006/relationships')\n"
                                                                                                'tree = '
                                                                                                'ET.parse(rels_path)\n'
                                                                                                'root = '
                                                                                                'tree.getroot()\n'
                                                                                                '\n'
                                                                                                '# Find max rId\n'
                                                                                                'max_id = 0\n'
                                                                                                'for rel in root:\n'
                                                                                                '    rid = '
                                                                                                "rel.get('Id', "
                                                                                                "'rId0')\n"
                                                                                                '    try:\n'
                                                                                                '        num = '
                                                                                                "int(rid.replace('rId', "
                                                                                                "''))\n"
                                                                                                '        if num > '
                                                                                                'max_id:\n'
                                                                                                '            max_id = '
                                                                                                'num\n'
                                                                                                '    except:\n'
                                                                                                '        pass\n'
                                                                                                '\n'
                                                                                                'new_rid = '
                                                                                                "f'rId{max_id + 1}'\n"
                                                                                                'audio_rel = '
                                                                                                'ET.SubElement(root, '
                                                                                                "'Relationship')\n"
                                                                                                "audio_rel.set('Id', "
                                                                                                'new_rid)\n'
                                                                                                "audio_rel.set('Type', "
                                                                                                "'http://schemas.openxmlformats.org/officeDocument/2006/relationships/audio')\n"
                                                                                                "audio_rel.set('Target', "
                                                                                                "'../media/planet.wav')\n"
                                                                                                '\n'
                                                                                                'tree.write(rels_path, '
                                                                                                'xml_declaration=True, '
                                                                                                "encoding='UTF-8')\n"
                                                                                                '\n'
                                                                                                '# Add audio element '
                                                                                                'to slide1.xml\n'
                                                                                                'slide_path = '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'ppt', 'slides', "
                                                                                                "'slide1.xml')\n"
                                                                                                'NS = {\n'
                                                                                                "    'a': "
                                                                                                "'http://schemas.openxmlformats.org/drawingml/2006/main',\n"
                                                                                                "    'r': "
                                                                                                "'http://schemas.openxmlformats.org/officeDocument/2006/relationships',\n"
                                                                                                "    'p': "
                                                                                                "'http://schemas.openxmlformats.org/presentationml/2006/main',\n"
                                                                                                '}\n'
                                                                                                'for prefix, uri in '
                                                                                                'NS.items():\n'
                                                                                                '    '
                                                                                                'ET.register_namespace(prefix, '
                                                                                                'uri)\n'
                                                                                                '\n'
                                                                                                'slide_tree = '
                                                                                                'ET.parse(slide_path)\n'
                                                                                                'slide_root = '
                                                                                                'slide_tree.getroot()\n'
                                                                                                '\n'
                                                                                                '# Find or create '
                                                                                                'spTree\n'
                                                                                                'sp_tree = '
                                                                                                "slide_root.find('.//p:cSld/p:spTree', "
                                                                                                'NS)\n'
                                                                                                'if sp_tree is None:\n'
                                                                                                '    sp_tree = '
                                                                                                "slide_root.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}cSld/{http://schemas.openxmlformats.org/presentationml/2006/main}spTree')\n"
                                                                                                '\n'
                                                                                                'if sp_tree is not '
                                                                                                'None:\n'
                                                                                                '    # Add a pic '
                                                                                                'element for audio\n'
                                                                                                '    pic = '
                                                                                                'ET.SubElement(sp_tree, '
                                                                                                "'{http://schemas.openxmlformats.org/presentationml/2006/main}pic')\n"
                                                                                                '    nvPicPr = '
                                                                                                'ET.SubElement(pic, '
                                                                                                "'{http://schemas.openxmlformats.org/presentationml/2006/main}nvPicPr')\n"
                                                                                                '    cNvPr = '
                                                                                                'ET.SubElement(nvPicPr, '
                                                                                                "'{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr')\n"
                                                                                                "    cNvPr.set('id', "
                                                                                                "'99999')\n"
                                                                                                "    cNvPr.set('name', "
                                                                                                "'Audio')\n"
                                                                                                '    # Add audio link\n'
                                                                                                '    aHlinkClick = '
                                                                                                'ET.SubElement(cNvPr, '
                                                                                                "'{http://schemas.openxmlformats.org/drawingml/2006/main}hlinkClick')\n"
                                                                                                '    '
                                                                                                "aHlinkClick.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id', "
                                                                                                "'')\n"
                                                                                                '    '
                                                                                                "aHlinkClick.set('action', "
                                                                                                "'ppaction://media')\n"
                                                                                                '\n'
                                                                                                'slide_tree.write(slide_path, '
                                                                                                'xml_declaration=True, '
                                                                                                "encoding='UTF-8')\n"
                                                                                                '\n'
                                                                                                '# Add content type '
                                                                                                'for wav\n'
                                                                                                'ct_path = '
                                                                                                'os.path.join(tmp_dir, '
                                                                                                "'[Content_Types].xml')\n"
                                                                                                'ct_tree = '
                                                                                                'ET.parse(ct_path)\n'
                                                                                                'ct_root = '
                                                                                                'ct_tree.getroot()\n'
                                                                                                'ct_ns = '
                                                                                                "'http://schemas.openxmlformats.org/package/2006/content-types'\n"
                                                                                                "ET.register_namespace('', "
                                                                                                'ct_ns)\n'
                                                                                                '# Check if wav '
                                                                                                'extension already '
                                                                                                'registered\n'
                                                                                                'has_wav = False\n'
                                                                                                'for elem in ct_root:\n'
                                                                                                '    if '
                                                                                                "elem.get('Extension', "
                                                                                                "'') == 'wav':\n"
                                                                                                '        has_wav = '
                                                                                                'True\n'
                                                                                                '        break\n'
                                                                                                'if not has_wav:\n'
                                                                                                '    default = '
                                                                                                'ET.SubElement(ct_root, '
                                                                                                "'Default')\n"
                                                                                                '    '
                                                                                                "default.set('Extension', "
                                                                                                "'wav')\n"
                                                                                                '    '
                                                                                                "default.set('ContentType', "
                                                                                                "'audio/wav')\n"
                                                                                                '    '
                                                                                                'ct_tree.write(ct_path, '
                                                                                                'xml_declaration=True, '
                                                                                                "encoding='UTF-8')\n"
                                                                                                '\n'
                                                                                                '# Repack pptx\n'
                                                                                                'os.remove(pptx_path)\n'
                                                                                                'with '
                                                                                                'zipfile.ZipFile(pptx_path, '
                                                                                                "'w', "
                                                                                                'zipfile.ZIP_DEFLATED) '
                                                                                                'as z:\n'
                                                                                                '    for root_dir, '
                                                                                                'dirs, files in '
                                                                                                'os.walk(tmp_dir):\n'
                                                                                                '        for f in '
                                                                                                'files:\n'
                                                                                                '            full_path '
                                                                                                '= '
                                                                                                'os.path.join(root_dir, '
                                                                                                'f)\n'
                                                                                                '            arc_name '
                                                                                                '= '
                                                                                                'os.path.relpath(full_path, '
                                                                                                'tmp_dir)\n'
                                                                                                '            '
                                                                                                'z.write(full_path, '
                                                                                                'arc_name)\n'
                                                                                                '\n'
                                                                                                "print('Done: audio "
                                                                                                'embedded in slide '
                                                                                                "1')\n"
                                                                                                'PY',
                                                                                     'shell': True}}]},
    '788b3701-3ec9-4b67-b679-418bfa726c22': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Documents/Novels/4th "
                                                                                                "Year in Tsinghua'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Documents/Novels/4th "
                                                                                                'Year in '
                                                                                                'Tsinghua/Early '
                                                                                                "Buildings.tex' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/788b3701-3ec9-4b67-b679-418bfa726c22/%C3%A6%C2%97%C2%A9%C3%A6%C2%9C%C2%9F%C3%A5%C2%BB%C2%BA%C3%A7%C2%AD%C2%91%C3%A7%C2%BE%C2%A4.tex'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True,
                                                'evaluator': {   'func': 'diff_text_file',
                                                                 'result': {   'type': 'vm_file',
                                                                               'path': '/home/user/Documents/Novels/4th '
                                                                                       'Year in Tsinghua/Early '
                                                                                       'Buildings.tex',
                                                                               'dest': 'download.tex'},
                                                                 'expected': {   'type': 'cloud_file',
                                                                                 'path': 'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/788b3701-3ec9-4b67-b679-418bfa726c22/%C3%A6%C2%97%C2%A9%C3%A6%C2%9C%C2%9F%C3%A5%C2%BB%C2%BA%C3%A7%C2%AD%C2%91%C3%A7%C2%BE%C2%A4.tex',
                                                                                 'dest': 'real.tex'}}},
    '7e287123-70ca-47b9-8521-47db09b69b14': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop && '
                                                                                                'wget -q -O '
                                                                                                "'/home/user/Desktop/GRF-p5y.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/7e287123-70ca-47b9-8521-47db09b69b14/GRF-p5y.bak.xlsx'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -f '
                                                                                                "'/home/user/Desktop/GRF-p5y-Sheet1.csv'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '7f35355e-02a6-45b5-b140-f0be698bcf85': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/result.txt' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/7f35355e-02a6-45b5-b140-f0be698bcf85/result_gold.txt'",
                                                                                     'shell': True}}]},
    '7ff48d5b-2df2-49da-b500-a5150ffc7f18': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': [   '/bin/bash',
                                                                                                    '-c',
                                                                                                    'python3 -c "from '
                                                                                                    'docx import '
                                                                                                    'Document; '
                                                                                                    'd=Document(); '
                                                                                                    "d.add_paragraph('深圳市福田区益田路5055号信息枢纽大厦西门一楼'); "
                                                                                                    "d.add_paragraph('深圳市福田区正义街1号'); "
                                                                                                    "d.add_paragraph('深圳市福田区振兴路108号'); "
                                                                                                    'd.save(\'/home/user/Desktop/AllLocations.docx\')" '
                                                                                                    '&& chown '
                                                                                                    'user:user '
                                                                                                    '/home/user/Desktop/AllLocations.docx']}}],
                                                'after_postconfig': True},
    '81c425f5-78f3-4771-afd6-3d2973825947': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/price.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/81c425f5-78f3-4771-afd6-3d2973825947/price.docx'",
                                                                                     'shell': True}}]},
    '82e3c869-49f6-4305-a7ce-f3e64a0618e7': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -rf /tmp/presenter '
                                                                                                '/tmp/presenter.zip && '
                                                                                                'mkdir -p '
                                                                                                '/tmp/presenter && cd '
                                                                                                '/tmp/presenter && for '
                                                                                                'f in DSC00657 '
                                                                                                'DSC00574 DSC00554 '
                                                                                                'DSC00495; do wget -q '
                                                                                                '-O $f.jpg '
                                                                                                'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/82e3c869-49f6-4305-a7ce-f3e64a0618e7/$f.jpg; '
                                                                                                'done',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop && '
                                                                                                'cd /tmp && rm -f '
                                                                                                '/home/user/Desktop/presenter.zip '
                                                                                                '&& zip -qr '
                                                                                                '/home/user/Desktop/presenter.zip '
                                                                                                'presenter/',
                                                                                     'shell': True}}]},
    '869de13e-bef9-4b91-ba51-f6708c40b096': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo user | sudo -S '
                                                                                                'bash -c "apt-get '
                                                                                                'update -qq && '
                                                                                                'DEBIAN_FRONTEND=noninteractive '
                                                                                                'apt-get install -y '
                                                                                                '-qq locales && sed -i '
                                                                                                "'s/# "
                                                                                                "en_US.UTF-8/en_US.UTF-8/' "
                                                                                                '/etc/locale.gen && '
                                                                                                'locale-gen '
                                                                                                'en_US.UTF-8 && sed -i '
                                                                                                "'/^exec python3/i "
                                                                                                'export '
                                                                                                'LC_ALL=en_US.UTF-8\\\\nexport '
                                                                                                "LANG=en_US.UTF-8' "
                                                                                                '/usr/local/bin/start-osworld-server.sh" '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Desktop/Paper_reading '
                                                                                                '/home/user/Desktop/Projects '
                                                                                                '/home/user/Desktop/Miscellaneous',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user/Desktop '
                                                                                                '&& mv -f '
                                                                                                '1706.03762.pdf '
                                                                                                '1802.05365.pdf '
                                                                                                '1909.10351.pdf '
                                                                                                'paper01.pdf '
                                                                                                'Paper_reading/ '
                                                                                                '2>/dev/null; mv -f '
                                                                                                "'GLUE: A MULTI-TASK "
                                                                                                'BENCHMARK AND '
                                                                                                "ANALYSIS.pdf' "
                                                                                                'Paper_reading/ '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user/Desktop '
                                                                                                '&& mv -f '
                                                                                                '2-if-for-array '
                                                                                                'assign1-data_python3 '
                                                                                                'Projects/ '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user/Desktop '
                                                                                                '&& mv -f '
                                                                                                "'07-cluster-kMean "
                                                                                                "(1).ppt' "
                                                                                                '2023_validation_7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx '
                                                                                                'assignment_mark_frontpage.docx '
                                                                                                'cco-return-to-school-survey-underlying-data-tables.xlsx '
                                                                                                'DOC_2480903712718068684.pdf '
                                                                                                "'Family Status "
                                                                                                'Equality-Eng (Aug '
                                                                                                "2021).pdf' "
                                                                                                'IA_Format.docx '
                                                                                                'Miscellaneous/ '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}}]},
    '873cafdd-a581-47f6-8b33-b9696ddb7b05': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome '
                                                                                                '2>/dev/null; sleep 3; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                '\n'
                                                                                                'extensions = {\n'
                                                                                                '    "ext_zoom": '
                                                                                                '{"name": "Zoom for '
                                                                                                'Google Chrome"},\n'
                                                                                                '    "ext_speechify": '
                                                                                                '{"name": "Speechify '
                                                                                                '\\u2014 Voice AI '
                                                                                                'Assistant"},\n'
                                                                                                '    "ext_react": '
                                                                                                '{"name": "React '
                                                                                                'Developer Tools"},\n'
                                                                                                '    "ext_momentum": '
                                                                                                '{"name": '
                                                                                                '"Momentum"},\n'
                                                                                                '    "ext_translate": '
                                                                                                '{"name": "Google '
                                                                                                'Translate"}\n'
                                                                                                '}\n'
                                                                                                '\n'
                                                                                                'for prefs_dir in '
                                                                                                "['/home/user/chrome-data/Default', "
                                                                                                "'/home/user/.config/google-chrome/Default']:\n"
                                                                                                '    '
                                                                                                'os.makedirs(prefs_dir, '
                                                                                                'exist_ok=True)\n'
                                                                                                '    path = '
                                                                                                'os.path.join(prefs_dir, '
                                                                                                "'Preferences')\n"
                                                                                                '    prefs = {}\n'
                                                                                                '    if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '        try: prefs = '
                                                                                                'json.load(open(path))\n'
                                                                                                '        except: prefs '
                                                                                                '= {}\n'
                                                                                                '    settings = '
                                                                                                "prefs.setdefault('extensions', "
                                                                                                "{}).setdefault('settings', "
                                                                                                '{})\n'
                                                                                                '    for ext_id, info '
                                                                                                'in '
                                                                                                'extensions.items():\n'
                                                                                                '        '
                                                                                                'settings[ext_id] = {\n'
                                                                                                '            '
                                                                                                "'active_permissions': "
                                                                                                "{'api': [], "
                                                                                                "'explicit_host': [], "
                                                                                                "'manifest_permissions': "
                                                                                                '[], '
                                                                                                "'scriptable_host': "
                                                                                                '[]},\n'
                                                                                                '            '
                                                                                                "'creation_flags': 1, "
                                                                                                "'from_webstore': "
                                                                                                'True,\n'
                                                                                                '            '
                                                                                                "'granted_permissions': "
                                                                                                "{'api': [], "
                                                                                                "'explicit_host': [], "
                                                                                                "'manifest_permissions': "
                                                                                                '[], '
                                                                                                "'scriptable_host': "
                                                                                                '[]},\n'
                                                                                                '            '
                                                                                                "'install_time': "
                                                                                                "'13349226702110891', "
                                                                                                "'location': 1,\n"
                                                                                                '            '
                                                                                                "'manifest': {'name': "
                                                                                                "info['name'], "
                                                                                                "'version': '1.0', "
                                                                                                "'manifest_version': "
                                                                                                '3},\n'
                                                                                                "            'path': "
                                                                                                "ext_id, 'state': 1, "
                                                                                                "'was_installed_by_default': "
                                                                                                'False,\n'
                                                                                                '        }\n'
                                                                                                '    with open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '        '
                                                                                                'json.dump(prefs, f)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '881deb30-9549-4583-a841-8270c65f2a17': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Documents/Fundings '
                                                                                                '&& wget -q -O '
                                                                                                "'/home/user/Documents/Fundings/supported_rate.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/881deb30-9549-4583-a841-8270c65f2a17/supported_rate_gt.xlsx'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -f '
                                                                                                "'/home/user/Documents/Fundings/supported_rate-Sheet1.csv'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '8df7e444-8e06-4f93-8a1a-c5c974269d82': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "mkdir -p '/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/essay_submission.zip' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/8df7e444-8e06-4f93-8a1a-c5c974269d82/Recruitment_and_retention_of_health_professionals_across_Europe.zip'",
                                                                                     'shell': True}}]},
    '8e116af7-7db7-4e35-a68b-b0939c066c78': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/my_bookkeeping.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/8e116af7-7db7-4e35-a68b-b0939c066c78/my_bookkeeping%20Gold.xlsx'",
                                                                                     'shell': True}}]},
    '91190194-f406-4cd6-b3f9-c43fac942b22': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/cropped.png' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/91190194-f406-4cd6-b3f9-c43fac942b22/cropped_gold.png'",
                                                                                     'shell': True}}]},
    '9219480b-3aed-47fc-8bac-d2cffc5849f7': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'cat > '
                                                                                                '/home/user/Desktop/tetris/tetris.py '
                                                                                                "<< 'PYFILE'\n"
                                                                                                '# tetris.py\n'
                                                                                                'from block import '
                                                                                                'Block, shapes\n'
                                                                                                'import random\n'
                                                                                                '\n'
                                                                                                '\n'
                                                                                                'class Tetris:\n'
                                                                                                '    def '
                                                                                                '__init__(self, '
                                                                                                'height, width):\n'
                                                                                                '        self.height = '
                                                                                                'height\n'
                                                                                                '        self.width = '
                                                                                                'width\n'
                                                                                                '        self.board = '
                                                                                                '[[0] * width for _ in '
                                                                                                'range(height)]\n'
                                                                                                '        self.score = '
                                                                                                '0\n'
                                                                                                '        self.state = '
                                                                                                '"start"\n'
                                                                                                '        self.block = '
                                                                                                'None\n'
                                                                                                '        '
                                                                                                'self.next_block = '
                                                                                                'Block(random.choice(shapes))\n'
                                                                                                '\n'
                                                                                                '    def '
                                                                                                'new_block(self):\n'
                                                                                                '        self.block = '
                                                                                                'self.next_block\n'
                                                                                                '        '
                                                                                                'self.next_block = '
                                                                                                'Block(random.choice(shapes))\n'
                                                                                                '        self.block.x '
                                                                                                '= int(self.width / 2) '
                                                                                                '- '
                                                                                                'int(len(self.block.shape[0]) '
                                                                                                '/ 2)\n'
                                                                                                '        self.block.y '
                                                                                                '= 0\n'
                                                                                                '        if '
                                                                                                'self.intersect():\n'
                                                                                                '            '
                                                                                                'self.state = '
                                                                                                '"gameover"\n'
                                                                                                '\n'
                                                                                                '    def '
                                                                                                'intersect(self):\n'
                                                                                                '        for i in '
                                                                                                'range(len(self.block.shape)):\n'
                                                                                                '            for j in '
                                                                                                'range(len(self.block.shape[i])):\n'
                                                                                                '                if '
                                                                                                'self.block.shape[i][j] '
                                                                                                '!= 0:\n'
                                                                                                '                    '
                                                                                                'if i + self.block.y > '
                                                                                                'self.height - 1 '
                                                                                                'or                             '
                                                                                                'j + self.block.x > '
                                                                                                'self.width - 1 '
                                                                                                'or                             '
                                                                                                'j + self.block.x < 0 '
                                                                                                'or                             '
                                                                                                'self.board[i + '
                                                                                                'self.block.y][j + '
                                                                                                'self.block.x] != 0:\n'
                                                                                                '                        '
                                                                                                'return True\n'
                                                                                                '        return False\n'
                                                                                                '\n'
                                                                                                '    def '
                                                                                                'freeze(self):\n'
                                                                                                '        for i in '
                                                                                                'range(len(self.block.shape)):\n'
                                                                                                '            for j in '
                                                                                                'range(len(self.block.shape[i])):\n'
                                                                                                '                if '
                                                                                                'self.block.shape[i][j] '
                                                                                                '!= 0:\n'
                                                                                                '                    '
                                                                                                'self.board[i + '
                                                                                                'self.block.y][j + '
                                                                                                'self.block.x] = '
                                                                                                'self.block.shape[i][j]\n'
                                                                                                '        '
                                                                                                'self.break_lines()\n'
                                                                                                '        '
                                                                                                'self.new_block()\n'
                                                                                                '\n'
                                                                                                '    def '
                                                                                                'break_lines(self):\n'
                                                                                                '        lines = 0\n'
                                                                                                '        for i in '
                                                                                                'range(1, '
                                                                                                'self.height):\n'
                                                                                                '            zeros = '
                                                                                                '0\n'
                                                                                                '            for j in '
                                                                                                'range(self.width):\n'
                                                                                                '                if '
                                                                                                'self.board[i][j] == '
                                                                                                '0:\n'
                                                                                                '                    '
                                                                                                'zeros += 1\n'
                                                                                                '            if zeros '
                                                                                                '== 0:\n'
                                                                                                '                lines '
                                                                                                '+= 1\n'
                                                                                                '                for '
                                                                                                'i1 in range(i, 1, '
                                                                                                '-1):\n'
                                                                                                '                    '
                                                                                                'for j in '
                                                                                                'range(self.width):\n'
                                                                                                '                        '
                                                                                                'self.board[i1][j] = '
                                                                                                'self.board[i1 - '
                                                                                                '1][j]\n'
                                                                                                '        self.score += '
                                                                                                'lines ** 2\n'
                                                                                                '\n'
                                                                                                '    def '
                                                                                                'go_space(self):\n'
                                                                                                '        while not '
                                                                                                'self.intersect():\n'
                                                                                                '            '
                                                                                                'self.block.y += 1\n'
                                                                                                '        self.block.y '
                                                                                                '-= 1\n'
                                                                                                '        '
                                                                                                'self.freeze()\n'
                                                                                                '\n'
                                                                                                '    def '
                                                                                                'go_down(self):\n'
                                                                                                '        self.block.y '
                                                                                                '+= 1\n'
                                                                                                '        if '
                                                                                                'self.intersect():\n'
                                                                                                '            '
                                                                                                'self.block.y -= 1\n'
                                                                                                '            '
                                                                                                'self.freeze()\n'
                                                                                                '\n'
                                                                                                '    def go_side(self, '
                                                                                                'dx):\n'
                                                                                                '        old_x = '
                                                                                                'self.block.x\n'
                                                                                                '        self.block.x '
                                                                                                '+= dx\n'
                                                                                                '        if '
                                                                                                'self.intersect():\n'
                                                                                                '            '
                                                                                                'self.block.x = old_x\n'
                                                                                                '\n'
                                                                                                '    def move(self, '
                                                                                                'dx, dy):\n'
                                                                                                '        old_x = '
                                                                                                'self.block.x\n'
                                                                                                '        old_y = '
                                                                                                'self.block.y\n'
                                                                                                '        self.block.x '
                                                                                                '+= dx\n'
                                                                                                '        self.block.y '
                                                                                                '+= dy\n'
                                                                                                '        if '
                                                                                                'self.intersect():\n'
                                                                                                '            '
                                                                                                'self.block.x = old_x\n'
                                                                                                '            '
                                                                                                'self.block.y = old_y\n'
                                                                                                '\n'
                                                                                                '    def '
                                                                                                'rotate(self):\n'
                                                                                                '        old_rotation '
                                                                                                '= '
                                                                                                'self.block.rotation\n'
                                                                                                '        '
                                                                                                'self.block.rotate()\n'
                                                                                                '        if '
                                                                                                'self.intersect():\n'
                                                                                                '            '
                                                                                                'self.block.rotation = '
                                                                                                'old_rotation\n'
                                                                                                'PYFILE',
                                                                                     'shell': True}}]},
    '937087b6-f668-4ba6-9110-60682ee33441': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'xdg-mime default '
                                                                                                'vlc.desktop video/mp4',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'xdg-mime default '
                                                                                                'vlc.desktop '
                                                                                                'video/x-matroska',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'xdg-mime default '
                                                                                                'vlc.desktop '
                                                                                                'video/webm',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'xdg-mime default '
                                                                                                'vlc.desktop '
                                                                                                'video/x-msvideo',
                                                                                     'shell': True}}]},
    '98e8e339-5f91-4ed2-b2b2-12647cb134f4': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/concat.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/98e8e339-5f91-4ed2-b2b2-12647cb134f4/concat_gold.docx'",
                                                                                     'shell': True}}]},
    '9f3bb592-209d-43bc-bb47-d77d9df56504': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'ffmpeg -y -i '
                                                                                                '/home/user/video.mp4 '
                                                                                                '-map 0:s:0 '
                                                                                                '/home/user/subtitles.srt '
                                                                                                '2>/dev/null || ffmpeg '
                                                                                                '-y -i '
                                                                                                '/home/user/video.mp4 '
                                                                                                '-map 0:s '
                                                                                                '/home/user/subtitles.srt '
                                                                                                '2>/dev/null || true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'ffmpeg -y -i '
                                                                                                '/home/user/video.mp4 '
                                                                                                '-map 0:v -map 0:a -c '
                                                                                                'copy '
                                                                                                '/tmp/video_nosub.mp4 '
                                                                                                '2>/dev/null && mv '
                                                                                                '/tmp/video_nosub.mp4 '
                                                                                                '/home/user/video.mp4 '
                                                                                                '|| true',
                                                                                     'shell': True}}]},
    'a503b07f-9119-456b-b75d-f5146737d24f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/receipt.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/a503b07f-9119-456b-b75d-f5146737d24f/receipt_Gold.pdf'",
                                                                                     'shell': True}}]},
    'a74b607e-6bb5-4ea8-8a7c-5d97c7bbcd2a': {   'actions': [   {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5, 'command': ''}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome; '
                                                                                                'sleep 3',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo '
                                                                                                'aW1wb3J0IGpzb24KcCA9ICIvaG9tZS91c2VyL2Nocm9tZS1kYXRhL0RlZmF1bHQvUHJlZmVyZW5jZXMiCmQgPSBqc29uLmxvYWQob3BlbihwKSkKbSA9IGpzb24ubG9hZChvcGVuKCIvaG9tZS91c2VyL0Rlc2t0b3AvaGVsbG9FeHRlbnNpb24vbWFuaWZlc3QuanNvbiIpKQpleHRzID0gZC5zZXRkZWZhdWx0KCJleHRlbnNpb25zIiwge30pLnNldGRlZmF1bHQoInNldHRpbmdzIiwge30pCiMgQWRkIHRoZSB1bnBhY2tlZCBleHRlbnNpb24KZXh0c1siaGVsbG9fZXh0X2lkIl0gPSB7CiAgICAicGF0aCI6ICIvaG9tZS91c2VyL0Rlc2t0b3AvaGVsbG9FeHRlbnNpb24iLAogICAgInN0YXRlIjogMSwKICAgICJsb2NhdGlvbiI6IDQsCiAgICAibWFuaWZlc3QiOiBtLAp9CiMgRW5zdXJlIGFsbCBleHRlbnNpb25zIGhhdmUgYSAncGF0aCcga2V5IChldmFsdWF0b3IgZ2V0dGVyIHJlcXVpcmVzIGl0KQpmb3IgayBpbiBleHRzOgogICAgaWYgInBhdGgiIG5vdCBpbiBleHRzW2tdOgogICAgICAgIGV4dHNba11bInBhdGgiXSA9ICIiCmpzb24uZHVtcChkLCBvcGVuKHAsICJ3IikpCnByaW50KCJkb25lIikK '
                                                                                                '| base64 -d > '
                                                                                                '/tmp/_install_ext.py '
                                                                                                '&& python3 '
                                                                                                '/tmp/_install_ext.py',
                                                                                     'shell': True}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5, 'command': ''}}]},
    'a82b78bb-7fde-4cb3-94a4-035baf10bcf0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/chrome-data/Default '
                                                                                                '&& python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                "path='/home/user/chrome-data/Default/Bookmarks'\n"
                                                                                                'bookmarks={"roots":{"bookmark_bar":{"children":[],"name":"Bookmarks '
                                                                                                'bar","type":"folder"},"other":{"children":[],"name":"Other '
                                                                                                'bookmarks","type":"folder"},"synced":{"children":[],"name":"Mobile '
                                                                                                'bookmarks","type":"folder"}},"version":1}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    try: '
                                                                                                'bookmarks=json.load(open(path))\n'
                                                                                                '    except: pass\n'
                                                                                                "bar=bookmarks['roots']['bookmark_bar']\n"
                                                                                                "bar['children']=[c "
                                                                                                'for c in '
                                                                                                "bar.get('children',[]) "
                                                                                                'if '
                                                                                                "not(c.get('type')=='folder' "
                                                                                                'and '
                                                                                                "c.get('name')=='Liked "
                                                                                                "Authors')]\n"
                                                                                                "urls=['https://jimfan.me/','https://research.nvidia.com/person/de-an-huang','https://yukezhu.me/','https://tensorlab.cms.caltech.edu/users/anima/']\n"
                                                                                                "folder={'children':[],'name':'Liked "
                                                                                                "Authors','type':'folder','date_added':'13365000000000000','date_modified':'0','guid':'00000000-0000-0000-0000-000000000010','id':'200'}\n"
                                                                                                'for i,u in '
                                                                                                'enumerate(urls):\n'
                                                                                                '    '
                                                                                                "folder['children'].append({'date_added':'13365000000000000','date_last_used':'0','guid':f'00000000-0000-0000-0000-0000000003{i:02d}','id':str(300+i),'name':f'author{i}','type':'url','url':u})\n"
                                                                                                "bar['children'].append(folder)\n"
                                                                                                'json.dump(bookmarks, '
                                                                                                "open(path,'w'), "
                                                                                                'indent=2)\n'
                                                                                                "print('OK')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'aad10cd7-9337-4b62-b704-a857848cedf2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/notes.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/aad10cd7-9337-4b62-b704-a857848cedf2/notes.docx'",
                                                                                     'shell': True}}]},
    'acb0f96b-e27c-44d8-b55f-7cb76609dfcd': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'echo user | sudo -S '
                                                                                                'bash -c "apt-get '
                                                                                                'update -qq && '
                                                                                                'DEBIAN_FRONTEND=noninteractive '
                                                                                                'apt-get install -y '
                                                                                                '-qq locales && sed -i '
                                                                                                "'s/# "
                                                                                                "en_US.UTF-8/en_US.UTF-8/' "
                                                                                                '/etc/locale.gen && '
                                                                                                'locale-gen '
                                                                                                'en_US.UTF-8 && sed -i '
                                                                                                "'/^exec python3/i "
                                                                                                'export '
                                                                                                'LC_ALL=en_US.UTF-8\\\\nexport '
                                                                                                "LANG=en_US.UTF-8' "
                                                                                                '/usr/local/bin/start-osworld-server.sh" '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user && git '
                                                                                                'clone --depth 1 '
                                                                                                'https://github.com/xlang-ai/instructor-embedding.git '
                                                                                                '2>/dev/null || true',
                                                                                     'shell': True}}]},
    'aceb0368-56b8-4073-b70e-3dc9aee184e0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/exam'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/exam/grades.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/aceb0368-56b8-4073-b70e-3dc9aee184e0/grades.xlsx'",
                                                                                     'shell': True}}]},
    'b337d106-053f-4d37-8da0-7f9c4043a66b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "echo 'set number' >> "
                                                                                                '/home/user/.vimrc',
                                                                                     'shell': True}}]},
    'b5062e3e-641c-4e3a-907b-ac864d2e7652': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "mkdir -p '/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/authors.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/b5062e3e-641c-4e3a-907b-ac864d2e7652/authors-ground_truth.xlsx'",
                                                                                     'shell': True}}]},
    'bb83cab4-e5c7-42c7-a67b-e46068032b86': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/script.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/bb83cab4-e5c7-42c7-a67b-e46068032b86/script.docx'",
                                                                                     'shell': True}}]},
    'bc2b57f3-686d-4ec9-87ce-edf850b7e442': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/workbook-with-sample-database.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/bc2b57f3-686d-4ec9-87ce-edf850b7e442/workbook-with-sample-database_Gold.xlsx'",
                                                                                     'shell': True}}]},
    'c2751594-0cd5-4088-be1b-b5f2f9ec97c4': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                '/tmp/background_gold.png '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/c2751594-0cd5-4088-be1b-b5f2f9ec97c4/background.png'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cp '
                                                                                                '/tmp/background_gold.png '
                                                                                                '/home/user/Desktop/background.png',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'gsettings set '
                                                                                                'org.gnome.desktop.background '
                                                                                                'picture-uri '
                                                                                                "'file:///home/user/Desktop/background.png'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'gsettings set '
                                                                                                'org.gnome.desktop.background '
                                                                                                'picture-uri-dark '
                                                                                                "'file:///home/user/Desktop/background.png'",
                                                                                     'shell': True}}]},
    'c7c1e4c3-9e92-4eba-a4b8-689953975ea4': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/Professor_Contact.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/c7c1e4c3-9e92-4eba-a4b8-689953975ea4/Professor_Contact_Gold.xlsx'",
                                                                                     'shell': True}}]},
    'c867c42d-a52d-4a24-8ae3-f75d256b5618': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/contacts.csv' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/c867c42d-a52d-4a24-8ae3-f75d256b5618/contacts.csv'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/contacts.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/c867c42d-a52d-4a24-8ae3-f75d256b5618/contacts.xlsx'",
                                                                                     'shell': True}}]},
    'ce2b64a2-ddc1-4f91-8c7d-a88be7121aac': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd '
                                                                                                '/home/user/Pictures '
                                                                                                '&& mv picture1.jpg '
                                                                                                "'Kilimanjaro.jpg' "
                                                                                                '2>/dev/null; mv '
                                                                                                "picture2.jpg 'Mount "
                                                                                                "Everest.jpg' "
                                                                                                '2>/dev/null; mv '
                                                                                                'picture3.jpg '
                                                                                                "'Huashan.jpg' "
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}}]},
    'd1acdb87-bb67-4f30-84aa-990e56a09c92': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/MUST_VISIT.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/d1acdb87-bb67-4f30-84aa-990e56a09c92/MUST_VISIT_gold.xlsx'",
                                                                                     'shell': True}}]},
    'd68204bf-11c1-4b13-b48b-d303c73d4bf6': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/rearranged.png' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/d68204bf-11c1-4b13-b48b-d303c73d4bf6/rearranged_gold.png'",
                                                                                     'shell': True}}]},
    'd9b7c649-c975-4f53-88f5-940b29c47247': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/report.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/d9b7c649-c975-4f53-88f5-940b29c47247/report.xlsx'",
                                                                                     'shell': True}}]},
    'da52d699-e8d2-4dc5-9191-a2199e0b6a9b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/book_list_result.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/da52d699-e8d2-4dc5-9191-a2199e0b6a9b/book_list_result_Gold.docx'",
                                                                                     'shell': True}}]},
    'da922383-bfa4-4cd3-bbad-6bebab3d7742': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/Documents',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install PyMuPDF '
                                                                                                '2>&1 | tail -1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF_INNER'\n"
                                                                                                'import fitz, os\n'
                                                                                                "os.makedirs('/home/user/Documents/Blog', "
                                                                                                'exist_ok=True)\n'
                                                                                                'for name, title in [\n'
                                                                                                "    ('LLM Powered "
                                                                                                "Autonomous Agents', "
                                                                                                "'LLM Powered "
                                                                                                "Autonomous Agents'),\n"
                                                                                                "    ('Thinking about "
                                                                                                'High-Quality Human '
                                                                                                "Data', 'Thinking "
                                                                                                'about High-Quality '
                                                                                                "Human Data'),\n"
                                                                                                ']:\n'
                                                                                                '    doc = '
                                                                                                'fitz.open()\n'
                                                                                                '    page = '
                                                                                                'doc.new_page()\n'
                                                                                                '    '
                                                                                                'page.insert_text((72, '
                                                                                                '72), title, '
                                                                                                'fontsize=14)\n'
                                                                                                '    '
                                                                                                "doc.save('/home/user/Documents/Blog/' "
                                                                                                "+ name + '.pdf')\n"
                                                                                                '    doc.close()\n'
                                                                                                "print('pdfs "
                                                                                                "written')\n"
                                                                                                'PYEOF_INNER\n',
                                                                                     'shell': True}}]},
    'dd60633f-2c72-42ba-8547-6f2c8cb0fdb0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "mkdir -p '/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/gpt_dev_pure_code.py' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/dd60633f-2c72-42ba-8547-6f2c8cb0fdb0/gpt_dev_pure_code_gold.py'",
                                                                                     'shell': True}}]},
    'deec51c9-3b1e-4b9e-993c-4776f20e8bb2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/New "
                                                                                                'Large Language '
                                                                                                "Models.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/deec51c9-3b1e-4b9e-993c-4776f20e8bb2/New%20Large%20Language%20Models%20Gold.xlsx'",
                                                                                     'shell': True}}]},
    'df67aebb-fb3a-44fd-b75b-51b6012df509': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/references.bib' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/df67aebb-fb3a-44fd-b75b-51b6012df509/references.bib'",
                                                                                     'shell': True}}]},
    'e135df7c-7687-4ac0-a5f0-76b74438b53e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 2; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user/Desktop '
                                                                                                '&& libreoffice '
                                                                                                '--headless '
                                                                                                '--convert-to html '
                                                                                                'annual-enterprise-survey-2021-financial-year-provisional.xlsx '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'from pathlib import '
                                                                                                'Path\n'
                                                                                                "p = Path('/home/user/Desktop/annual-enterprise-survey-2021-financial-year-provisional.html')\n"
                                                                                                'if p.exists():\n'
                                                                                                "    text = p.read_text(encoding='utf-8')\n"
                                                                                                "    text = text.replace('<colgroup span=\"10\" width=\"94\"></colgroup>', '<colgroup span=\"10\" width=\"107\"></colgroup>', 1)\n"
                                                                                                "    p.write_text(text, encoding='utf-8')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {'command': 'sleep 3', 'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome '
                                                                                                '2>/dev/null || true; '
                                                                                                'pkill -9 -f '
                                                                                                "'socat "
                                                                                                "tcp-listen:9222' "
                                                                                                '2>/dev/null || true; '
                                                                                                'sleep 2; true',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'google-chrome '
                                                                                                '--remote-debugging-port=1337 '
                                                                                                '--no-first-run '
                                                                                                '--no-default-browser-check '
                                                                                                'https://aclanthology.org/ '
                                                                                                'https://openai.com/ '
                                                                                                'https://www.linkedin.com/home/ '
                                                                                                'file:///home/user/Desktop/annual-enterprise-survey-2021-financial-year-provisional.html'}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'socat '
                                                                                                'tcp-listen:9222,fork '
                                                                                                'tcp:localhost:1337'}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'sleep 3',
                                                                                     'shell': True}}]},
    'e1fc0df3-c8b9-4ee7-864c-d0b590d3aa56': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'sudo '
                                                                                                'DEBIAN_FRONTEND=noninteractive '
                                                                                                'apt-get update -qq '
                                                                                                '2>/dev/null && sudo '
                                                                                                'DEBIAN_FRONTEND=noninteractive '
                                                                                                'apt-get install -y '
                                                                                                '-qq default-jre '
                                                                                                'libreoffice-java-common '
                                                                                                '2>/dev/null; echo '
                                                                                                "'JRE done'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 3; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'curl -sL -o '
                                                                                                '/tmp/LanguageTool.oxt '
                                                                                                "'https://languagetool.org/download/LanguageTool-stable.oxt' "
                                                                                                '&& ls -la '
                                                                                                '/tmp/LanguageTool.oxt '
                                                                                                "&& echo 'Download OK'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'unopkg add --force '
                                                                                                '/tmp/LanguageTool.oxt '
                                                                                                "2>&1; echo 'unopkg "
                                                                                                "done'",
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'e2392362-125e-4f76-a2ee-524b183a3412': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import yaml, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/Code/Website/academicpages.github.io/_config.yml'\n"
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        config = '
                                                                                                'yaml.safe_load(f)\n'
                                                                                                "    config['name'] = "
                                                                                                "'Test Account'\n"
                                                                                                "    if 'author' not "
                                                                                                'in config:\n'
                                                                                                '        '
                                                                                                "config['author'] = "
                                                                                                '{}\n'
                                                                                                '    '
                                                                                                "config['author']['name'] "
                                                                                                "= 'Test Account'\n"
                                                                                                '    '
                                                                                                "config['author']['email'] "
                                                                                                "= 'Test@gmail.com'\n"
                                                                                                '    with open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '        '
                                                                                                'yaml.dump(config, f, '
                                                                                                'default_flow_style=False)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'e8172110-ec08-421b-a6f5-842e6451911f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/character_gimp.png' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/e8172110-ec08-421b-a6f5-842e6451911f/character_no_background_gold.png'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/character_code.png' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/e8172110-ec08-421b-a6f5-842e6451911f/character_no_background_gold.png'",
                                                                                     'shell': True}}]},
    'eb303e01-261e-4972-8c07-c9b4e7a4922a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/lecture1-2021-with-ink.pptx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/eb303e01-261e-4972-8c07-c9b4e7a4922a/lecture1-2021-with-ink_Gold.pptx'",
                                                                                     'shell': True}}]},
    'ee9a3c83-f437-4879-8918-be5efbb9fac7': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "echo 'libreoffice "
                                                                                                '--headless '
                                                                                                '--convert-to csv '
                                                                                                "/home/user/Desktop/file_example_ODS_5000.ods' "
                                                                                                '>> '
                                                                                                '/home/user/.bash_history',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 3',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user/Desktop '
                                                                                                '&& libreoffice '
                                                                                                '--headless '
                                                                                                '--convert-to csv '
                                                                                                'file_example_ODS_5000.ods '
                                                                                                '2>&1 && echo '
                                                                                                "'Converted OK' && ls "
                                                                                                '-la '
                                                                                                'file_example_ODS_5000.csv '
                                                                                                "|| echo 'Conversion "
                                                                                                "failed'",
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'test -f '
                                                                                                '/home/user/Desktop/file_example_ODS_5000.csv '
                                                                                                "&& echo 'CSV exists' "
                                                                                                '|| (curl -sL -o '
                                                                                                '/home/user/Desktop/file_example_ODS_5000.csv '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/ee9a3c83-f437-4879-8918-be5efbb9fac7/file_example_ODS_5000.csv' "
                                                                                                "&& echo 'Downloaded "
                                                                                                "fallback')",
                                                                                     'shell': True}}]},
    'f5c13cdd-205c-4719-a562-348ae5cd1d91': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': [   '/bin/bash',
                                                                                                    '-c',
                                                                                                    'sudo -u user '
                                                                                                    'DISPLAY=:1 '
                                                                                                    'dbus-send '
                                                                                                    '--session '
                                                                                                    '--print-reply '
                                                                                                    '--dest=org.a11y.Bus '
                                                                                                    '/org/a11y/bus '
                                                                                                    'org.freedesktop.DBus.Properties.Set '
                                                                                                    'string:org.a11y.Status '
                                                                                                    'string:IsEnabled '
                                                                                                    'variant:boolean:true '
                                                                                                    '>/dev/null 2>&1; '
                                                                                                    'true']}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': '/usr/bin/thunderbird '
                                                                                                '-compose '
                                                                                                '"to=\'fox@someuniversity.edu,iron@someuniversity.edu,nancy@someuniversity.edu,stella@someuniversity.edu\',from=\'Anonym '
                                                                                                'Tester '
                                                                                                "<anonym-x2024@outlook.com>',subject='Reminder "
                                                                                                'of '
                                                                                                "Payment',body='$(cat "
                                                                                                '/home/user/.payment-reminder-mail-body.html)\'"',
                                                                                     'shell': True}},
                                                               {'type': 'wait', 'parameters': {'seconds': 25}}]},
    'f7dfbef3-7697-431c-883a-db8583a4e4f9': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "echo 'libreoffice "
                                                                                                '--headless '
                                                                                                '--convert-to pdf '
                                                                                                "*.doc' >> "
                                                                                                '/home/user/.bash_history',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 2; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                '/tmp/pdf_gold.tar.gz '
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/f7dfbef3-7697-431c-883a-db8583a4e4f9/pdf.tar.gz' "
                                                                                                '&& tar -zxf '
                                                                                                '/tmp/pdf_gold.tar.gz '
                                                                                                '-C /home/user/Desktop',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'f8369178-fafe-40c2-adc4-b9b08a125456': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'gsettings set '
                                                                                                'org.gnome.desktop.interface '
                                                                                                "gtk-theme 'Orchis' "
                                                                                                '2>/dev/null || true',
                                                                                     'shell': True}}]},
    'f8cfa149-d1c1-4215-8dac-4a0932bad3c2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -f '
                                                                                                '"/home/user/chrome-data/Default/Last Session" '
                                                                                                '"/home/user/chrome-data/Default/Last Tabs" '
                                                                                                '2>/dev/null; true',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    'https://www.google.com/search?q=nereida']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'f918266a-b3e0-4914-865d-4faa564f1aef': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall soffice.bin '
                                                                                                '2>/dev/null; sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/Desktop'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Desktop/log.txt' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/multi_apps/f918266a-b3e0-4914-865d-4faa564f1aef/log_Gold.txt'",
                                                                                     'shell': True}}]}}
