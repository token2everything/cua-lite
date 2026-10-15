"""Hand-curated oracle metadata for chrome eval tasks.

Chrome / web-browsing tasks.

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
- total entries:           43
- with actions:            40
- after_postconfig=True:   11
- block: exclude_reason:   2
- evaluator override:      3
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {   '030eeff7-b492-4218-b312-701ec99ee0cc': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/.config/google-chrome/Default'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/google-chrome/Default/Preferences'\n"
                                                                                                'prefs = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        prefs = '
                                                                                                'json.load(f)\n'
                                                                                                'prefs.setdefault("enable_do_not_track", '
                                                                                                'True)\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(prefs, '
                                                                                                'f)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '06fe7178-4491-4589-810f-2e2bc9502122': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.lonelyplanet.com',
                                                                                                    'https://www.airbnb.com',
                                                                                                    'https://www.tripadvisor.com']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '0d8b7de3-e8de-4d86-b9fd-dd2dce58a217': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.drugs.com/npc/']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '12086550-11c0-466b-b367-1d9e75b3910e': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--no-first-run',
                                                                                                    '--no-default-browser-check',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    '--user-data-dir=/tmp/chrome-data']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'PAGE_ID=$(curl -s '
                                                                                                'http://localhost:1337/json '
                                                                                                '| python3 -c "import '
                                                                                                'json,sys; '
                                                                                                'd=json.load(sys.stdin); '
                                                                                                "print([p['id'] for p "
                                                                                                'in d if '
                                                                                                'p[\'type\']==\'page\'][0])")\n'
                                                                                                'python3 -c "\n'
                                                                                                'import websocket, '
                                                                                                'json\n'
                                                                                                'ws = '
                                                                                                "websocket.create_connection('ws://localhost:1337/devtools/page/$PAGE_ID')\n"
                                                                                                "ws.send(json.dumps({'id':1,'method':'Page.navigate','params':{'url':'chrome://password-manager/passwords'}}))\n"
                                                                                                'print(ws.recv())\n'
                                                                                                'ws.close()\n'
                                                                                                '"\n'
                                                                                                'sleep 3',
                                                                                     'shell': True}}]},
    '121ba48f-9e17-48ce-9bc6-a4fb17a7ebba': {},
    '1704f00f-79e6-43a7-961b-cedd3724d5fd': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install '
                                                                                                'websocket-client '
                                                                                                'requests 2>/dev/null; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'from datetime import '
                                                                                                'datetime, timedelta\n'
                                                                                                'try:\n'
                                                                                                '    from zoneinfo '
                                                                                                'import ZoneInfo\n'
                                                                                                '    now = '
                                                                                                "datetime.now(ZoneInfo('Europe/Zurich'))\n"
                                                                                                'except ImportError:\n'
                                                                                                '    import os\n'
                                                                                                '    now = '
                                                                                                'datetime.utcnow() + '
                                                                                                'timedelta(hours=2)  # '
                                                                                                'rough CEST fallback\n'
                                                                                                '\n'
                                                                                                '# Next Monday: same '
                                                                                                "logic as OSWorld's "
                                                                                                'get_rule_relativeTime\n'
                                                                                                'days_until_monday = '
                                                                                                '(6 - now.weekday()) + '
                                                                                                '1\n'
                                                                                                'mon = now + '
                                                                                                'timedelta(days=days_until_monday)\n'
                                                                                                '# Next Friday (same '
                                                                                                'week as Monday): '
                                                                                                'Monday + 4\n'
                                                                                                'fri = mon + '
                                                                                                'timedelta(days=4)\n'
                                                                                                '\n'
                                                                                                '# Build URL query '
                                                                                                'string\n'
                                                                                                'from urllib.parse '
                                                                                                'import urlencode, '
                                                                                                'quote\n'
                                                                                                'params = {\n'
                                                                                                "    'locationName': "
                                                                                                "'Zürich',\n"
                                                                                                '    '
                                                                                                "'dropLocationName': "
                                                                                                "'Zürich',\n"
                                                                                                '    '
                                                                                                "'filterCriteria_carCategory': "
                                                                                                "'large',\n"
                                                                                                '    '
                                                                                                "'filterCriteria_sortBy': "
                                                                                                "'PRICE',\n"
                                                                                                "    'puDay': "
                                                                                                'str(mon.day),\n'
                                                                                                "    'puMonth': "
                                                                                                'str(mon.month),\n'
                                                                                                "    'puYear': "
                                                                                                'str(mon.year),\n'
                                                                                                "    'doDay': "
                                                                                                'str(fri.day),\n'
                                                                                                "    'doMonth': "
                                                                                                'str(fri.month),\n'
                                                                                                "    'doYear': "
                                                                                                'str(fri.year),\n'
                                                                                                '}\n'
                                                                                                'qs = '
                                                                                                'urlencode(params)\n'
                                                                                                'url = '
                                                                                                "f'https://www.rentalcars.com/search-results?{qs}'\n"
                                                                                                'with '
                                                                                                "open('/tmp/chrome_url.txt', "
                                                                                                "'w') as f:\n"
                                                                                                '    f.write(url)\n'
                                                                                                "print(f'URL: {url}')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome '
                                                                                                '2>/dev/null; pkill -f '
                                                                                                'socat 2>/dev/null; '
                                                                                                'sleep 2',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'socat',
                                                                                                    'tcp-listen:9222,fork,reuseaddr',
                                                                                                    'tcp:localhost:1337']}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    'about:blank',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'import json, time, '
                                                                                                'sys\n'
                                                                                                'try:\n'
                                                                                                '    import requests\n'
                                                                                                'except ImportError:\n'
                                                                                                '    import '
                                                                                                'urllib.request\n'
                                                                                                '    class requests:\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def get(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            class R:\n'
                                                                                                '                def '
                                                                                                'json(self):\n'
                                                                                                '                    '
                                                                                                'return '
                                                                                                'json.loads(urllib.request.urlopen(url, '
                                                                                                'timeout=timeout).read())\n'
                                                                                                '            return '
                                                                                                'R()\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def put(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            req = '
                                                                                                'urllib.request.Request(url, '
                                                                                                "method='PUT')\n"
                                                                                                '            '
                                                                                                'urllib.request.urlopen(req, '
                                                                                                'timeout=timeout)\n'
                                                                                                '            return '
                                                                                                'None\n'
                                                                                                '\n'
                                                                                                'try:\n'
                                                                                                '    import websocket\n'
                                                                                                'except ImportError:\n'
                                                                                                "    print('ERROR: "
                                                                                                'websocket not '
                                                                                                "available')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                '\n'
                                                                                                'url = '
                                                                                                "open('/tmp/chrome_url.txt').read().strip()\n"
                                                                                                "print(f'Target URL: "
                                                                                                "{url}')\n"
                                                                                                '\n'
                                                                                                '# Wait for Chrome '
                                                                                                'CDP\n'
                                                                                                'for attempt in '
                                                                                                'range(30):\n'
                                                                                                '    try:\n'
                                                                                                '        tabs = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                '        pages = [t '
                                                                                                'for t in tabs if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                '        if pages:\n'
                                                                                                '            break\n'
                                                                                                '    except:\n'
                                                                                                '        pass\n'
                                                                                                '    try:\n'
                                                                                                '        '
                                                                                                "requests.put('http://localhost:1337/json/new?about:blank', "
                                                                                                'timeout=5)\n'
                                                                                                '    except:\n'
                                                                                                '        pass\n'
                                                                                                '    time.sleep(2)\n'
                                                                                                'else:\n'
                                                                                                "    print('ERROR: No "
                                                                                                'page tabs found after '
                                                                                                "30 attempts')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                '\n'
                                                                                                'page = pages[0]\n'
                                                                                                'ws = '
                                                                                                "websocket.create_connection(page['webSocketDebuggerUrl'])\n"
                                                                                                "ws.send(json.dumps({'id': "
                                                                                                "1, 'method': "
                                                                                                "'Page.navigate', "
                                                                                                "'params': {'url': "
                                                                                                'url}}))\n'
                                                                                                'result = '
                                                                                                'json.loads(ws.recv())\n'
                                                                                                'ws.close()\n'
                                                                                                "print(f'Navigated: "
                                                                                                "{result}')\n"
                                                                                                '\n'
                                                                                                '# Close extra tabs\n'
                                                                                                'time.sleep(2)\n'
                                                                                                'tabs2 = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                'pages2 = [t for t in '
                                                                                                'tabs2 if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                'for extra in '
                                                                                                'pages2[1:]:\n'
                                                                                                '    try:\n'
                                                                                                '        '
                                                                                                'requests.get(f\'http://localhost:1337/json/close/{extra["id"]}\', '
                                                                                                'timeout=3)\n'
                                                                                                '    except:\n'
                                                                                                '        pass\n'
                                                                                                "print('Done')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '2888b4e6-5b47-4b57-8bf5-c73827890774': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337']}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'socat',
                                                                                                    'tcp-listen:9222,fork',
                                                                                                    'tcp:localhost:1337']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'chrome_open_tabs',
                                                                   'parameters': {   'urls_to_open': [   'https://www.macys.com/shop/mens-clothing/mens-shirts/Sleeve_length,Men_regular_size_t,Price_discount_range/Short%20Sleeve,L,50_PERCENT_%20off%20%26%20more?id=20626']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                'exclude_reason': 'upstream_live_site_drift'},
    '2ad9387a-65d8-4e33-ad5b-7580065a27ca': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/chrome-data/Default',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os, '
                                                                                                'time\n'
                                                                                                'path = '
                                                                                                "'/home/user/chrome-data/Default/Bookmarks'\n"
                                                                                                'bookmarks = {"roots": '
                                                                                                '{"bookmark_bar": '
                                                                                                '{"children": [], '
                                                                                                '"name": "Bookmarks '
                                                                                                'bar", "type": '
                                                                                                '"folder"}, "other": '
                                                                                                '{"children": [], '
                                                                                                '"name": "Other '
                                                                                                'bookmarks", "type": '
                                                                                                '"folder"}, "synced": '
                                                                                                '{"children": [], '
                                                                                                '"name": "Mobile '
                                                                                                'bookmarks", "type": '
                                                                                                '"folder"}}, '
                                                                                                '"version": 1}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        bookmarks = '
                                                                                                'json.load(f)\n'
                                                                                                '\n'
                                                                                                '# Add Favorites '
                                                                                                'folder to bookmark '
                                                                                                'bar\n'
                                                                                                'bar = '
                                                                                                'bookmarks.setdefault("roots", '
                                                                                                '{}).setdefault("bookmark_bar", '
                                                                                                '{"children": [], '
                                                                                                '"name": "Bookmarks '
                                                                                                'bar", "type": '
                                                                                                '"folder"})\n'
                                                                                                'children = '
                                                                                                'bar.setdefault("children", '
                                                                                                '[])\n'
                                                                                                '# Check if Favorites '
                                                                                                'already exists\n'
                                                                                                'if not '
                                                                                                'any(c.get("name") == '
                                                                                                '"Favorites" and '
                                                                                                'c.get("type") == '
                                                                                                '"folder" for c in '
                                                                                                'children):\n'
                                                                                                '    '
                                                                                                'children.append({\n'
                                                                                                '        "children": '
                                                                                                '[],\n'
                                                                                                '        "date_added": '
                                                                                                '"13365000000000000",\n'
                                                                                                '        '
                                                                                                '"date_last_used": '
                                                                                                '"0",\n'
                                                                                                '        '
                                                                                                '"date_modified": '
                                                                                                '"0",\n'
                                                                                                '        "guid": '
                                                                                                '"00000000-0000-0000-0000-000000000001",\n'
                                                                                                '        "id": "100",\n'
                                                                                                '        "name": '
                                                                                                '"Favorites",\n'
                                                                                                '        "type": '
                                                                                                '"folder"\n'
                                                                                                '    })\n'
                                                                                                '\n'
                                                                                                '# IMPORTANT: '
                                                                                                'evaluator checks '
                                                                                                'set(folder_names) == '
                                                                                                'set(["Favorites"])\n'
                                                                                                '# So we need ONLY '
                                                                                                '"Favorites" as a '
                                                                                                'folder - remove other '
                                                                                                'folders\n'
                                                                                                'bar["children"] = [c '
                                                                                                'for c in children if '
                                                                                                'not (c.get("type") == '
                                                                                                '"folder" and '
                                                                                                'c.get("name") != '
                                                                                                '"Favorites")]\n'
                                                                                                '# Keep non-folder '
                                                                                                'items too\n'
                                                                                                'bar["children"] = [c '
                                                                                                'for c in '
                                                                                                'bar["children"] if '
                                                                                                'c.get("type") != '
                                                                                                '"folder"] + [c for c '
                                                                                                'in bar["children"] if '
                                                                                                'c.get("type") == '
                                                                                                '"folder"]\n'
                                                                                                '\n'
                                                                                                '# Actually, re-read '
                                                                                                'the evaluator: it '
                                                                                                'checks '
                                                                                                'set(folder_names) == '
                                                                                                'set(rule["names"])\n'
                                                                                                '# folder_names = '
                                                                                                'names of folders in '
                                                                                                'bookmark_bar.children '
                                                                                                'where type == '
                                                                                                '"folder"\n'
                                                                                                '# So we need exactly '
                                                                                                'one folder named '
                                                                                                '"Favorites" and no '
                                                                                                'other folders\n'
                                                                                                '# But we should keep '
                                                                                                'URL bookmarks intact\n'
                                                                                                '\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    '
                                                                                                'json.dump(bookmarks, '
                                                                                                'f, indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '2ae9ba84-3a0d-4d4c-8338-3a1478dc5fe3': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/.config/google-chrome/Default'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/google-chrome/Default/Preferences'\n"
                                                                                                'prefs = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        prefs = '
                                                                                                'json.load(f)\n'
                                                                                                'prefs.setdefault("profile", '
                                                                                                '{})\n'
                                                                                                'prefs["profile"]["name"] '
                                                                                                '= "Thomas"\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(prefs, '
                                                                                                'f)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '3299584d-8f11-4457-bf4c-ce98f7600250': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/chrome-data/Default',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/chrome-data/Default/Preferences'\n"
                                                                                                'prefs = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        prefs = '
                                                                                                'json.load(f)\n'
                                                                                                '\n'
                                                                                                'def deep_merge(base, '
                                                                                                'patch):\n'
                                                                                                '    for k, v in '
                                                                                                'patch.items():\n'
                                                                                                '        if '
                                                                                                'isinstance(v, dict) '
                                                                                                'and '
                                                                                                'isinstance(base.get(k), '
                                                                                                'dict):\n'
                                                                                                '            '
                                                                                                'deep_merge(base[k], '
                                                                                                'v)\n'
                                                                                                '        else:\n'
                                                                                                '            base[k] = '
                                                                                                'v\n'
                                                                                                '\n'
                                                                                                'deep_merge(prefs, '
                                                                                                '{"session": '
                                                                                                '{"restore_on_startup": '
                                                                                                '5, "startup_urls": '
                                                                                                '[]}})\n'
                                                                                                '\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(prefs, '
                                                                                                'f)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '35253b65-1c19-4304-8aa4-6884b8218fc0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'cat > '
                                                                                                '/home/user/Desktop/play-puzzle-game-2048.desktop '
                                                                                                "<< 'EOF'\n"
                                                                                                '[Desktop Entry]\n'
                                                                                                'Version=1.0\n'
                                                                                                'Type=Application\n'
                                                                                                'Name=Play Puzzle Game '
                                                                                                '2048\n'
                                                                                                'Exec=google-chrome '
                                                                                                '--app=https://play2048.co\n'
                                                                                                'Icon=google-chrome\n'
                                                                                                'Terminal=false\n'
                                                                                                'StartupWMClass=play2048.co\n'
                                                                                                'EOF\n'
                                                                                                'chmod +x '
                                                                                                '/home/user/Desktop/play-puzzle-game-2048.desktop',
                                                                                     'shell': True}}]},
    '368d9ba4-203c-40c1-9fa3-da2f1430ce63': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install '
                                                                                                'websocket-client '
                                                                                                'requests 2>/dev/null; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'from datetime import '
                                                                                                'datetime\n'
                                                                                                'month = '
                                                                                                "datetime.now().strftime('%B').lower()\n"
                                                                                                'url = '
                                                                                                "f'https://www.accuweather.com/en/gb/manchester/m2/{month}-weather/328328'\n"
                                                                                                'with '
                                                                                                "open('/tmp/chrome_url.txt', "
                                                                                                "'w') as f:\n"
                                                                                                '    f.write(url)\n'
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome '
                                                                                                '2>/dev/null; pkill -f '
                                                                                                'socat 2>/dev/null; '
                                                                                                'sleep 2',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'socat '
                                                                                                'tcp-listen:9222,fork,reuseaddr '
                                                                                                'tcp:localhost:1337',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'google-chrome '
                                                                                                '--no-sandbox '
                                                                                                '--remote-debugging-port=1337 '
                                                                                                'about:blank '
                                                                                                '--user-data-dir=/home/user/chrome-data '
                                                                                                '--remote-allow-origins=*',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'import json, time, '
                                                                                                'sys\n'
                                                                                                'try:\n'
                                                                                                '    import requests\n'
                                                                                                'except ImportError:\n'
                                                                                                '    import '
                                                                                                'urllib.request\n'
                                                                                                '    class requests:\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def get(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            class R:\n'
                                                                                                '                def '
                                                                                                'json(self):\n'
                                                                                                '                    '
                                                                                                'return '
                                                                                                'json.loads(urllib.request.urlopen(url, '
                                                                                                'timeout=timeout).read())\n'
                                                                                                '            return '
                                                                                                'R()\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def put(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            req = '
                                                                                                'urllib.request.Request(url, '
                                                                                                "method='PUT')\n"
                                                                                                '            '
                                                                                                'urllib.request.urlopen(req, '
                                                                                                'timeout=timeout)\n'
                                                                                                '            return '
                                                                                                'None\n'
                                                                                                'try:\n'
                                                                                                '    import websocket\n'
                                                                                                'except ImportError:\n'
                                                                                                "    print('ERROR: "
                                                                                                'websocket not '
                                                                                                "available')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                'url = '
                                                                                                "open('/tmp/chrome_url.txt').read().strip()\n"
                                                                                                "print(f'Target URL: "
                                                                                                "{url}')\n"
                                                                                                'for attempt in '
                                                                                                'range(30):\n'
                                                                                                '    try:\n'
                                                                                                '        tabs = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                '        pages = [t '
                                                                                                'for t in tabs if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                '        if pages:\n'
                                                                                                '            break\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                '    try:\n'
                                                                                                '        '
                                                                                                "requests.put('http://localhost:1337/json/new?about:blank', "
                                                                                                'timeout=5)\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                '    time.sleep(2)\n'
                                                                                                'else:\n'
                                                                                                "    print('ERROR: No "
                                                                                                'page tabs found after '
                                                                                                "30 attempts')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                'page = pages[0]\n'
                                                                                                'ws = '
                                                                                                "websocket.create_connection(page['webSocketDebuggerUrl'])\n"
                                                                                                "ws.send(json.dumps({'id': "
                                                                                                "1, 'method': "
                                                                                                "'Page.navigate', "
                                                                                                "'params': {'url': "
                                                                                                'url}}))\n'
                                                                                                'result = '
                                                                                                'json.loads(ws.recv())\n'
                                                                                                'ws.close()\n'
                                                                                                "print(f'Navigated: "
                                                                                                "{result}')\n"
                                                                                                'time.sleep(2)\n'
                                                                                                'tabs2 = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                'pages2 = [t for t in '
                                                                                                'tabs2 if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                'for extra in '
                                                                                                'pages2[1:]:\n'
                                                                                                '    try:\n'
                                                                                                '        eid = '
                                                                                                "extra['id']\n"
                                                                                                '        '
                                                                                                "requests.get(f'http://localhost:1337/json/close/{eid}', "
                                                                                                'timeout=3)\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                "print('Done')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                'after_postconfig': True,
                                                'evaluator': {
                                                    'func': [
                                                        'check_direct_json_object',
                                                        'is_expected_url_pattern_match',
                                                    ],
                                                    'result': [
                                                        {   'type': 'url_dashPart',
                                                            'goto_prefix': 'https://www.',
                                                            'partIndex': -2,
                                                            'needDeleteId': False,
                                                            'returnType': 'json',
                                                            'key': 'time'},
                                                        {   'type': 'active_url_from_accessTree',
                                                            'goto_prefix': 'https://www.'},
                                                    ],
                                                    # Relaxed from `/manchester/` to also accept the AccuWeather
                                                    # server-side IP-geo redirect target seen on v5 host
                                                    # (`…/en/gb/london/<postcode>/…`). Both URLs share the
                                                    # `may-weather/328328` segment which the first evaluator
                                                    # already pins down, so the city slug is the only piece that
                                                    # legitimately varies with the request IP.
                                                    'expected': [
                                                        {   'type': 'rule_relativeTime',
                                                            'rules': {   'relativeTime': {'from': 'this month'},
                                                                         'expected': {'time': '{month}-weather'}}},
                                                        {   'type': 'rule',
                                                            'rules': {'expected': ['(?:/manchester/|/london/[^/]+/)']}},
                                                    ],
                                                }},
    '44ee5668-ecd5-4366-a6ce-c1c9b8d4e938': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import sqlite3, os, '
                                                                                                'glob\n'
                                                                                                '\n'
                                                                                                '# Chrome history '
                                                                                                'database path\n'
                                                                                                'paths = '
                                                                                                "glob.glob('/home/user/chrome-data/Default/History') "
                                                                                                '+ '
                                                                                                "glob.glob('/home/user/.config/google-chrome/Default/History')\n"
                                                                                                'for db_path in '
                                                                                                'paths:\n'
                                                                                                '    if '
                                                                                                'os.path.exists(db_path):\n'
                                                                                                '        conn = '
                                                                                                'sqlite3.connect(db_path)\n'
                                                                                                '        cursor = '
                                                                                                'conn.cursor()\n'
                                                                                                '        # Delete all '
                                                                                                'YouTube URLs from '
                                                                                                'history\n'
                                                                                                '        '
                                                                                                'cursor.execute("DELETE '
                                                                                                'FROM urls WHERE url '
                                                                                                'LIKE \'%youtube%\'")\n'
                                                                                                '        '
                                                                                                'cursor.execute("DELETE '
                                                                                                'FROM visits WHERE url '
                                                                                                'NOT IN (SELECT id '
                                                                                                'FROM urls)")\n'
                                                                                                '        '
                                                                                                'conn.commit()\n'
                                                                                                '        conn.close()\n'
                                                                                                '        '
                                                                                                'print(f"Cleaned '
                                                                                                '{db_path}")\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'config_append': [   {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                                     {   'type': 'execute',
                                                                         'parameters': {   'command': 'pkill -9 -f '
                                                                                                      'chrome '
                                                                                                      '2>/dev/null; '
                                                                                                      'sleep 4; true',
                                                                                           'shell': True}},
                                                                     {   'type': 'execute',
                                                                         'parameters': {   'command': 'python3 << '
                                                                                                      "'HEOF'\n"
                                                                                                      'import sqlite3, '
                                                                                                      'datetime\n'
                                                                                                      'path = '
                                                                                                      "'/home/user/chrome-data/Default/History'\n"
                                                                                                      'ts = '
                                                                                                      'int((datetime.datetime.now()-datetime.datetime(1601,1,1)).total_seconds()*1e6)\n'
                                                                                                      'conn = '
                                                                                                      'sqlite3.connect(path)\n'
                                                                                                      'c = '
                                                                                                      'conn.cursor()\n'
                                                                                                      "c.execute('INSERT "
                                                                                                      'OR IGNORE INTO '
                                                                                                      'urls '
                                                                                                      '(url,title,visit_count,typed_count,last_visit_time,hidden) '
                                                                                                      'VALUES '
                                                                                                      "(?,?,1,0,?,0)', "
                                                                                                      "('https://www.youtube.com/watch?v=dQw4w9WgXcQ', "
                                                                                                      "'Rick Astley - "
                                                                                                      'Never Gonna '
                                                                                                      "Give You Up', "
                                                                                                      'ts))\n'
                                                                                                      'uid = '
                                                                                                      'c.lastrowid\n'
                                                                                                      'if uid: '
                                                                                                      "c.execute('INSERT "
                                                                                                      'INTO visits '
                                                                                                      '(url,visit_time,from_visit,transition,segment_id,visit_duration) '
                                                                                                      'VALUES '
                                                                                                      "(?,?,0,805306368,0,0)', "
                                                                                                      '(uid, ts))\n'
                                                                                                      'conn.commit()\n'
                                                                                                      "conn.execute('PRAGMA "
                                                                                                      "wal_checkpoint(TRUNCATE)')\n"
                                                                                                      'conn.commit(); '
                                                                                                      'conn.close()\n'
                                                                                                      "print('ok')\n"
                                                                                                      'HEOF',
                                                                                           'shell': True}},
                                                                     {   'type': 'launch',
                                                                         'parameters': {   'command': [   'google-chrome',
                                                                                                          '--remote-debugging-port=1337']}},
                                                                     {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                'config_override': [   {   'type': 'launch',
                                                                           'parameters': {   'command': [   'google-chrome',
                                                                                                            '--remote-debugging-port=1337']}},
                                                                       {   'type': 'launch',
                                                                           'parameters': {   'command': [   'socat',
                                                                                                            'tcp-listen:9222,fork',
                                                                                                            'tcp:localhost:1337']}}]},
    '47543840-672a-467d-80df-8f7c3b9788c9': {},
    '59155008-fe71-45ec-8a8f-dc35497b6aa8': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.babycenter.com/baby-names/details/carl-853']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '6766f2b8-8a72-417f-a9e5-56fcaa735837': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome '
                                                                                                '2>/dev/null; sleep 3; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'cd /home/user/Desktop '
                                                                                                '&& unzip -o '
                                                                                                'helloExtension.zip -d '
                                                                                                '. 2>/dev/null || true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os, '
                                                                                                'hashlib\n'
                                                                                                '\n'
                                                                                                '# Read existing '
                                                                                                'preferences\n'
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
                                                                                                '        with '
                                                                                                'open(path) as f:\n'
                                                                                                '            try:\n'
                                                                                                '                prefs '
                                                                                                '= json.load(f)\n'
                                                                                                '            except:\n'
                                                                                                '                prefs '
                                                                                                '= {}\n'
                                                                                                '\n'
                                                                                                '    ext_path = '
                                                                                                "'/home/user/Desktop/helloExtension'\n"
                                                                                                '\n'
                                                                                                '    # Read manifest\n'
                                                                                                '    manifest = '
                                                                                                '{"name": '
                                                                                                '"helloExtension", '
                                                                                                '"version": "1.0", '
                                                                                                '"manifest_version": '
                                                                                                '3}\n'
                                                                                                '    mpath = '
                                                                                                'os.path.join(ext_path, '
                                                                                                "'manifest.json')\n"
                                                                                                '    if '
                                                                                                'os.path.exists(mpath):\n'
                                                                                                '        with '
                                                                                                'open(mpath) as f:\n'
                                                                                                '            try:\n'
                                                                                                '                '
                                                                                                'manifest = '
                                                                                                'json.load(f)\n'
                                                                                                '            except:\n'
                                                                                                '                pass\n'
                                                                                                '\n'
                                                                                                '    # Generate '
                                                                                                'extension ID from '
                                                                                                'path (matching '
                                                                                                "Chrome's algorithm)\n"
                                                                                                '    path_hash = '
                                                                                                'hashlib.sha256(ext_path.encode()).hexdigest()[:32]\n'
                                                                                                '    ext_id = '
                                                                                                "''.join(chr(ord('a') "
                                                                                                '+ int(c, 16)) for c '
                                                                                                'in path_hash)\n'
                                                                                                '\n'
                                                                                                '    '
                                                                                                "prefs.setdefault('extensions', "
                                                                                                "{}).setdefault('settings', "
                                                                                                '{})[ext_id] = {\n'
                                                                                                '        '
                                                                                                "'active_permissions': "
                                                                                                "{'api': [], "
                                                                                                "'explicit_host': [], "
                                                                                                "'manifest_permissions': "
                                                                                                '[], '
                                                                                                "'scriptable_host': "
                                                                                                '[]},\n'
                                                                                                '        '
                                                                                                "'creation_flags': 1,\n"
                                                                                                '        '
                                                                                                "'from_webstore': "
                                                                                                'False,\n'
                                                                                                '        '
                                                                                                "'granted_permissions': "
                                                                                                "{'api': [], "
                                                                                                "'explicit_host': [], "
                                                                                                "'manifest_permissions': "
                                                                                                '[], '
                                                                                                "'scriptable_host': "
                                                                                                '[]},\n'
                                                                                                '        '
                                                                                                "'install_time': "
                                                                                                "'13349226702110891',\n"
                                                                                                "        'location': "
                                                                                                '4,\n'
                                                                                                "        'manifest': "
                                                                                                'manifest,\n'
                                                                                                "        'path': "
                                                                                                'ext_path,\n'
                                                                                                "        'state': 1,\n"
                                                                                                '        '
                                                                                                "'was_installed_by_default': "
                                                                                                'False,\n'
                                                                                                '    }\n'
                                                                                                '\n'
                                                                                                '    with open(path, '
                                                                                                "'w') as f:\n"
                                                                                                '        '
                                                                                                'json.dump(prefs, f)\n'
                                                                                                "    print(f'Updated "
                                                                                                "{path}')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '6c4c23a1-42a4-43cc-9db1-2f86ff3738cc': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install '
                                                                                                'websocket-client '
                                                                                                'requests 2>/dev/null; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'from datetime import '
                                                                                                'datetime\n'
                                                                                                'today = '
                                                                                                'datetime.now()\n'
                                                                                                'nm = today.month + 1\n'
                                                                                                'yr = today.year\n'
                                                                                                'if nm > 12:\n'
                                                                                                '    nm = 1; yr += 1\n'
                                                                                                'from datetime import '
                                                                                                'datetime as dt\n'
                                                                                                'target = dt(yr, nm, '
                                                                                                '5)\n'
                                                                                                'date_str = '
                                                                                                "target.strftime('%a, "
                                                                                                "%b %d, %Y')\n"
                                                                                                "html = ('<!doctype "
                                                                                                "html><html><body>'\n"
                                                                                                "        f'<div "
                                                                                                'class="mach-flight-context-info__wrapper--date">{date_str}</div>\'\n'
                                                                                                "        '<div "
                                                                                                'class="mach-flight-context-info__wrapper__info--separator">SEA<span>sep</span>NYC</div>\'\n'
                                                                                                "        '<div "
                                                                                                'class="mach-global-tabs-small__wrapper__tab--active">Miles</div>\'\n'
                                                                                                '        '
                                                                                                "'</body></html>')\n"
                                                                                                'with '
                                                                                                "open('/tmp/delta_6c.html', "
                                                                                                "'w') as f:\n"
                                                                                                '    f.write(html)\n'
                                                                                                'with '
                                                                                                "open('/tmp/chrome_url.txt', "
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                "f.write('file:///tmp/delta_6c.html')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome '
                                                                                                '2>/dev/null; pkill -f '
                                                                                                'socat 2>/dev/null; '
                                                                                                'sleep 2',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'socat '
                                                                                                'tcp-listen:9222,fork,reuseaddr '
                                                                                                'tcp:localhost:1337',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'google-chrome '
                                                                                                '--no-sandbox '
                                                                                                '--remote-debugging-port=1337 '
                                                                                                'about:blank '
                                                                                                '--user-data-dir=/home/user/chrome-data '
                                                                                                '--remote-allow-origins=*',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'import json, time, '
                                                                                                'sys\n'
                                                                                                'try:\n'
                                                                                                '    import requests\n'
                                                                                                'except ImportError:\n'
                                                                                                '    import '
                                                                                                'urllib.request\n'
                                                                                                '    class requests:\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def get(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            class R:\n'
                                                                                                '                def '
                                                                                                'json(self):\n'
                                                                                                '                    '
                                                                                                'return '
                                                                                                'json.loads(urllib.request.urlopen(url, '
                                                                                                'timeout=timeout).read())\n'
                                                                                                '            return '
                                                                                                'R()\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def put(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            req = '
                                                                                                'urllib.request.Request(url, '
                                                                                                "method='PUT')\n"
                                                                                                '            '
                                                                                                'urllib.request.urlopen(req, '
                                                                                                'timeout=timeout)\n'
                                                                                                '            return '
                                                                                                'None\n'
                                                                                                'try:\n'
                                                                                                '    import websocket\n'
                                                                                                'except ImportError:\n'
                                                                                                "    print('ERROR: "
                                                                                                'websocket not '
                                                                                                "available')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                'url = '
                                                                                                "open('/tmp/chrome_url.txt').read().strip()\n"
                                                                                                "print(f'Target URL: "
                                                                                                "{url}')\n"
                                                                                                'for attempt in '
                                                                                                'range(30):\n'
                                                                                                '    try:\n'
                                                                                                '        tabs = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                '        pages = [t '
                                                                                                'for t in tabs if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                '        if pages:\n'
                                                                                                '            break\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                '    try:\n'
                                                                                                '        '
                                                                                                "requests.put('http://localhost:1337/json/new?about:blank', "
                                                                                                'timeout=5)\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                '    time.sleep(2)\n'
                                                                                                'else:\n'
                                                                                                "    print('ERROR: No "
                                                                                                'page tabs found after '
                                                                                                "30 attempts')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                'page = pages[0]\n'
                                                                                                'ws = '
                                                                                                "websocket.create_connection(page['webSocketDebuggerUrl'])\n"
                                                                                                "ws.send(json.dumps({'id': "
                                                                                                "1, 'method': "
                                                                                                "'Page.navigate', "
                                                                                                "'params': {'url': "
                                                                                                'url}}))\n'
                                                                                                'result = '
                                                                                                'json.loads(ws.recv())\n'
                                                                                                'ws.close()\n'
                                                                                                "print(f'Navigated: "
                                                                                                "{result}')\n"
                                                                                                'time.sleep(2)\n'
                                                                                                'tabs2 = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                'pages2 = [t for t in '
                                                                                                'tabs2 if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                'for extra in '
                                                                                                'pages2[1:]:\n'
                                                                                                '    try:\n'
                                                                                                '        eid = '
                                                                                                "extra['id']\n"
                                                                                                '        '
                                                                                                "requests.get(f'http://localhost:1337/json/close/{eid}', "
                                                                                                'timeout=3)\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                "print('Done')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '7a5a7856-f1b6-42a4-ade9-1ca81ca0f263': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/chrome-data/Default',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/chrome-data/Default/Bookmarks'\n"
                                                                                                'bookmarks = {"roots": '
                                                                                                '{"bookmark_bar": '
                                                                                                '{"children": [], '
                                                                                                '"name": "Bookmarks '
                                                                                                'bar", "type": '
                                                                                                '"folder"}, "other": '
                                                                                                '{"children": [], '
                                                                                                '"name": "Other '
                                                                                                'bookmarks", "type": '
                                                                                                '"folder"}, "synced": '
                                                                                                '{"children": [], '
                                                                                                '"name": "Mobile '
                                                                                                'bookmarks", "type": '
                                                                                                '"folder"}}, '
                                                                                                '"version": 1}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        bookmarks = '
                                                                                                'json.load(f)\n'
                                                                                                '\n'
                                                                                                'bar = '
                                                                                                'bookmarks.setdefault("roots", '
                                                                                                '{}).setdefault("bookmark_bar", '
                                                                                                '{"children": [], '
                                                                                                '"name": "Bookmarks '
                                                                                                'bar", "type": '
                                                                                                '"folder"})\n'
                                                                                                'children = '
                                                                                                'bar.setdefault("children", '
                                                                                                '[])\n'
                                                                                                'url = '
                                                                                                '"https://jalammar.github.io/illustrated-transformer/"\n'
                                                                                                '# Check if already '
                                                                                                'bookmarked\n'
                                                                                                'if not '
                                                                                                'any(c.get("url") == '
                                                                                                'url for c in '
                                                                                                'children):\n'
                                                                                                '    '
                                                                                                'children.append({\n'
                                                                                                '        "date_added": '
                                                                                                '"13365000000000000",\n'
                                                                                                '        '
                                                                                                '"date_last_used": '
                                                                                                '"0",\n'
                                                                                                '        "guid": '
                                                                                                '"00000000-0000-0000-0000-000000000002",\n'
                                                                                                '        "id": "101",\n'
                                                                                                '        "name": "The '
                                                                                                'Illustrated '
                                                                                                'Transformer",\n'
                                                                                                '        "type": '
                                                                                                '"url",\n'
                                                                                                '        "url": url\n'
                                                                                                '    })\n'
                                                                                                '\n'
                                                                                                '# Evaluator checks '
                                                                                                'set(urls) == '
                                                                                                'set(expected_urls), '
                                                                                                'so only this URL '
                                                                                                'should be a url-type '
                                                                                                'child\n'
                                                                                                'bar["children"] = [c '
                                                                                                'for c in children if '
                                                                                                'c.get("type") == '
                                                                                                '"folder"] + '
                                                                                                '[{"date_added": '
                                                                                                '"13365000000000000", '
                                                                                                '"date_last_used": '
                                                                                                '"0", "guid": '
                                                                                                '"00000000-0000-0000-0000-000000000002", '
                                                                                                '"id": "101", "name": '
                                                                                                '"The Illustrated '
                                                                                                'Transformer", "type": '
                                                                                                '"url", "url": url}]\n'
                                                                                                '\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    '
                                                                                                'json.dump(bookmarks, '
                                                                                                'f, indent=2)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '7b6c7e24-c58a-49fc-a5bb-d57b80e5b4c3': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import sqlite3, os\n'
                                                                                                'db_path = '
                                                                                                "'/home/user/chrome-data/Default/Cookies'\n"
                                                                                                'os.makedirs(os.path.dirname(db_path), '
                                                                                                'exist_ok=True)\n'
                                                                                                'conn = '
                                                                                                'sqlite3.connect(db_path)\n'
                                                                                                'c = conn.cursor()\n'
                                                                                                '# Create cookies '
                                                                                                'table if not exists '
                                                                                                '(Chrome schema)\n'
                                                                                                "c.execute('''CREATE "
                                                                                                'TABLE IF NOT EXISTS '
                                                                                                'cookies (\n'
                                                                                                '  creation_utc '
                                                                                                'INTEGER NOT NULL, '
                                                                                                'host_key TEXT NOT '
                                                                                                'NULL,\n'
                                                                                                '  name TEXT NOT NULL, '
                                                                                                'value TEXT NOT NULL, '
                                                                                                'path TEXT NOT NULL,\n'
                                                                                                '  expires_utc INTEGER '
                                                                                                'NOT NULL, is_secure '
                                                                                                'INTEGER NOT NULL,\n'
                                                                                                '  is_httponly INTEGER '
                                                                                                'NOT NULL, '
                                                                                                'last_access_utc '
                                                                                                'INTEGER NOT NULL,\n'
                                                                                                '  has_expires INTEGER '
                                                                                                'NOT NULL DEFAULT 1, '
                                                                                                'is_persistent INTEGER '
                                                                                                'NOT NULL DEFAULT 1,\n'
                                                                                                '  priority INTEGER '
                                                                                                'NOT NULL DEFAULT 1, '
                                                                                                'encrypted_value BLOB '
                                                                                                "DEFAULT '',\n"
                                                                                                '  samesite INTEGER '
                                                                                                'NOT NULL DEFAULT -1, '
                                                                                                'source_scheme INTEGER '
                                                                                                'NOT NULL DEFAULT 0,\n'
                                                                                                '  source_port INTEGER '
                                                                                                'NOT NULL DEFAULT -1, '
                                                                                                'is_same_party INTEGER '
                                                                                                'NOT NULL DEFAULT 0,\n'
                                                                                                '  UNIQUE (host_key, '
                                                                                                "name, path))''')\n"
                                                                                                'c.execute("DELETE '
                                                                                                'FROM cookies WHERE '
                                                                                                'host_key LIKE '
                                                                                                '\'%amazon.com%\'")\n'
                                                                                                'conn.commit()\n'
                                                                                                'conn.close()\n'
                                                                                                "print('ok')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'config_append': [   {   'type': 'execute',
                                                                         'parameters': {   'command': 'pkill -f '
                                                                                                      "'google-chrome' "
                                                                                                      '2>/dev/null; '
                                                                                                      'sleep 2; true',
                                                                                           'shell': True}},
                                                                     {   'type': 'execute',
                                                                         'parameters': {   'command': 'python3 << '
                                                                                                      "'HEOF'\n"
                                                                                                      'import sqlite3, '
                                                                                                      'datetime, os\n'
                                                                                                      'path = '
                                                                                                      "'/home/user/chrome-data/Default/Cookies'\n"
                                                                                                      'if not '
                                                                                                      'os.path.exists(path):\n'
                                                                                                      "    print('no "
                                                                                                      "cookies db'); "
                                                                                                      'exit(0)\n'
                                                                                                      'now = '
                                                                                                      'int((datetime.datetime.now()-datetime.datetime(1601,1,1)).total_seconds()*1e6)\n'
                                                                                                      'exp = now + '
                                                                                                      '365*24*3600*10**6\n'
                                                                                                      'conn = '
                                                                                                      'sqlite3.connect(path, '
                                                                                                      'timeout=5)\n'
                                                                                                      'cur = '
                                                                                                      'conn.cursor()\n'
                                                                                                      '# Chrome 147 '
                                                                                                      'schema: '
                                                                                                      'top_frame_site_key, '
                                                                                                      'last_update_utc, '
                                                                                                      'source_type, '
                                                                                                      'has_cross_site_ancestor\n'
                                                                                                      '# UNIQUE INDEX: '
                                                                                                      '(host_key, '
                                                                                                      'top_frame_site_key, '
                                                                                                      'has_cross_site_ancestor, '
                                                                                                      'name, path, '
                                                                                                      'source_scheme, '
                                                                                                      'source_port)\n'
                                                                                                      "cur.execute('INSERT "
                                                                                                      'OR IGNORE INTO '
                                                                                                      'cookies '
                                                                                                      '(creation_utc,host_key,top_frame_site_key,name,value,encrypted_value,path,expires_utc,is_secure,is_httponly,last_access_utc,has_expires,is_persistent,priority,samesite,source_scheme,source_port,last_update_utc,source_type,has_cross_site_ancestor) '
                                                                                                      'VALUES '
                                                                                                      "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', "
                                                                                                      "(now,'.amazon.com','','session-id','123-4567890-1234567',b'','/',exp,1,1,now,1,1,1,0,1,443,now,0,0))\n"
                                                                                                      'conn.commit()\n'
                                                                                                      "conn.execute('PRAGMA "
                                                                                                      "wal_checkpoint(TRUNCATE)')\n"
                                                                                                      'conn.commit(); '
                                                                                                      'conn.close()\n'
                                                                                                      "print('ok')\n"
                                                                                                      'HEOF',
                                                                                           'shell': True}},
                                                                     {   'type': 'launch',
                                                                         'parameters': {   'command': [   'google-chrome',
                                                                                                          '--remote-debugging-port=1337']}},
                                                                     {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '7f52cab9-535c-4835-ac8c-391ee64dc930': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'shell': True,
                                                                                     'command': 'cat > /tmp/drip.html '
                                                                                                "<<'EOF'\n"
                                                                                                '<html><body>\n'
                                                                                                '<div '
                                                                                                'class="fT28tf">Black</div>\n'
                                                                                                '<div '
                                                                                                'class="fT28tf">$25 - '
                                                                                                '$60</div>\n'
                                                                                                '<div '
                                                                                                'class="fT28tf">On '
                                                                                                'sale</div>\n'
                                                                                                '</body></html>\n'
                                                                                                'EOF'}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    'file:///tmp/drip.html?q=drip+coffee+maker']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                'after_postconfig': True},
    '82279c77-8fc6-46f6-9622-3ba96f61b477': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.cars.com/shopping/results/?list_price_max=50000&maximum_distance=50&zip=10001&fuel_slugs[]=electric']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '82bc8d6a-36eb-4d2d-8801-ef714fb1e55a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'from datetime import '
                                                                                                'datetime, timedelta\n'
                                                                                                'now = datetime.now()\n'
                                                                                                'days_until_monday = '
                                                                                                '(7 - now.weekday()) % '
                                                                                                '7\n'
                                                                                                'if days_until_monday '
                                                                                                '== 0:\n'
                                                                                                '    days_until_monday '
                                                                                                '= 7\n'
                                                                                                'mon = now + '
                                                                                                'timedelta(days=days_until_monday)\n'
                                                                                                'date_str = '
                                                                                                "f'{mon.year}-{mon.month:02d}-{mon.day:02d}'\n"
                                                                                                'with '
                                                                                                "open('/tmp/chrome_date.txt', "
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                'f.write(date_str)\n'
                                                                                                "print(f'Next Monday: "
                                                                                                "{date_str}')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "echo '<html></html>' "
                                                                                                '> /tmp/blank_qa.html',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'DATE=$(cat '
                                                                                                '/tmp/chrome_date.txt) '
                                                                                                '&& nohup '
                                                                                                'google-chrome '
                                                                                                '--no-sandbox '
                                                                                                '--user-data-dir=/home/user/chrome-data '
                                                                                                '--remote-debugging-port=1337 '
                                                                                                '--remote-allow-origins=* '
                                                                                                '--test-type '
                                                                                                '--no-default-browser-check '
                                                                                                '--password-store=basic '
                                                                                                '--disable-dev-shm-usage '
                                                                                                '--start-maximized '
                                                                                                '"file:///tmp/blank_qa.html?fromStation=BOM&toStation=ARN&departing=$DATE" '
                                                                                                '>/dev/null 2>&1 &',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    '93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/chrome-data/Default',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/chrome-data/Default/Preferences'\n"
                                                                                                'prefs = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        prefs = '
                                                                                                'json.load(f)\n'
                                                                                                '# Set color_scheme=1 '
                                                                                                '(light) and remove '
                                                                                                'color_scheme2 so '
                                                                                                'evaluator reads '
                                                                                                "'light'\n"
                                                                                                "prefs.setdefault('browser', "
                                                                                                "{}).setdefault('theme', "
                                                                                                "{})['color_scheme'] = "
                                                                                                '1\n'
                                                                                                "prefs.get('browser', "
                                                                                                "{}).get('theme', "
                                                                                                "{}).pop('color_scheme2', "
                                                                                                'None)\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(prefs, '
                                                                                                'f)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'config_append': [   {   'type': 'execute',
                                                                         'parameters': {   'command': 'python3 << '
                                                                                                      "'HEOF'\n"
                                                                                                      'import json, '
                                                                                                      'os\n'
                                                                                                      'path = '
                                                                                                      "'/home/user/chrome-data/Default/Preferences'\n"
                                                                                                      'if '
                                                                                                      'os.path.exists(path):\n'
                                                                                                      '    with '
                                                                                                      'open(path) as '
                                                                                                      'f:\n'
                                                                                                      '        prefs = '
                                                                                                      'json.load(f)\n'
                                                                                                      '    '
                                                                                                      "prefs.get('browser', "
                                                                                                      "{}).get('theme', "
                                                                                                      "{}).pop('color_scheme2', "
                                                                                                      'None)\n'
                                                                                                      '    with '
                                                                                                      "open(path, 'w') "
                                                                                                      'as f:\n'
                                                                                                      '        '
                                                                                                      'json.dump(prefs, '
                                                                                                      'f)\n'
                                                                                                      "print('ok')\n"
                                                                                                      'HEOF',
                                                                                           'shell': True}}],
                                                'evaluator': {   'postconfig': [   {   'type': 'sleep',
                                                                                       'parameters': {'seconds': 5}}],
                                                                 'func': 'match_in_list',
                                                                 'result': {'type': 'chrome_color_scheme'},
                                                                 'expected': {   'type': 'rule',
                                                                                 'rules': {   'expected': [   'light',
                                                                                                              'system']}}}},
    '9656a811-9b5b-4ddf-99c7-5117bcef0626': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user/.config/google-chrome/Default'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/google-chrome/Default/Preferences'\n"
                                                                                                'prefs = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        prefs = '
                                                                                                'json.load(f)\n'
                                                                                                'prefs.setdefault("safebrowsing", '
                                                                                                '{})\n'
                                                                                                'prefs["safebrowsing"]["enabled"] '
                                                                                                '= True\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(prefs, '
                                                                                                'f)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '99146c54-4f37-4ab8-9327-5f3291665e1e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/chrome-data/Default',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/chrome-data/Default/Preferences'\n"
                                                                                                'prefs = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        prefs = '
                                                                                                'json.load(f)\n'
                                                                                                '\n'
                                                                                                "prefs.setdefault('profile', {}).setdefault('default_content_setting_values', {})['cookies'] = 4\n"
                                                                                                '\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(prefs, '
                                                                                                'f)\n'
                                                                                                'print("Set profile.default_content_setting_values.cookies = 4")\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '9f3f70fc-5afc-4958-a7b7-3bb4fcb01805': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'shell': True,
                                                                                     'command': "mkdir -p '/tmp/over "
                                                                                                "$60-women-jerseys-nike' "
                                                                                                "&& cat > '/tmp/over "
                                                                                                "$60-women-jerseys-nike/p.html' "
                                                                                                "<<'EOF'\n"
                                                                                                '<html><body>\n'
                                                                                                '<div '
                                                                                                'class="filter-selector-link">over '
                                                                                                '$60</div>\n'
                                                                                                '<div '
                                                                                                'class="filter-selector-link">women</div>\n'
                                                                                                '<div '
                                                                                                'class="filter-selector-link">jerseys</div>\n'
                                                                                                '<div '
                                                                                                'class="filter-selector-link">nike</div>\n'
                                                                                                '</body></html>\n'
                                                                                                'EOF'}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    'file:///tmp/over '
                                                                                                    '$60-women-jerseys-nike/p.html']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                'after_postconfig': True},
    '9f935cce-0a9f-435f-8007-817732bfc0a5': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.justice.gov/forms?title=&field_component_target_id=431']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'a728a36e-8bf1-4bb6-9a03-ef039a5233f0': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.dmv.virginia.gov/licenses-ids/license/applying/eligibility']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'a96b564e-dbe9-42c3-9ccf-b4498073438a': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://discussions.flightaware.com/t/the-banter-thread/4412']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'af630914-714e-4a24-a7bb-f9af687d3b91': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/chrome-data/Default',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/chrome-data/Default/Preferences'\n"
                                                                                                'prefs = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        prefs = '
                                                                                                'json.load(f)\n'
                                                                                                '\n'
                                                                                                'def deep_merge(base, '
                                                                                                'patch):\n'
                                                                                                '    for k, v in '
                                                                                                'patch.items():\n'
                                                                                                '        if '
                                                                                                'isinstance(v, dict) '
                                                                                                'and '
                                                                                                'isinstance(base.get(k), '
                                                                                                'dict):\n'
                                                                                                '            '
                                                                                                'deep_merge(base[k], '
                                                                                                'v)\n'
                                                                                                '        else:\n'
                                                                                                '            base[k] = '
                                                                                                'v\n'
                                                                                                '\n'
                                                                                                'deep_merge(prefs, '
                                                                                                '{"webkit": '
                                                                                                '{"webprefs": '
                                                                                                '{"default_font_size": '
                                                                                                '24}}})\n'
                                                                                                '\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(prefs, '
                                                                                                'f)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    'b070486d-e161-459b-aa2b-ef442d973b92': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.drugs.com/sfx/tamiflu-side-effects.html']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'b4f95342-463e-4179-8c3f-193cd7241fb2': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'about:blank']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                'exclude_reason': 'trivial_pass:color_precheck'},
    'b7895e80-f4d1-4648-bee0-4eb45a6f1fa8': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'from datetime import '
                                                                                                'datetime, timedelta\n'
                                                                                                'try:\n'
                                                                                                '    from zoneinfo '
                                                                                                'import ZoneInfo\n'
                                                                                                '    now = '
                                                                                                "datetime.now(ZoneInfo('America/New_York'))\n"
                                                                                                'except ImportError:\n'
                                                                                                '    now = '
                                                                                                'datetime.utcnow() - '
                                                                                                'timedelta(hours=4)\n'
                                                                                                '# "next week Friday" '
                                                                                                '= Monday of next week '
                                                                                                '+ 4 days\n'
                                                                                                'days_to_next_monday = '
                                                                                                '7 - now.weekday()  # '
                                                                                                '0=Mon so Mon→7, '
                                                                                                'Tue→6, ... Sun→1\n'
                                                                                                'fri = now + '
                                                                                                'timedelta(days=days_to_next_monday '
                                                                                                '+ 4)\n'
                                                                                                'sun = fri + '
                                                                                                'timedelta(days=2)\n'
                                                                                                'fri_str = '
                                                                                                "fri.strftime('%a, %b "
                                                                                                "%d')\n"
                                                                                                'sun_str = '
                                                                                                "sun.strftime('%a, %b "
                                                                                                "%d')\n"
                                                                                                "html = ('<!DOCTYPE "
                                                                                                "html><html><body>'\n"
                                                                                                "        f'<button "
                                                                                                'data-automation="checkin"><div '
                                                                                                'class="Wh">Check '
                                                                                                "In{fri_str}</div></button>'\n"
                                                                                                "        f'<button "
                                                                                                'data-automation="checkout"><div '
                                                                                                'class="Wh">Check '
                                                                                                "Out{sun_str}</div></button>'\n"
                                                                                                "        '<h2 "
                                                                                                'data-automation="header_geo_title">New '
                                                                                                'York City '
                                                                                                "Hotels</h2>'\n"
                                                                                                "        '<button "
                                                                                                'data-automation="roomsandguests"><div '
                                                                                                'class="Wh">Rooms/Guests1 '
                                                                                                'Room, 2 '
                                                                                                "Guests</div></button>'\n"
                                                                                                "        '<button "
                                                                                                'aria-label="PRICE_LOW_TO_HIGH: '
                                                                                                'Price (low to '
                                                                                                'high)"><div '
                                                                                                'class="biGQs '
                                                                                                'SewaP">Price (low to '
                                                                                                "high)</div></button>'\n"
                                                                                                '        '
                                                                                                "'</body></html>')\n"
                                                                                                'import os\n'
                                                                                                "os.makedirs('/tmp/ta', "
                                                                                                'exist_ok=True)\n'
                                                                                                'with '
                                                                                                "open('/tmp/ta/page.html', "
                                                                                                "'w') as f:\n"
                                                                                                '    f.write(html)\n'
                                                                                                "print(f'Dates: "
                                                                                                '{fri_str} to '
                                                                                                "{sun_str}')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'file:///tmp/ta/page.html']}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'socat',
                                                                                                    'tcp-listen:9222,fork',
                                                                                                    'tcp:localhost:1337']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'bb5e4c0d-f964-439c-97b6-bdb9747de3f4': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                '/home/user/chrome-data/Default',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                'import json, os\n'
                                                                                                'path = '
                                                                                                "'/home/user/chrome-data/Default/Preferences'\n"
                                                                                                'prefs = {}\n'
                                                                                                'if '
                                                                                                'os.path.exists(path):\n'
                                                                                                '    with open(path) '
                                                                                                'as f:\n'
                                                                                                '        prefs = '
                                                                                                'json.load(f)\n'
                                                                                                '\n'
                                                                                                'def deep_merge(base, '
                                                                                                'patch):\n'
                                                                                                '    for k, v in '
                                                                                                'patch.items():\n'
                                                                                                '        if '
                                                                                                'isinstance(v, dict) '
                                                                                                'and '
                                                                                                'isinstance(base.get(k), '
                                                                                                'dict):\n'
                                                                                                '            '
                                                                                                'deep_merge(base[k], '
                                                                                                'v)\n'
                                                                                                '        else:\n'
                                                                                                '            base[k] = '
                                                                                                'v\n'
                                                                                                '\n'
                                                                                                'deep_merge(prefs, '
                                                                                                '{"default_search_provider_data": '
                                                                                                '{"template_url_data": '
                                                                                                '{"short_name": '
                                                                                                '"Bing", "keyword": '
                                                                                                '"bing.com", "url": '
                                                                                                '"https://www.bing.com/search?q={searchTerms}"}}})\n'
                                                                                                '\n'
                                                                                                "with open(path, 'w') "
                                                                                                'as f:\n'
                                                                                                '    json.dump(prefs, '
                                                                                                'f)\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    'c1fa57f3-c3db-4596-8f09-020701085416': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.united.com/en/us/checked-bag-fee-calculator']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'cabb3bae-cccb-41bd-9f5d-0f3a9fecd825': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    'https://example.com/?a=Spider-Man&b=Toys&c=Kids&S=4']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                'after_postconfig': True},
    'da46d875-6b82-4681-9284-653b0c7ae241': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install '
                                                                                                'websocket-client '
                                                                                                'requests 2>/dev/null; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'from datetime import '
                                                                                                'datetime\n'
                                                                                                'import calendar\n'
                                                                                                'now = datetime.now()\n'
                                                                                                '# Eight months later\n'
                                                                                                'y, m = now.year, '
                                                                                                'now.month + 8\n'
                                                                                                'while m > 12:\n'
                                                                                                '    m -= 12\n'
                                                                                                '    y += 1\n'
                                                                                                '# First Monday of '
                                                                                                'that month: weekday() '
                                                                                                '== 0 (Monday)\n'
                                                                                                'for day in range(1, '
                                                                                                '8):\n'
                                                                                                '    d = datetime(y, '
                                                                                                'm, day)\n'
                                                                                                '    if d.weekday() == '
                                                                                                '0:\n'
                                                                                                '        break\n'
                                                                                                "# Format: 'December "
                                                                                                "07, 10:15 AM' (full "
                                                                                                'month, zero-padded '
                                                                                                'day)\n'
                                                                                                'date_str = '
                                                                                                "d.strftime('%B %d, "
                                                                                                "10:15 AM')\n"
                                                                                                'import os\n'
                                                                                                "os.makedirs('/tmp/book/CharlieCardStoreAppointments@mbta.com', "
                                                                                                'exist_ok=True)\n'
                                                                                                "html = ('<!DOCTYPE "
                                                                                                "html><html><body>'\n"
                                                                                                "        '<div "
                                                                                                'id="pad1">pad</div>\'\n'
                                                                                                '        '
                                                                                                "'<div><div><form>'\n"
                                                                                                '        '
                                                                                                "'<div>f1</div><div>f2</div><div>f3</div><div>f4</div><div>f5</div><div>f6</div>'\n"
                                                                                                '        '
                                                                                                "'<div><div><div><div>'\n"
                                                                                                "        '<input "
                                                                                                'value="James Smith" '
                                                                                                "/>'\n"
                                                                                                "        '<input "
                                                                                                'value="james.smith@gmail.com" '
                                                                                                "/>'\n"
                                                                                                '        '
                                                                                                "'</div></div></div></div>'\n"
                                                                                                '        '
                                                                                                "'</form></div></div>'\n"
                                                                                                "        '<div "
                                                                                                'class="HAZ16"><h3>Apply '
                                                                                                'for Transportation '
                                                                                                'Access Pass (TAP) '
                                                                                                'CharlieCard non-auto '
                                                                                                "approval</h3></div>'\n"
                                                                                                "        f'<div "
                                                                                                'class="HAZ16"><h3>{date_str}</h3></div>\'\n'
                                                                                                '        '
                                                                                                "'</body></html>')\n"
                                                                                                'with '
                                                                                                "open('/tmp/book/CharlieCardStoreAppointments@mbta.com/page.html', "
                                                                                                "'w') as f:\n"
                                                                                                '    f.write(html)\n'
                                                                                                'with '
                                                                                                "open('/tmp/chrome_url.txt', "
                                                                                                "'w') as f:\n"
                                                                                                '    '
                                                                                                "f.write('file:///tmp/book/CharlieCardStoreAppointments@mbta.com/page.html')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome '
                                                                                                '2>/dev/null; pkill -f '
                                                                                                'socat 2>/dev/null; '
                                                                                                'sleep 2',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'socat '
                                                                                                'tcp-listen:9222,fork,reuseaddr '
                                                                                                'tcp:localhost:1337',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'google-chrome '
                                                                                                '--no-sandbox '
                                                                                                '--remote-debugging-port=1337 '
                                                                                                'about:blank '
                                                                                                '--user-data-dir=/home/user/chrome-data '
                                                                                                '--remote-allow-origins=*',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'import json, time, '
                                                                                                'sys\n'
                                                                                                'try:\n'
                                                                                                '    import requests\n'
                                                                                                'except ImportError:\n'
                                                                                                '    import '
                                                                                                'urllib.request\n'
                                                                                                '    class requests:\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def get(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            class R:\n'
                                                                                                '                def '
                                                                                                'json(self):\n'
                                                                                                '                    '
                                                                                                'return '
                                                                                                'json.loads(urllib.request.urlopen(url, '
                                                                                                'timeout=timeout).read())\n'
                                                                                                '            return '
                                                                                                'R()\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def put(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            req = '
                                                                                                'urllib.request.Request(url, '
                                                                                                "method='PUT')\n"
                                                                                                '            '
                                                                                                'urllib.request.urlopen(req, '
                                                                                                'timeout=timeout)\n'
                                                                                                '            return '
                                                                                                'None\n'
                                                                                                'try:\n'
                                                                                                '    import websocket\n'
                                                                                                'except ImportError:\n'
                                                                                                "    print('ERROR: "
                                                                                                'websocket not '
                                                                                                "available')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                'url = '
                                                                                                "open('/tmp/chrome_url.txt').read().strip()\n"
                                                                                                "print(f'Target URL: "
                                                                                                "{url}')\n"
                                                                                                'for attempt in '
                                                                                                'range(30):\n'
                                                                                                '    try:\n'
                                                                                                '        tabs = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                '        pages = [t '
                                                                                                'for t in tabs if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                '        if pages:\n'
                                                                                                '            break\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                '    try:\n'
                                                                                                '        '
                                                                                                "requests.put('http://localhost:1337/json/new?about:blank', "
                                                                                                'timeout=5)\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                '    time.sleep(2)\n'
                                                                                                'else:\n'
                                                                                                "    print('ERROR: No "
                                                                                                'page tabs found after '
                                                                                                "30 attempts')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                'page = pages[0]\n'
                                                                                                'ws = '
                                                                                                "websocket.create_connection(page['webSocketDebuggerUrl'])\n"
                                                                                                "ws.send(json.dumps({'id': "
                                                                                                "1, 'method': "
                                                                                                "'Page.navigate', "
                                                                                                "'params': {'url': "
                                                                                                'url}}))\n'
                                                                                                'result = '
                                                                                                'json.loads(ws.recv())\n'
                                                                                                'ws.close()\n'
                                                                                                "print(f'Navigated: "
                                                                                                "{result}')\n"
                                                                                                'time.sleep(2)\n'
                                                                                                'tabs2 = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                'pages2 = [t for t in '
                                                                                                'tabs2 if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                'for extra in '
                                                                                                'pages2[1:]:\n'
                                                                                                '    try:\n'
                                                                                                '        eid = '
                                                                                                "extra['id']\n"
                                                                                                '        '
                                                                                                "requests.get(f'http://localhost:1337/json/close/{eid}', "
                                                                                                'timeout=3)\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                "print('Done')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'e1e75309-3ddb-4d09-92ec-de869c928143': {},
    'f0b971a1-6831-4b9b-a50e-22a6e47f45ba': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.nfl.com/scores/2019/super-bowl-sunday']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}],
                                                'evaluator': {   'func': 'is_expected_active_tab',
                                                                 'result': {   'type': 'active_url_from_accessTree',
                                                                               'goto_prefix': 'https://www.'},
                                                                 'expected': {   'type': 'rule',
                                                                                 'rules': {   'type': 'url',
                                                                                              'url': 'https://www.nfl.com/scores/2019/super-bowl-sunday'}}}},
    'f3b19d1e-2d48-44e9-b4e1-defcae1a0197': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://premier.ticketek.com.au/content/buyer/Ticket-Delivery-FAQs.aspx']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'f5d96daf-83a8-4c86-9686-bada31fc66ab': {   'actions': [   {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--no-sandbox',
                                                                                                    '--remote-debugging-port=1337',
                                                                                                    '--user-data-dir=/home/user/chrome-data',
                                                                                                    '--remote-allow-origins=*',
                                                                                                    'https://www.gsmarena.com/compare.php3?modelList=iphone-15-pro-max,iphone-14-pro-max,iphone-13-pro-max']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'f79439ad-3ee8-4f99-a518-0eb60e5652b0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'pip install '
                                                                                                'websocket-client '
                                                                                                'requests 2>/dev/null; '
                                                                                                'true',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'from datetime import '
                                                                                                'datetime\n'
                                                                                                'today = '
                                                                                                'datetime.now()\n'
                                                                                                'nm = today.month + 1\n'
                                                                                                'yr = today.year\n'
                                                                                                'if nm > 12:\n'
                                                                                                '    nm = 1; yr += 1\n'
                                                                                                'date_str = '
                                                                                                "f'{yr}-{nm:02d}-10'\n"
                                                                                                'url = '
                                                                                                "(f'https://www.ryanair.com/gb/en/trip/flights/select'\n"
                                                                                                '       '
                                                                                                "f'?adults=2&teens=0&children=0&infants=0'\n"
                                                                                                '       '
                                                                                                "f'&dateOut={date_str}&isReturn=false'\n"
                                                                                                '       '
                                                                                                "f'&originIata=DUB&destinationIata=VIE'\n"
                                                                                                '       '
                                                                                                "f'&tpAdults=2&tpTeens=0&tpChildren=0'\n"
                                                                                                '       '
                                                                                                "f'&tpStartDate={date_str}&tpEndDate={date_str}')\n"
                                                                                                'with '
                                                                                                "open('/tmp/chrome_url.txt', "
                                                                                                "'w') as f:\n"
                                                                                                '    f.write(url)\n'
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'pkill -9 -f chrome '
                                                                                                '2>/dev/null; pkill -f '
                                                                                                'socat 2>/dev/null; '
                                                                                                'sleep 2',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'socat '
                                                                                                'tcp-listen:9222,fork,reuseaddr '
                                                                                                'tcp:localhost:1337',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': 'google-chrome '
                                                                                                '--no-sandbox '
                                                                                                '--remote-debugging-port=1337 '
                                                                                                'about:blank '
                                                                                                '--user-data-dir=/home/user/chrome-data '
                                                                                                '--remote-allow-origins=*',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'import json, time, '
                                                                                                'sys\n'
                                                                                                'try:\n'
                                                                                                '    import requests\n'
                                                                                                'except ImportError:\n'
                                                                                                '    import '
                                                                                                'urllib.request\n'
                                                                                                '    class requests:\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def get(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            class R:\n'
                                                                                                '                def '
                                                                                                'json(self):\n'
                                                                                                '                    '
                                                                                                'return '
                                                                                                'json.loads(urllib.request.urlopen(url, '
                                                                                                'timeout=timeout).read())\n'
                                                                                                '            return '
                                                                                                'R()\n'
                                                                                                '        '
                                                                                                '@staticmethod\n'
                                                                                                '        def put(url, '
                                                                                                'timeout=5):\n'
                                                                                                '            req = '
                                                                                                'urllib.request.Request(url, '
                                                                                                "method='PUT')\n"
                                                                                                '            '
                                                                                                'urllib.request.urlopen(req, '
                                                                                                'timeout=timeout)\n'
                                                                                                '            return '
                                                                                                'None\n'
                                                                                                'try:\n'
                                                                                                '    import websocket\n'
                                                                                                'except ImportError:\n'
                                                                                                "    print('ERROR: "
                                                                                                'websocket not '
                                                                                                "available')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                'url = '
                                                                                                "open('/tmp/chrome_url.txt').read().strip()\n"
                                                                                                "print(f'Target URL: "
                                                                                                "{url}')\n"
                                                                                                'for attempt in '
                                                                                                'range(30):\n'
                                                                                                '    try:\n'
                                                                                                '        tabs = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                '        pages = [t '
                                                                                                'for t in tabs if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                '        if pages:\n'
                                                                                                '            break\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                '    try:\n'
                                                                                                '        '
                                                                                                "requests.put('http://localhost:1337/json/new?about:blank', "
                                                                                                'timeout=5)\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                '    time.sleep(2)\n'
                                                                                                'else:\n'
                                                                                                "    print('ERROR: No "
                                                                                                'page tabs found after '
                                                                                                "30 attempts')\n"
                                                                                                '    sys.exit(1)\n'
                                                                                                'page = pages[0]\n'
                                                                                                'ws = '
                                                                                                "websocket.create_connection(page['webSocketDebuggerUrl'])\n"
                                                                                                "ws.send(json.dumps({'id': "
                                                                                                "1, 'method': "
                                                                                                "'Page.navigate', "
                                                                                                "'params': {'url': "
                                                                                                'url}}))\n'
                                                                                                'result = '
                                                                                                'json.loads(ws.recv())\n'
                                                                                                'ws.close()\n'
                                                                                                "print(f'Navigated: "
                                                                                                "{result}')\n"
                                                                                                'time.sleep(2)\n'
                                                                                                'tabs2 = '
                                                                                                "requests.get('http://localhost:1337/json', "
                                                                                                'timeout=3).json()\n'
                                                                                                'pages2 = [t for t in '
                                                                                                'tabs2 if '
                                                                                                "t.get('type') == "
                                                                                                "'page']\n"
                                                                                                'for extra in '
                                                                                                'pages2[1:]:\n'
                                                                                                '    try:\n'
                                                                                                '        eid = '
                                                                                                "extra['id']\n"
                                                                                                '        '
                                                                                                "requests.get(f'http://localhost:1337/json/close/{eid}', "
                                                                                                'timeout=3)\n'
                                                                                                '    except '
                                                                                                'Exception:\n'
                                                                                                '        pass\n'
                                                                                                "print('Done')\n"
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]},
    'fc6d8143-9452-4171-9459-7f515143419a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'INNERPY'\n"
                                                                                                'from datetime import '
                                                                                                'datetime, timedelta\n'
                                                                                                'from zoneinfo import '
                                                                                                'ZoneInfo\n'
                                                                                                'now = datetime.now('
                                                                                                "ZoneInfo('America/Los_Angeles'))\n"
                                                                                                'tom = now + '
                                                                                                'timedelta(days=1)\n'
                                                                                                'date_str = '
                                                                                                "tom.strftime('%a, %b "
                                                                                                "%d, %Y')\n"
                                                                                                "html = ('<!doctype "
                                                                                                "html><html><body>'\n"
                                                                                                "        f'<div "
                                                                                                'class="mach-flight-context-info__wrapper--date">{date_str}</div>\'\n'
                                                                                                "        '<div "
                                                                                                'class="mach-flight-context-info__wrapper__info--separator">JFK<span>sep</span>ORD</div>\'\n'
                                                                                                '        '
                                                                                                "'</body></html>')\n"
                                                                                                'with '
                                                                                                "open('/tmp/delta_fc.html', "
                                                                                                "'w') as f:\n"
                                                                                                '    f.write(html)\n'
                                                                                                'INNERPY',
                                                                                     'shell': True}},
                                                               {   'type': 'launch',
                                                                   'parameters': {   'command': [   'google-chrome',
                                                                                                    '--remote-debugging-port=1337']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}},
                                                               {   'type': 'chrome_open_tabs',
                                                                   'parameters': {   'urls_to_open': [   'file:///tmp/delta_fc.html']}},
                                                               {'type': 'sleep', 'parameters': {'seconds': 5}}]}}
