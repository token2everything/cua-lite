"""Hand-curated oracle metadata for libreoffice_writer eval tasks.

LibreOffice Writer document tasks.

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
- total entries:           22
- with actions:            22
- after_postconfig=True:   0
- block: exclude_reason:   0
- evaluator override:      0
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {   '0810415c-bde4-4443-9047-d5f70165a697': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Novels_Intro_Packet.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/0810415c-bde4-4443-9047-d5f70165a697/Novels_Intro_Packet_Gold.docx'",
                                                                                     'shell': True}}]},
    '0a0faba3-5580-44df-965d-f562a99b291c': {   'actions': [   {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': [   'python3',
                                                                                                    '-c',
                                                                                                    'from '
                                                                                                    'docx '
                                                                                                    'import '
                                                                                                    'Document\n'
                                                                                                    'from '
                                                                                                    'docx.enum.text '
                                                                                                    'import '
                                                                                                    'WD_TAB_ALIGNMENT\n'
                                                                                                    '\n'
                                                                                                    'path = '
                                                                                                    "'/home/user/Desktop/04 "
                                                                                                    'CHIN9505 '
                                                                                                    'EBook '
                                                                                                    'Purchasing '
                                                                                                    'info '
                                                                                                    '2021 '
                                                                                                    "Jan.docx'\n"
                                                                                                    'doc = '
                                                                                                    'Document(path)\n'
                                                                                                    'section '
                                                                                                    '= '
                                                                                                    'doc.sections[0]\n'
                                                                                                    'pw = '
                                                                                                    'section.page_width '
                                                                                                    '- '
                                                                                                    'section.left_margin '
                                                                                                    '- '
                                                                                                    'section.right_margin\n'
                                                                                                    '\n'
                                                                                                    'for p '
                                                                                                    'in '
                                                                                                    'doc.paragraphs:\n'
                                                                                                    '    if '
                                                                                                    'not '
                                                                                                    'p.text.strip():\n'
                                                                                                    '        '
                                                                                                    'continue\n'
                                                                                                    '    '
                                                                                                    'words = '
                                                                                                    'p.text.split()\n'
                                                                                                    '    if '
                                                                                                    'len(words) '
                                                                                                    '< 3:\n'
                                                                                                    '        '
                                                                                                    'continue\n'
                                                                                                    '    '
                                                                                                    'first = '
                                                                                                    "' "
                                                                                                    "'.join(words[:3])\n"
                                                                                                    '    '
                                                                                                    'rest = '
                                                                                                    "' "
                                                                                                    "'.join(words[3:])\n"
                                                                                                    '    for '
                                                                                                    'run in '
                                                                                                    'p.runs:\n'
                                                                                                    '        '
                                                                                                    'run.text '
                                                                                                    "= ''\n"
                                                                                                    '    if '
                                                                                                    'p.runs:\n'
                                                                                                    '        '
                                                                                                    'p.runs[0].text '
                                                                                                    '= first '
                                                                                                    "+ '\\t' "
                                                                                                    '+ rest\n'
                                                                                                    '    '
                                                                                                    'p.paragraph_format.tab_stops.add_tab_stop(pw, '
                                                                                                    'WD_TAB_ALIGNMENT.RIGHT)\n'
                                                                                                    '\n'
                                                                                                    'doc.save(path)\n'
                                                                                                    "print('OK')"]}},
                                                               {   'type': 'sleep',
                                                                   'parameters': {'seconds': 5}}]},
    '0b17a146-2934-46c7-8727-73ff6b6483e8': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/H2O_Factsheet_WA.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/0b17a146-2934-46c7-8727-73ff6b6483e8/H2O_Factsheet_WA_Gold.docx'",
                                                                                     'shell': True}}]},
    '0e47de2a-32e0-456c-a366-8c607ef7a9d2': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from docx '
                                                                                                'import '
                                                                                                'Document\n'
                                                                                                'from '
                                                                                                'docx.oxml.ns '
                                                                                                'import qn\n'
                                                                                                'from '
                                                                                                'docx.oxml '
                                                                                                'import '
                                                                                                'OxmlElement\n'
                                                                                                'from '
                                                                                                'docx.enum.text '
                                                                                                'import '
                                                                                                'WD_ALIGN_PARAGRAPH\n'
                                                                                                '\n'
                                                                                                'doc = '
                                                                                                "Document('/home/user/Desktop/LibreOffice_Open_Source_Word_Processing.docx')\n"
                                                                                                'for section '
                                                                                                'in '
                                                                                                'doc.sections:\n'
                                                                                                '    footer '
                                                                                                '= '
                                                                                                'section.footer\n'
                                                                                                '    '
                                                                                                'footer.is_linked_to_previous '
                                                                                                '= False\n'
                                                                                                '    # Clear '
                                                                                                'existing '
                                                                                                'footer '
                                                                                                'content\n'
                                                                                                '    for p '
                                                                                                'in '
                                                                                                'footer.paragraphs:\n'
                                                                                                '        for '
                                                                                                'r in '
                                                                                                'p.runs:\n'
                                                                                                '            '
                                                                                                'r.text = '
                                                                                                '""\n'
                                                                                                '\n'
                                                                                                '    # Add '
                                                                                                'page number '
                                                                                                'field with '
                                                                                                'rendered '
                                                                                                'value\n'
                                                                                                '    p = '
                                                                                                'footer.paragraphs[0] '
                                                                                                'if '
                                                                                                'footer.paragraphs '
                                                                                                'else '
                                                                                                'footer.add_paragraph()\n'
                                                                                                '    '
                                                                                                'p.alignment '
                                                                                                '= '
                                                                                                'WD_ALIGN_PARAGRAPH.LEFT\n'
                                                                                                '\n'
                                                                                                '    # Begin '
                                                                                                'field\n'
                                                                                                '    run1 = '
                                                                                                'p.add_run()\n'
                                                                                                '    '
                                                                                                'fldChar1 = '
                                                                                                "OxmlElement('w:fldChar')\n"
                                                                                                '    '
                                                                                                "fldChar1.set(qn('w:fldCharType'), "
                                                                                                "'begin')\n"
                                                                                                '    '
                                                                                                'run1._r.append(fldChar1)\n'
                                                                                                '\n'
                                                                                                '    # Field '
                                                                                                'instruction\n'
                                                                                                '    run2 = '
                                                                                                'p.add_run()\n'
                                                                                                '    '
                                                                                                'instrText = '
                                                                                                "OxmlElement('w:instrText')\n"
                                                                                                '    '
                                                                                                "instrText.set(qn('xml:space'), "
                                                                                                "'preserve')\n"
                                                                                                '    '
                                                                                                'instrText.text '
                                                                                                "= ' PAGE '\n"
                                                                                                '    '
                                                                                                'run2._r.append(instrText)\n'
                                                                                                '\n'
                                                                                                '    # '
                                                                                                'Separate '
                                                                                                '(marks '
                                                                                                'start of '
                                                                                                'rendered '
                                                                                                'content)\n'
                                                                                                '    run3 = '
                                                                                                'p.add_run()\n'
                                                                                                '    '
                                                                                                'fldChar2 = '
                                                                                                "OxmlElement('w:fldChar')\n"
                                                                                                '    '
                                                                                                "fldChar2.set(qn('w:fldCharType'), "
                                                                                                "'separate')\n"
                                                                                                '    '
                                                                                                'run3._r.append(fldChar2)\n'
                                                                                                '\n'
                                                                                                '    # '
                                                                                                'Rendered '
                                                                                                'text '
                                                                                                '(actual '
                                                                                                'page number '
                                                                                                'placeholder '
                                                                                                '- gets '
                                                                                                'updated '
                                                                                                'when '
                                                                                                'opened)\n'
                                                                                                '    run4 = '
                                                                                                "p.add_run('1')\n"
                                                                                                '\n'
                                                                                                '    # End '
                                                                                                'field\n'
                                                                                                '    run5 = '
                                                                                                'p.add_run()\n'
                                                                                                '    '
                                                                                                'fldChar3 = '
                                                                                                "OxmlElement('w:fldChar')\n"
                                                                                                '    '
                                                                                                "fldChar3.set(qn('w:fldCharType'), "
                                                                                                "'end')\n"
                                                                                                '    '
                                                                                                'run5._r.append(fldChar3)\n'
                                                                                                '\n'
                                                                                                "doc.save('/home/user/Desktop/LibreOffice_Open_Source_Word_Processing.docx')\n"
                                                                                                "print('Added "
                                                                                                'page '
                                                                                                'numbers '
                                                                                                'with '
                                                                                                'rendered '
                                                                                                "text')\n"
                                                                                                '\n'
                                                                                                '# Verify\n'
                                                                                                'doc2 = '
                                                                                                "Document('/home/user/Desktop/LibreOffice_Open_Source_Word_Processing.docx')\n"
                                                                                                'for i, '
                                                                                                'section in '
                                                                                                'enumerate(doc2.sections):\n'
                                                                                                '    footer '
                                                                                                '= '
                                                                                                'section.footer\n'
                                                                                                '    text = '
                                                                                                'footer.paragraphs[0].text '
                                                                                                'if '
                                                                                                'footer.paragraphs '
                                                                                                "else ''\n"
                                                                                                '    '
                                                                                                'has_digits '
                                                                                                '= '
                                                                                                'any(c.isdigit() '
                                                                                                'for c in '
                                                                                                'text)\n'
                                                                                                '    '
                                                                                                'print(f"Section '
                                                                                                '{i}: footer '
                                                                                                "text='{text}' "
                                                                                                'has_digits={has_digits}")\n'
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '0e763496-b6bb-4508-a427-fad0b6c3e195': {   'after_postconfig': True,
                                                'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                '# Step 1: Apply Times New Roman to every run in the body.\n'
                                                                                                'from docx import Document\n'
                                                                                                "doc = Document('/home/user/Desktop/Dublin_Zoo_Intro.docx')\n"
                                                                                                'for para in doc.paragraphs:\n'
                                                                                                '    for run in para.runs:\n'
                                                                                                "        run.font.name = 'Times New Roman'\n"
                                                                                                "doc.save('/home/user/Desktop/Dublin_Zoo_Intro.docx')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "tmpd=$(mktemp -d) && DISPLAY=:1 soffice --headless --norestore --nofirststartwizard --convert-to docx --outdir \"$tmpd\" '/home/user/Desktop/Dublin_Zoo_Intro.docx' 2>/dev/null && [ -f \"$tmpd/Dublin_Zoo_Intro.docx\" ] && cp \"$tmpd/Dublin_Zoo_Intro.docx\" '/home/user/Desktop/Dublin_Zoo_Intro.docx'; rm -rf \"$tmpd\"; true",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': "python3 << 'PYEOF'\n"
                                                                                                '# Step 3: LO normalize preserves empty runs in trailing\n'
                                                                                                "# paragraphs (p17/p18) whose font.name reads 'Georgia' or\n"
                                                                                                "# None (the original style cascade). compare_font_names\n"
                                                                                                '# rejects the doc on the first non-TNR run. Give every\n'
                                                                                                '# empty run a non-empty text payload (single space) so the\n'
                                                                                                "# next LO normalize doesn't strip the rFonts ascii=TNR\n"
                                                                                                '# attribute, and re-apply TNR run-by-run.\n'
                                                                                                'from docx import Document\n'
                                                                                                "doc = Document('/home/user/Desktop/Dublin_Zoo_Intro.docx')\n"
                                                                                                'for para in doc.paragraphs:\n'
                                                                                                '    for run in para.runs:\n'
                                                                                                "        if run.text == '':\n"
                                                                                                "            run.text = ' '\n"
                                                                                                "        run.font.name = 'Times New Roman'\n"
                                                                                                "doc.save('/home/user/Desktop/Dublin_Zoo_Intro.docx')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '3ef2b351-8a84-4ff2-8724-d86eae9b842e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from docx '
                                                                                                'import '
                                                                                                'Document\n'
                                                                                                'from '
                                                                                                'docx.enum.text '
                                                                                                'import '
                                                                                                'WD_ALIGN_PARAGRAPH\n'
                                                                                                '\n'
                                                                                                'doc = '
                                                                                                "Document('/home/user/Desktop/Constitution_Template_With_Guidelines.docx')\n"
                                                                                                'if '
                                                                                                'doc.paragraphs:\n'
                                                                                                '    '
                                                                                                'doc.paragraphs[0].alignment '
                                                                                                '= '
                                                                                                'WD_ALIGN_PARAGRAPH.CENTER\n'
                                                                                                '    '
                                                                                                "print(f'Set "
                                                                                                'alignment '
                                                                                                'of first '
                                                                                                'paragraph '
                                                                                                'to CENTER: '
                                                                                                '"{doc.paragraphs[0].text[:50]}"\')\n'
                                                                                                "doc.save('/home/user/Desktop/Constitution_Template_With_Guidelines.docx')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '4bcb1253-a636-4df4-8cb0-a35c04dfef31': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/View_Person_Organizational_Summary.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/4bcb1253-a636-4df4-8cb0-a35c04dfef31/View_Person_Organizational_Summary.pdf'",
                                                                                     'shell': True}}]},
    '66399b0d-8fda-4618-95c4-bfc6191617e9': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Table_Of_Work_Effort_Instructions.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/66399b0d-8fda-4618-95c4-bfc6191617e9/Table_Of_Work_Effort_Instructions_Gold.docx'",
                                                                                     'shell': True}}]},
    '6a33f9b9-0a56-4844-9c3f-96ec3ffb3ba2': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/sample-recruitment-phone-script.odt' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/6a33f9b9-0a56-4844-9c3f-96ec3ffb3ba2/sample-recruitment-phone-script_Gold.odt'",
                                                                                     'shell': True}}]},
    '6ada715d-3aae-4a32-a6a7-429b2e43fb93': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Viewing_Your_Class_Schedule_and_Textbooks.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/6ada715d-3aae-4a32-a6a7-429b2e43fb93/Viewing_Your_Class_Schedule_and_Textbooks_Gold.docx'",
                                                                                     'shell': True}}]},
    '6f81754e-285d-4ce0-b59e-af7edb02d108': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/HK_train_record.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/6f81754e-285d-4ce0-b59e-af7edb02d108/HK_train_record_Gold.docx'",
                                                                                     'shell': True}}]},
    '72b810ef-4156-4d09-8f08-a0cf57e7cefe': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/GEOG2169_Course_Outline_2022-23.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/72b810ef-4156-4d09-8f08-a0cf57e7cefe/GEOG2169_Course_Outline_2022-23_Gold.docx'",
                                                                                     'shell': True}}]},
    '8472fece-c7dd-4241-8d65-9b3cd1a0b568': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from docx '
                                                                                                'import '
                                                                                                'Document\n'
                                                                                                'from '
                                                                                                'docx.shared '
                                                                                                'import '
                                                                                                'RGBColor\n'
                                                                                                '\n'
                                                                                                'doc = '
                                                                                                "Document('/home/user/Desktop/Dolch_Sight_Words_Primer.docx')\n"
                                                                                                '\n'
                                                                                                'vowels = '
                                                                                                "set('aeiouAEIOU')\n"
                                                                                                '\n'
                                                                                                'for table '
                                                                                                'in '
                                                                                                'doc.tables:\n'
                                                                                                '    for row '
                                                                                                'in '
                                                                                                'table.rows:\n'
                                                                                                '        for '
                                                                                                'cell in '
                                                                                                'row.cells:\n'
                                                                                                '            '
                                                                                                'for '
                                                                                                'paragraph '
                                                                                                'in '
                                                                                                'cell.paragraphs:\n'
                                                                                                '                '
                                                                                                'for run in '
                                                                                                'paragraph.runs:\n'
                                                                                                '                    '
                                                                                                'word = '
                                                                                                'run.text.strip()\n'
                                                                                                '                    '
                                                                                                'if word:\n'
                                                                                                '                        '
                                                                                                'first_letter '
                                                                                                '= '
                                                                                                'word[0].lower()\n'
                                                                                                '                        '
                                                                                                'if '
                                                                                                'first_letter '
                                                                                                'in '
                                                                                                "'aeiou':\n"
                                                                                                '                            '
                                                                                                'run.font.color.rgb '
                                                                                                '= '
                                                                                                'RGBColor(255, '
                                                                                                '0, 0)  # '
                                                                                                'Red for '
                                                                                                'vowels\n'
                                                                                                '                        '
                                                                                                'else:\n'
                                                                                                '                            '
                                                                                                'run.font.color.rgb '
                                                                                                '= '
                                                                                                'RGBColor(0, '
                                                                                                '0, 255)  # '
                                                                                                'Blue for '
                                                                                                'consonants\n'
                                                                                                '\n'
                                                                                                "doc.save('/home/user/Desktop/Dolch_Sight_Words_Primer.docx')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    '88fe4b2d-3040-4c70-9a70-546a47764b48': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/CCCH9003_Tutorial_guidelines.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/88fe4b2d-3040-4c70-9a70-546a47764b48/CCCH9003_Tutorial_guidelines_Gold_1.docx'",
                                                                                     'shell': True}}]},
    '936321ce-5236-426a-9a20-e0e3c5dc536f': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Graphemes_Sound_Letter_Patterns.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/936321ce-5236-426a-9a20-e0e3c5dc536f/Graphemes_Sound_Letter_Patterns_Gold.docx'",
                                                                                     'shell': True}}]},
    'adf5e2c3-64c7-4644-b7b6-d2f0167927e7': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Essay_Writing_English_for_uni.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/adf5e2c3-64c7-4644-b7b6-d2f0167927e7/Essay_Writing_English_for_uni_Gold.docx'",
                                                                                     'shell': True}}]},
    'b21acd93-60fd-4127-8a43-2f5178f4a830': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/CCHU9045_Course_Outline_2019-20.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/b21acd93-60fd-4127-8a43-2f5178f4a830/CCHU9045_Course_Outline_2019-20_Gold.docx'",
                                                                                     'shell': True}}]},
    'd53ff5ee-3b1a-431e-b2be-30ed2673079b': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/presentation_instruction_2023_Feb.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/d53ff5ee-3b1a-431e-b2be-30ed2673079b/presentation_instruction_2023_Feb_Gold.docx'",
                                                                                     'shell': True}}]},
    'e246f6d8-78d7-44ac-b668-fcf47946cb50': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Y22-2119-assign4.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/e246f6d8-78d7-44ac-b668-fcf47946cb50/Y22-2119-assign4_Gold.docx'",
                                                                                     'shell': True}}]},
    'e528b65e-1107-4b8c-8988-490e4fece599': {   'actions': [   {   'type': 'execute',
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
                                                                                                "'/home/user/Desktop/Geography_And_Magical_Realism.docx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_writer/e528b65e-1107-4b8c-8988-490e4fece599/Geography_And_Magical_Realism_Gold.docx'",
                                                                                     'shell': True}}]},
    'ecc2413d-8a48-416e-a3a2-d30106ca36cb': {   'after_postconfig': True,
                                                'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'PYEOF'\n"
                                                                                                'from docx '
                                                                                                'import '
                                                                                                'Document\n'
                                                                                                'from '
                                                                                                'docx.oxml.ns '
                                                                                                'import qn\n'
                                                                                                'from '
                                                                                                'docx.oxml '
                                                                                                'import '
                                                                                                'OxmlElement\n'
                                                                                                '\n'
                                                                                                'doc = '
                                                                                                "Document('/home/user/Desktop/Sample_Statutory_Declaration.docx')\n"
                                                                                                '# Count '
                                                                                                'existing '
                                                                                                'page '
                                                                                                'breaks\n'
                                                                                                'existing_breaks '
                                                                                                '= 0\n'
                                                                                                'for para in '
                                                                                                'doc.paragraphs:\n'
                                                                                                '    for run '
                                                                                                'in '
                                                                                                'para.runs:\n'
                                                                                                '        for '
                                                                                                'br in '
                                                                                                "run._element.findall(qn('w:br')):\n"
                                                                                                '            '
                                                                                                'if '
                                                                                                "br.get(qn('w:type')) "
                                                                                                "== 'page':\n"
                                                                                                '                '
                                                                                                'existing_breaks '
                                                                                                '+= 1\n'
                                                                                                '\n'
                                                                                                '# Add page '
                                                                                                'breaks '
                                                                                                'until we '
                                                                                                'reach 5. '
                                                                                                'Append '
                                                                                                'breaks to '
                                                                                                'the LAST '
                                                                                                'non-empty '
                                                                                                'paragraph '
                                                                                                'so LO '
                                                                                                'normalize '
                                                                                                'does not '
                                                                                                'drop the '
                                                                                                'trailing '
                                                                                                'empty '
                                                                                                'paragraph '
                                                                                                'carrying '
                                                                                                'the page '
                                                                                                'break.\n'
                                                                                                'needed = 5 '
                                                                                                '- '
                                                                                                'existing_breaks\n'
                                                                                                'target = '
                                                                                                'None\n'
                                                                                                'for p in '
                                                                                                'reversed(doc.paragraphs):\n'
                                                                                                '    if '
                                                                                                'p.text.strip():\n'
                                                                                                '        '
                                                                                                'target = p\n'
                                                                                                '        '
                                                                                                'break\n'
                                                                                                'if target '
                                                                                                'is None:\n'
                                                                                                '    target '
                                                                                                '= '
                                                                                                'doc.add_paragraph("X")\n'
                                                                                                'for _ in '
                                                                                                'range(max(0, '
                                                                                                'needed)):\n'
                                                                                                '    run = '
                                                                                                'target.add_run()\n'
                                                                                                '    br = '
                                                                                                "OxmlElement('w:br')\n"
                                                                                                '    '
                                                                                                "br.set(qn('w:type'), "
                                                                                                "'page')\n"
                                                                                                '    '
                                                                                                'run._element.append(br)\n'
                                                                                                '\n'
                                                                                                "doc.save('/home/user/Desktop/Sample_Statutory_Declaration.docx')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]},
    'f178a4a9-d090-4b56-bc4c-4b72a61a035d': {   'actions': [   {   'type': 'execute',
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
                                                                                                'import '
                                                                                                'xml.etree.ElementTree '
                                                                                                'as ET\n'
                                                                                                'import os\n'
                                                                                                '\n'
                                                                                                'path = '
                                                                                                "'/home/user/.config/libreoffice/4/user/registrymodifications.xcu'\n"
                                                                                                '\n'
                                                                                                'NS = '
                                                                                                "'http://openoffice.org/2001/registry'\n"
                                                                                                "ET.register_namespace('oor', "
                                                                                                'NS)\n'
                                                                                                "ET.register_namespace('xs', "
                                                                                                "'http://www.w3.org/2001/XMLSchema')\n"
                                                                                                "ET.register_namespace('xsi', "
                                                                                                "'http://www.w3.org/2001/XMLSchema-instance')\n"
                                                                                                '\n'
                                                                                                'if '
                                                                                                'os.path.exists(path) '
                                                                                                'and '
                                                                                                'os.path.getsize(path) '
                                                                                                '> 0:\n'
                                                                                                '    tree = '
                                                                                                'ET.parse(path)\n'
                                                                                                '    root = '
                                                                                                'tree.getroot()\n'
                                                                                                'else:\n'
                                                                                                '    root = '
                                                                                                'ET.fromstring(\n'
                                                                                                '        '
                                                                                                "'<?xml "
                                                                                                'version="1.0" '
                                                                                                'encoding="UTF-8"?>\'\n'
                                                                                                '        '
                                                                                                "'<oor:items "
                                                                                                'xmlns:oor="http://openoffice.org/2001/registry"\'\n'
                                                                                                "        ' "
                                                                                                'xmlns:xs="http://www.w3.org/2001/XMLSchema"\'\n'
                                                                                                "        ' "
                                                                                                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\'\n'
                                                                                                '        '
                                                                                                "'</oor:items>'\n"
                                                                                                '    )\n'
                                                                                                '    tree = '
                                                                                                'ET.ElementTree(root)\n'
                                                                                                '\n'
                                                                                                '# Remove '
                                                                                                'any '
                                                                                                'existing '
                                                                                                'DefaultFont/Standard '
                                                                                                'item\n'
                                                                                                'to_remove = '
                                                                                                '[]\n'
                                                                                                'for item in '
                                                                                                'root:\n'
                                                                                                '    '
                                                                                                'oor_path = '
                                                                                                "item.get('{http://openoffice.org/2001/registry}path', "
                                                                                                "'')\n"
                                                                                                '    if '
                                                                                                'oor_path == '
                                                                                                "'/org.openoffice.Office.Writer/DefaultFont':\n"
                                                                                                '        for '
                                                                                                'prop in '
                                                                                                'item:\n'
                                                                                                '            '
                                                                                                'prop_name = '
                                                                                                "prop.get('{http://openoffice.org/2001/registry}name', "
                                                                                                "'')\n"
                                                                                                '            '
                                                                                                'if '
                                                                                                'prop_name '
                                                                                                '== '
                                                                                                "'Standard':\n"
                                                                                                '                '
                                                                                                'to_remove.append(item)\n'
                                                                                                '                '
                                                                                                'break\n'
                                                                                                'for item in '
                                                                                                'to_remove:\n'
                                                                                                '    '
                                                                                                'root.remove(item)\n'
                                                                                                '\n'
                                                                                                '# Add the '
                                                                                                'correct '
                                                                                                'item\n'
                                                                                                'new_item = '
                                                                                                'ET.SubElement(root, '
                                                                                                "'item')\n"
                                                                                                "new_item.set('{http://openoffice.org/2001/registry}path', "
                                                                                                "'/org.openoffice.Office.Writer/DefaultFont')\n"
                                                                                                'new_prop = '
                                                                                                'ET.SubElement(new_item, '
                                                                                                "'prop')\n"
                                                                                                "new_prop.set('{http://openoffice.org/2001/registry}name', "
                                                                                                "'Standard')\n"
                                                                                                "new_prop.set('{http://openoffice.org/2001/registry}op', "
                                                                                                "'fuse')\n"
                                                                                                'new_value = '
                                                                                                'ET.SubElement(new_prop, '
                                                                                                "'value')\n"
                                                                                                'new_value.text '
                                                                                                "= 'Times "
                                                                                                "New Roman'\n"
                                                                                                '\n'
                                                                                                'tree.write(path, '
                                                                                                'xml_declaration=True, '
                                                                                                "encoding='UTF-8')\n"
                                                                                                "print('Done')\n"
                                                                                                'PYEOF',
                                                                                     'shell': True}}]}}
