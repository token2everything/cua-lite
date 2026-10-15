"""Hand-curated oracle metadata for libreoffice_calc eval tasks.

LibreOffice Calc spreadsheet tasks.

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
- total entries:           46
- with actions:            46
- after_postconfig=True:   6
- block: exclude_reason:   0
- evaluator override:      0
"""

from __future__ import annotations

ORACLES: dict[str, dict] = {   '01b269ae-2111-4a07-81fd-3fcd711993b0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Student_Level_Fill_Blank.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/01b269ae-2111-4a07-81fd-3fcd711993b0/Student_Level_Fill_Blank_gold.xlsx'",
                                                                                     'shell': True}}]},
    '0326d92d-d218-48a8-9ca1-981cd6d064c7': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/SalesRep.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/0326d92d-d218-48a8-9ca1-981cd6d064c7/2_SalesRep_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '035f41ba-6653-43ab-aa63-c86d449d62e5': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/IncomeStatement2.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/035f41ba-6653-43ab-aa63-c86d449d62e5/5_IncomeStatement2_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '04d9aeaf-7bed-4024-bedb-e10e6f00eb7f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/SmallBalanceSheet.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/04d9aeaf-7bed-4024-bedb-e10e6f00eb7f/4_SmallBalanceSheet_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '0a2e43bf-b26c-4631-a966-af9dfa12c9e5': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/SalesRep.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/0a2e43bf-b26c-4631-a966-af9dfa12c9e5/5_SalesRep_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '0bf05a7d-b28b-44d2-955a-50b41e24012a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Customers_New_7digit_Id.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/0bf05a7d-b28b-44d2-955a-50b41e24012a/Customers_New_7digit_Id_gold.xlsx'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -f '
                                                                                                "'/home/user/Customers_New_7digit_Id-Sheet1.csv'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '0cecd4f3-74de-457b-ba94-29ad6b5dafb6': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/copy_sheet_insert.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/0cecd4f3-74de-457b-ba94-29ad6b5dafb6/LARS_Science_Assessment_Resource_List_7-30-21_gold.xlsx'",
                                                                                     'shell': True}}]},
    '12382c62-0cd1-4bf2-bdc8-1d20bf9b2371': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/WeeklySales.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/12382c62-0cd1-4bf2-bdc8-1d20bf9b2371/13_WeeklySales_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '1273e544-688f-496b-8d89-3e0f40aa0606': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/NetIncome.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/1273e544-688f-496b-8d89-3e0f40aa0606/3_NetIncome_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '1334ca3e-f9e3-4db8-9ca7-b4c653be7d17': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Zoom_Out_Oversized_Cells.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/1334ca3e-f9e3-4db8-9ca7-b4c653be7d17/Zoom_Out_Oversized_Cells_gold.xlsx'",
                                                                                     'shell': True}}]},
    '1954cced-e748-45c4-9c26-9855b97fbc5e': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Invoices.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/1954cced-e748-45c4-9c26-9855b97fbc5e/8_Invoices_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '1d17d234-e39d-4ed7-b46f-4417922a4e7c': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/FutureValue.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/1d17d234-e39d-4ed7-b46f-4417922a4e7c/5_FutureValue_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '1de60575-bb6e-4c3d-9e6a-2fa699f9f197': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/EntireSummerSales.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/1de60575-bb6e-4c3d-9e6a-2fa699f9f197/6_EntireSummerSales_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '1e8df695-bd1b-45b3-b557-e7d599cf7597': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/WeeklySales.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/1e8df695-bd1b-45b3-b557-e7d599cf7597/6_WeeklySales_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '21ab7b40-77c2-4ae6-8321-e00d3a086c73': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/PeriodRate.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/21ab7b40-77c2-4ae6-8321-e00d3a086c73/1_PeriodRate_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '21df9241-f8d7-4509-b7f1-37e501a823f7': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Represent_in_millions_billions.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/21df9241-f8d7-4509-b7f1-37e501a823f7/Represent_in_millions_billions_gold.xlsx'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -f '
                                                                                                "'/home/user/Represent_in_millions_billions-Sheet1.csv'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '26a8440e-c166-4c50-aef4-bfb77314b46b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/SalesRep.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/26a8440e-c166-4c50-aef4-bfb77314b46b/3_SalesRep_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '30e3e107-1cfb-46ee-a755-2cd080d7ba6a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/DemographicProfile.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/30e3e107-1cfb-46ee-a755-2cd080d7ba6a/1_DemographicProfile_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '347ef137-7eeb-4c80-a3bb-0951f26a8aff': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Create_column_charts_using_statistics.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/347ef137-7eeb-4c80-a3bb-0951f26a8aff/Create_column_charts_using_statistics_gold.xlsx'",
                                                                                     'shell': True}}]},
    '357ef137-7eeb-4c80-a3bb-0951f26a8aff': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'python3 << '
                                                                                                "'CELLEOF'\n"
                                                                                                'import '
                                                                                                'openpyxl\n'
                                                                                                'wb = '
                                                                                                "openpyxl.load_workbook('/home/user/Multiply_Time_Number.xlsx')\n"
                                                                                                'updates = '
                                                                                                "[(0, 'E3', "
                                                                                                '191.6667)]\n'
                                                                                                'for '
                                                                                                'sheet_idx, '
                                                                                                'coord, '
                                                                                                'value in '
                                                                                                'updates:\n'
                                                                                                '    ws = '
                                                                                                'wb.worksheets[sheet_idx]\n'
                                                                                                '    '
                                                                                                'ws[coord] = '
                                                                                                'value\n'
                                                                                                "wb.save('/home/user/Multiply_Time_Number.xlsx')\n"
                                                                                                'CELLEOF',
                                                                                     'shell': True}}]},
    '37608790-6147-45d0-9f20-1137bb35703d': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Employee_Roles_and_Ranks.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/37608790-6147-45d0-9f20-1137bb35703d/Employee_Roles_and_Ranks_gold.xlsx'",
                                                                                     'shell': True}}]},
    '3a7c8185-25c1-4941-bd7b-96e823c9f21f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/BoomerangSales.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/3a7c8185-25c1-4941-bd7b-96e823c9f21f/6_BoomerangSales_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '3aaa4e37-dc91-482e-99af-132a612d40f3': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Export_Calc_to_CSV.csv' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/3aaa4e37-dc91-482e-99af-132a612d40f3/Export_Calc_to_CSV.csv'",
                                                                                     'shell': True}}]},
    '4172ea6e-6b77-4edb-a9cc-c0014bd1603b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/MaturityDate.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/4172ea6e-6b77-4edb-a9cc-c0014bd1603b/1_MaturityDate_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '4188d3a4-077d-46b7-9c86-23e1a036f6c1': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Freeze_row_column.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/4188d3a4-077d-46b7-9c86-23e1a036f6c1/Freeze_row_column2_gold.xlsx'",
                                                                                     'shell': True}}]},
    '42e0a640-4f19-4b28-973d-729602b5a4a7': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/NetIncome.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/42e0a640-4f19-4b28-973d-729602b5a4a7/2_NetIncome_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '4de54231-e4b5-49e3-b2ba-61a0bec721c0': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/RampUpAndDown.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/4de54231-e4b5-49e3-b2ba-61a0bec721c0/3_RampUpAndDown_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '4e6fcf72-daf3-439f-a232-c434ce416af6': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Employee_Age_By_Birthday.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/4e6fcf72-daf3-439f-a232-c434ce416af6/Employee_Age_By_Birthday_gold.xlsx'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Employee_Age_By_Birthday_gold.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/4e6fcf72-daf3-439f-a232-c434ce416af6/Employee_Age_By_Birthday_gold.xlsx'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '4f07fbe9-70de-4927-a4d5-bb28bc12c52c': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Padding_Decimals_In_Formular.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/4f07fbe9-70de-4927-a4d5-bb28bc12c52c/Padding_Decimals_In_Formular_gold.xlsx'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -f '
                                                                                                "'/home/user/Padding_Decimals_In_Formular-Sheet1.csv'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '51719eea-10bc-4246-a428-ac7c433dd4b3': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/BoomerangSales.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/51719eea-10bc-4246-a428-ac7c433dd4b3/8_BoomerangSales_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '51b11269-2ca8-4b2a-9163-f21758420e78': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Arrang_Value_min_to_max.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/51b11269-2ca8-4b2a-9163-f21758420e78/Arrang_Value_min_to_max_gold.xlsx'",
                                                                                     'shell': True}}]},
    '535364ea-05bd-46ea-9937-9f55c68507e8': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/SummerSales.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/535364ea-05bd-46ea-9937-9f55c68507e8/5_SummerSales_gt1.xlsx'",
                                                                                     'shell': True}}]},
    '6054afcb-5bab-4702-90a0-b259b5d3217c': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Date_Budget_Variance_HideNA.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/6054afcb-5bab-4702-90a0-b259b5d3217c/Date_Budget_Variance_HideNA_gold.xlsx'",
                                                                                     'shell': True}}]},
    '6e99a1ad-07d2-4b66-a1ce-ece6d99c20a5': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Keep_Two_decimal_points.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/6e99a1ad-07d2-4b66-a1ce-ece6d99c20a5/Keep_Two_decimal_points_gold.xlsx'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -f '
                                                                                                "'/home/user/Keep_Two_decimal_points-Sheet1.csv'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    '7a4e4bc8-922c-4c84-865c-25ba34136be1': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Name_Order_Id_move_column.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/7a4e4bc8-922c-4c84-865c-25ba34136be1/Name_Order_Id_move_column_gold.xlsx'",
                                                                                     'shell': True}}]},
    '7e429b8d-a3f0-4ed0-9b58-08957d00b127': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/VLOOKUP_Fill_the_form.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/7e429b8d-a3f0-4ed0-9b58-08957d00b127/VLOOKUP_Fill_the_form_gold.xlsx'",
                                                                                     'shell': True}}]},
    '7efeb4b1-3d19-4762-b163-63328d66303b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Order_Sales_Serial#.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/7efeb4b1-3d19-4762-b163-63328d66303b/Order_Sales_Serial%23_gold.xlsx'",
                                                                                     'shell': True}}]},
    '8b1ce5f2-59d2-4dcc-b0b0-666a714b9a14': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Calendar_Highlight_Weekend_Days.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/8b1ce5f2-59d2-4dcc-b0b0-666a714b9a14/Calendar_Highlight_Weekend_Days_gold.xlsx'",
                                                                                     'shell': True}}]},
    'a01fbce3-2793-461f-ab86-43680ccbae25': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Set_Decimal_Separator_Dot.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/a01fbce3-2793-461f-ab86-43680ccbae25/Set_Decimal_Separator_Dot_gold_ru.xlsx'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'rm -f '
                                                                                                "'/home/user/Set_Decimal_Separator_Dot-Sheet1.csv'",
                                                                                     'shell': True}}],
                                                'after_postconfig': True},
    'a9f325aa-8c05-4e4f-8341-9e4358565f4f': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Movie_title_garbage_clean.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/a9f325aa-8c05-4e4f-8341-9e4358565f4f/Movie_titles_garbage_clean_gold.xlsx'",
                                                                                     'shell': True}}]},
    'aa3a8974-2e85-438b-b29e-a64df44deb4b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Resize_Cells_Fit_Page.pdf' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/aa3a8974-2e85-438b-b29e-a64df44deb4b/Resize_Cells_Fit_Page_gt.pdf'",
                                                                                     'shell': True}}]},
    'abed40dc-063f-4598-8ba5-9fe749c0615d': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Names_Duplicate_Unique.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/abed40dc-063f-4598-8ba5-9fe749c0615d/Names_Duplicate_Unique_gold.xlsx'",
                                                                                     'shell': True}}]},
    'd681960f-7bc3-4286-9913-a8812ba3261a': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Student_Grades_and_Remarks.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/d681960f-7bc3-4286-9913-a8812ba3261a/Student_Grades_and_Remarks_gold.xlsx'",
                                                                                     'shell': True}}]},
    'eb03d19a-b88d-4de4-8a64-ca0ac66f426b': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Students_Class_Subject_Marks.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/eb03d19a-b88d-4de4-8a64-ca0ac66f426b/Students_Class_Subject_Marks_gold.xlsx'",
                                                                                     'shell': True}}]},
    'ecb0df7a-4e8d-4a03-b162-053391d3afaf': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Order_Id_Mark_Pass_Fail.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/ecb0df7a-4e8d-4a03-b162-053391d3afaf/Order_Id_Mark_Pass_Fail_gold.xlsx'",
                                                                                     'shell': True}}]},
    'f9584479-3d0d-4c79-affa-9ad7afdd8850': {   'actions': [   {   'type': 'execute',
                                                                   'parameters': {   'command': 'killall '
                                                                                                'soffice.bin '
                                                                                                '2>/dev/null; '
                                                                                                'sleep 1',
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'mkdir -p '
                                                                                                "'/home/user'",
                                                                                     'shell': True}},
                                                               {   'type': 'execute',
                                                                   'parameters': {   'command': 'wget -q -O '
                                                                                                "'/home/user/Quarterly_Product_Sales_by_Zone.xlsx' "
                                                                                                "'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/libreoffice_calc/f9584479-3d0d-4c79-affa-9ad7afdd8850/Quarterly_Product_Sales_by_Zone_gold.xlsx'",
                                                                                     'shell': True}}]}}
