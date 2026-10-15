"""Authenticate and bind an actual fresh, owned admin image to one execution."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--baseline-receipt', type=Path, required=True)
    parser.add_argument('--storage-state', type=Path, required=True)
    parser.add_argument('--site-urls', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--sandbox-id', required=True)
    parser.add_argument('--generated-task-id', required=True)
    parser.add_argument('--model-profile', choices=('author','teacher','student'), required=True)
    parser.add_argument('--site-url', default='http://localhost:7780/admin')
    args = parser.parse_args()
    inspected = json.loads(args.inspection.read_text())
    if inspected['run_id'] != args.run_id or inspected['session_nonce'] != args.nonce:
        raise ValueError('container is outside this execution scope')
    os.environ.update(REDDIT='http://localhost:9999',SHOPPING='http://localhost:7770',SHOPPING_ADMIN=args.site_url,GITLAB='http://localhost:8023',WIKIPEDIA='http://localhost:8888',MAP='http://localhost:8080',HOMEPAGE='http://localhost:4399')
    from webarena.browser_env.env_config import ACCOUNTS
    from playwright.sync_api import sync_playwright
    account = ACCOUNTS['shopping_admin']
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(extra_http_headers={'X-M2-Admin-Auto-Login':str(account['username'])+':'+str(account['password'])})
        page = context.new_page()
        auth_probes = []
        deadline = time.monotonic() + 120
        while True:
            response = page.goto(args.site_url+'/', wait_until='networkidle', timeout=30000)
            auth_probes.append({'http_status': response.status, 'path': urlsplit(page.url).path, 'title': page.title()})
            ready = response.status == 200 and 'Dashboard' in page.locator('body').inner_text()
            if ready or time.monotonic() >= deadline:
                break
            page.wait_for_timeout(2000)
        args.baseline_receipt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with args.baseline_receipt.with_suffix('.auth_probes.json').open('x') as stream:
            json.dump({'standard_auth_only': True, 'model_started': False, 'probes': auth_probes}, stream, indent=2)
        if not ready:
            diagnostic = {
                'container_id': inspected['container_id'],
                'health_http_status': response.status,
                'terminal_path': urlsplit(page.url).path,
                'page_title': page.title(),
                'visible_error_messages': page.locator('.message-error, .messages .error').all_text_contents(),
                'model_started': False,
                'business_assets_changed': False,
            }
            args.baseline_receipt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            diagnostic_path = args.baseline_receipt.with_suffix('.auth_failure.json')
            with diagnostic_path.open('x') as stream:
                json.dump(diagnostic, stream, indent=2)
            print(json.dumps(diagnostic))
            raise RuntimeError('standard baseline authentication did not reach the dashboard')
        args.storage_state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        args.storage_state.write_text(json.dumps(context.storage_state()))
        args.storage_state.chmod(0o600)
        context.close()
        browser.close()
    receipt = {
        'session_id': args.run_id+'-'+args.model_profile,
        'sandbox_id': args.sandbox_id,
        'generated_task_id': args.generated_task_id,
        'created_at': inspected['created_at'],
        'snapshot': 'webarena-verified-1.2.3-seed',
        'image_digest': inspected['image'].split('@',1)[1],
        'site_url': args.site_url,
        'fresh_baseline': True,
        'container_id': inspected['container_id'],
        'container_name': inspected['container_name'],
        'image_id': inspected['image_id'],
        'run_id': args.run_id,
        'session_nonce': args.nonce,
        'authentication_only_preparation': True,
        'health_http_status': 200,
    }
    args.baseline_receipt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.baseline_receipt.open('x') as stream:
        json.dump(receipt,stream,indent=2)
    args.site_urls.parent.mkdir(parents=True, exist_ok=True)
    args.site_urls.write_text(json.dumps({'shopping_admin':args.site_url}))
    print(json.dumps({'baseline':str(args.baseline_receipt),'model_profile':args.model_profile,'health_http_status':200}))

if __name__ == '__main__':
    main()
