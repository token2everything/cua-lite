"""Read visible baseline fields on the new owned Magento admin site."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
# Upstream account constants require all configured origins on import; this
# author process accesses only its explicitly scoped admin URL.
os.environ.update(REDDIT="http://localhost:9999",SHOPPING="http://localhost:7770",SHOPPING_ADMIN="http://localhost:7780/admin",GITLAB="http://localhost:8023",WIKIPEDIA="http://localhost:8888",MAP="http://localhost:8080",HOMEPAGE="http://localhost:4399")
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--site-url', required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--storage-state', type=Path, required=True)
args=parser.parse_args()
output=args.output_dir
output.mkdir(parents=True,exist_ok=False)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    context=browser.new_context(storage_state=str(args.storage_state),viewport={"width":1920,"height":1080},record_har_path=str(output/"network.har"),record_har_content="embed")
    context.tracing.start(screenshots=True,snapshots=True,sources=True)
    page=context.new_page()
    observations={}
    for name,path in (("dashboard",""),("orders","sales/order/"),("products","catalog/product/"),("customers","customer/index/")):
        page.goto(args.site_url.rstrip('/')+'/'+path,wait_until="networkidle",timeout=120000)
        page.wait_for_timeout(5000)
        page.screenshot(path=str(output/f"{name}.png"))
        observations[name]={"url":page.url,"body_text":page.locator("body").inner_text(),"tables":[table.inner_text() for table in page.locator("table:visible").all()],"inputs":page.locator("input:visible").evaluate_all("els=>els.map(e=>({type:e.type,name:e.name,placeholder:e.placeholder,value:e.value}))"),"buttons":page.get_by_role("button").all_text_contents()}
        (output/f"{name}.json").write_text(json.dumps(observations[name],ensure_ascii=False,indent=2))
    (output/"observations.json").write_text(json.dumps(observations,ensure_ascii=False,indent=2))
    context.tracing.stop(path=str(output/"trace.zip"))
    context.close()
    browser.close()
print("admin_visible_baseline_exploration_completed",flush=True)
