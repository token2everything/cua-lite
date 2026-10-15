"""Project a fresh authenticated execution receipt into the authoring contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.execution_baseline.read_text())
    if source['fresh_baseline'] is not True or source['authentication_only_preparation'] is not True:
        raise ValueError('Authoring requires an actual fresh, authentication-only baseline')
    receipt = {
        'container_id': source['container_id'],
        'container_name': source['container_name'],
        'created_at': source['created_at'],
        'image': source.get('image', source['image_digest']),
        'image_id': source['image_id'],
        'image_digest': source['image_digest'],
        'snapshot': source['snapshot'],
        'site_url': source['site_url'],
        'health_http_status': source['health_http_status'],
        'owned_run': source['run_id'],
        'session_nonce': source['session_nonce'],
        'sandbox_id': source['sandbox_id'],
        'fresh_from_image': True,
        'authentication_only_preparation': True,
        'execution_baseline_sha256': hashlib.sha256(args.execution_baseline.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'author_baseline': str(args.output), 'model_calls': 0}))


if __name__ == '__main__':
    main()
