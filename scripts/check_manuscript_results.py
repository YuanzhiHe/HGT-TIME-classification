#!/usr/bin/env python3
"""Check manuscript result links, checksums and reported AUROC values."""
from pathlib import Path
import hashlib
import json
import math


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'configs/hgt_time/experiment_manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    for record in manifest['records']:
        for name in record['source_files']:
            if not (root / name).is_file():
                raise FileNotFoundError(name)
        if record['config_snapshot']:
            config = json.loads((root / record['config_snapshot']).read_text())
            if config['evidence_id'] != record['evidence_id']:
                raise ValueError(f'Configuration ID mismatch: {record["evidence_id"]}')
            if not (root / config['source_script']).is_file():
                raise FileNotFoundError(config['source_script'])
    for query in manifest['metric_queries']:
        value = json.loads((root / query['source']).read_text())
        for key in query['path']:
            value = value[key]
        if not math.isclose(value, query['full_precision'], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'Result mismatch: {query["item"]}')
        if f'{value:.3f}' != query['reported_3dp']:
            raise ValueError(f'Rounding mismatch: {query["item"]}')
        print(f'{query["item"]}: {value:.10f} -> {query["reported_3dp"]}')
    print(f'Checked {len(manifest["file_sha256"])} files, '
          f'{len(manifest["records"])} table/panel mappings and '
          f'{len(manifest["metric_queries"])} reported values.')


if __name__ == '__main__':
    main()
