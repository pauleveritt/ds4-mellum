#!/usr/bin/env python3
"""CPU-only Mellum storage preparation and evidence collection."""
import argparse
import importlib.util
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity)
common = identity.sibling('mellum_storage_common')
index_tools = identity.sibling('mellum_expert_index')


def main() -> None:
    if sys.version_info < (3, 14):
        raise SystemExit('Python 3.14+ required')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    index = commands.add_parser('index')
    index.add_argument('--model', type=Path, required=True)
    index.add_argument('--output', type=Path, required=True)
    pack = commands.add_parser('pack')
    pack.add_argument('--index', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('prepare')
    for name in ('index', 'packed', 'trace', 'binary', 'output'):
        prepare.add_argument(f'--{name}', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--batch', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'index':
        value = index_tools.build_index(args.model)
        result = common.publication.publish(args.output, lambda stage: common.write_json(stage / 'index.json', value))
        print(result)
    elif args.command == 'pack':
        print(identity.sibling('mellum_expert_pack').pack(args.index, args.output))
    elif args.command == 'prepare':
        print(identity.sibling('mellum_read_batch').prepare_batch(args.index, args.packed, args.trace, args.binary, args.output))
    elif args.command == 'run':
        print(identity.sibling('mellum_read_batch').run_batch(args.batch, args.output))


if __name__ == '__main__':
    main()
