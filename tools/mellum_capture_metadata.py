"""Bounded native receipt/catalog parsing and derived publication metadata."""
from copy import deepcopy
import json
from pathlib import Path
import platform
import re


def bounded_read(path: Path, limit: int) -> bytes:
    with path.open('rb') as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'{path.name} exceeds {limit} bytes')
    return data


def read_json(path: Path, limit: int, exporter) -> dict:
    return exporter.replay_module().parse_object(bounded_read(path, limit).decode('utf-8'), 1)


def profile_definitions(path: Path) -> dict:
    definitions = {}
    roles = ('EMBED', 'PROJECTION', 'ROUTER', 'MID', 'DOWN', 'REDUCE', 'ATTENTION', 'HEAD', 'OTHER')
    for line in bounded_read(path, 256 * 1024).decode('ascii').splitlines():
        if not line.startswith('MELLUM_KERNEL'):
            continue
        match = re.fullmatch(r'MELLUM_KERNEL\(([A-Za-z_][A-Za-z_0-9]*),([A-Z]+),([0-6]),(.+)\)', line)
        if not match:
            raise ValueError('profile kernel catalog syntax is unsupported')
        name, role, count, tail = match.groups()
        labels = re.findall(r'"([A-Za-z_][A-Za-z_0-9]*)"', tail)
        expected = ','.join(f'"{label}"' for label in labels) if int(count) else 'NULL'
        if role not in roles or name in definitions or len(labels) != int(count) or tail != expected:
            raise ValueError('profile kernel catalog is invalid')
        definitions[name] = len(definitions), role, labels
    if not definitions:
        raise ValueError('profile kernel catalog is empty')
    return definitions


def validate_profile(text: str, definitions: dict) -> None:
    roles = ('EMBED', 'PROJECTION', 'ROUTER', 'MID', 'DOWN', 'REDUCE', 'ATTENTION', 'HEAD', 'OTHER')
    parts = text.split(';')
    if parts[:2] not in (['profile-v1', 'head=encoded'], ['profile-v1', 'head=not-encoded']) or not 1 <= len(parts)-2 <= 64:
        raise ValueError('profile header or descriptor count is invalid')
    previous, observed = None, set()
    for item in parts[2:]:
        match = re.fullmatch(r'([A-Z]+):([A-Za-z_][A-Za-z_0-9]{0,94})\((.*)\)', item)
        if not match or match[1] not in roles or match[2] not in definitions:
            raise ValueError('profile descriptor is incomplete or unknown')
        role, name, arguments = match.groups()
        number, expected_role, labels = definitions[name]
        if role != expected_role and not (role == 'HEAD' and expected_role == 'PROJECTION'):
            raise ValueError('profile role conflicts with kernel catalog')
        values = arguments.split(',') if arguments else []
        if len(values) != len(labels):
            raise ValueError('profile constants are incomplete')
        numbers = []
        for label, value in zip(labels, values, strict=True):
            match = re.fullmatch(re.escape(label) + r'=(0|[1-9][0-9]{0,9})', value)
            if not match or int(match[1]) > 2**32-1:
                raise ValueError('profile constants are malformed')
            numbers.append(int(match[1]))
        key = roles.index(role), number, len(numbers), *numbers
        if previous is not None and key <= previous:
            raise ValueError('profile descriptors are duplicated or not canonical')
        previous = key
        observed.add(role)
    if not set(roles[:7]) <= observed or ('HEAD' in observed) != (parts[1] == 'head=encoded'):
        raise ValueError('profile is missing required roles or has inconsistent head state')


def read_profiles(path: Path, catalog: Path | None = None) -> dict[str, str]:
    definitions = profile_definitions(catalog or Path(__file__).resolve().parents[1] / 'ds4_mellum_kernels.inc')
    result = {}
    with path.open('rb') as source:
        for number in range(129):
            line = source.readline(16384 + 1)
            if not line:
                break
            if number == 128 or len(line) > 16384 or not line.endswith(b'\n'):
                raise ValueError('profile catalog exceeds bounded complete lines')
            text = line[:-1].decode('ascii')
            key, sep, description = text.partition(' ')
            if key != str(number) or not sep or not re.match(r'profile-v1;head=(?:encoded|not-encoded)(?:;|$)', description):
                raise ValueError('profile catalog has invalid ID or description')
            if any(ord(char) < 32 or ord(char) > 126 for char in description):
                raise ValueError('profile catalog contains control characters')
            validate_profile(description, definitions)
            result[key] = description
    if not result:
        raise ValueError('profile catalog is empty')
    return result


def read_receipt(path: Path, exporter) -> dict:
    data = read_json(path, 16384, exporter)
    if set(data) != {'receipt_version', 'context_tokens', 'cap_tokens', 'capabilities'}:
        raise ValueError('invalid runtime receipt fields')
    if type(data['receipt_version']) is not int or data['receipt_version'] != 1:
        raise ValueError('invalid runtime receipt version')
    for key, limit in (('context_tokens', 2**31-1), ('cap_tokens', 1024)):
        if type(data[key]) is not int or not 1 <= data[key] <= limit:
            raise ValueError(f'invalid runtime receipt {key}')
    flags = data['capabilities']
    if not isinstance(flags, dict) or set(flags) != {'tensor_tiles_available', 'pre_m5_apple', 'm5_apple'} or any(type(v) is not bool for v in flags.values()):
        raise ValueError('invalid runtime receipt capabilities')
    return data


def derive(target: dict, values: dict, path: str = 'header') -> None:
    for key, value in values.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict) and isinstance(target[key], dict):
            derive(target[key], value, f'{path}.{key}')
        elif json.dumps(target[key], sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f'conflicting derived field {path}.{key}')


def complete_header(header: dict, *, source: dict, binary: dict, receipt: dict, token_file: dict,
                    model_stat: dict, build_log: str, first: int, count: int, exact: bool, chunk: int) -> dict:
    result = deepcopy(header)
    mode = 'exact-prefill' if exact else 'release'
    derive(result, dict(schema_version=2, capture_kind='measured', engine_revision=source['revision'], numerical_mode=mode))
    manifest = result.setdefault('capture_manifest', {})
    if not isinstance(manifest, dict):
        raise ValueError('capture_manifest must be an object')
    execution = manifest.setdefault('execution', {})
    if not isinstance(execution, dict) or not isinstance(execution.get('device'), str) or not execution['device'].strip():
        raise ValueError('caller-declared execution.device is required')
    derive(manifest, {
        'manifest_version': 1,
        'build': dict(engine_revision=source['revision'], recorder_revision=source['revision'],
                      source_state_id=source['source_state_id'], binary_sha256=binary['sha256'],
                      source_inputs={name: value['sha256'] for name, value in source['files'].items()},
                      forced_rebuild=True, command_log=build_log),
        'artifacts': dict(input_file_sha256=token_file['sha256'], model_file_stat=model_stat),
        'execution': dict(context_tokens=receipt['context_tokens'], prefill_chunk_tokens=receipt['cap_tokens'],
                          numerical_mode=mode, requested_kernel='default-exact-prefill' if exact else 'default',
                          requested_controls=dict(prefill_chunk=chunk, exact_prefill=exact),
                          capabilities=receipt['capabilities'], os=platform.platform(), device_label_source='caller-declared',
                          capture_schedule=dict(protocol='owned-parity-proof-v1', first_prefill=first,
                                                token_count=count, final_append_tokens=1)),
        'generation': dict(mode='teacher-forced', settings={}),
    })
    if manifest['generation']['settings']:
        raise ValueError('teacher-forced capture cannot declare sampling settings')
    if set(execution['capabilities']) != set(receipt['capabilities']):
        raise ValueError('capabilities must contain only observed flags')
    if set(manifest['build']['source_inputs']) != set(source['files']):
        raise ValueError('source_inputs must contain only observed files')
    return result
