"""Frozen descriptive windows from one cold-start global LRU replay."""
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
replay=identity.sibling('mellum_ssd_replay')
SEGMENTS={'0':'initial-prefill','1':'response0','2':'append-prefill','3':'response1'}


def window(items: list[dict], expert_bytes: int, *, requested: int | None = None, overlap: bool = False) -> dict:
    counts={key:sum(item[key] for item in items) for key in replay.COUNTERS}
    positions=[p for item in items for p in item['positions']]
    return dict(**counts,tokens=len(positions),positions=positions,forwards=[item['forward'] for item in items],
                hit_bytes=counts['group_expert_hits']*expert_bytes,
                demand_distribution=replay.distribution([item['demand_bytes'] for item in items]),
                requested_tokens=requested,shortened=requested is not None and len(positions)<requested,
                overlaps_other_windows=overlap)


def analyze(trace, operations: Path | None = None, capacity: int = 3*2**30) -> dict:
    """Replay once; windows inherit complete earlier state and may overlap."""
    whole=replay.replay(trace,capacity,'lru')
    observed={item['segment'] for item in whole['forwards']}
    if not observed<=set(SEGMENTS.values()):raise ValueError('unexpected generated trace segment')
    windows={}
    for name in ('initial-prefill','append-prefill'):
        items=[item for item in whole['forwards'] if item['segment']==name]
        if any(item['phase']!='prefill' for item in items):raise ValueError('prefill segment contains decode')
        windows[name]=window(items,trace.header.geometry.expert_bytes)
    for name in ('response0','response1'):
        items=[item for item in whole['forwards'] if item['segment']==name]
        if any(item['phase']!='decode' for item in items):raise ValueError('response contains prefill')
        windows[name+'-first32']=window(items[:32],trace.header.geometry.expert_bytes,requested=32)
        windows[name+'-after32']=window(items[32:],trace.header.geometry.expert_bytes)
        windows[name+'-tail256']=window(items[-256:],trace.header.geometry.expert_bytes,requested=256,overlap=True)
    return dict(whole=whole,windows=windows,token_series=whole['decode_tokens'],
                method='one complete cold-prefix global grouped LRU replay; overlapping descriptive windows are not summed',
                limits=['n=1 scripted coding review; no tool execution or medium/8K coverage',
                        'cache demand is not inference stall time or physical 8GB feasibility'])
