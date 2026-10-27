"""Conservative diagnostic bins; device totals never establish file attribution."""
from fractions import Fraction

COUNTERS=('Bytes (Read)','Bytes (Write)','Operations (Read)','Operations (Write)',
          'Errors (Read)','Retries (Read)')


def integer(value: object) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError('unsigned observation value required')
    return value


def interval(sample: dict) -> tuple[int,int]:
    c=sample['command']
    start,end=integer(c['start_ns']),integer(c['end_ns'])
    if end < start or c['returncode'] != 0:
        raise ValueError('invalid snapshot interval/status')
    return start,end


def delta(before: dict,after: dict) -> dict:
    if any(integer(before[k]) != integer(after[k]) for k in ('driver_id','media_id')):
        raise ValueError('driver or media changed')
    if interval(before)[1] > interval(after)[0]:
        raise ValueError('overlapping snapshot intervals')
    result={}
    for name in COUNTERS:
        start,end=integer(before['counters'][name]),integer(after['counters'][name])
        if end < start:raise ValueError('counter regressed')
        result[name]=end-start
    if result['Errors (Read)'] or result['Retries (Read)']:
        raise ValueError('new read error or retry')
    return result


def classify(logical_bytes: int,opening: dict,closing: dict,idle_before: tuple[dict,dict],
             idle_after: tuple[dict,dict],native_start_ns: int,native_end_ns: int) -> dict:
    result=dict(status='invalid',physical_io_verified=False)
    try:
        logical=integer(logical_bytes)
        if not logical:raise ValueError('completed logical bytes required')
        samples=(*idle_before,opening,closing,*idle_after)
        if len(samples)!=6:raise ValueError('six snapshots required')
        for before,after in zip(samples,samples[1:]):delta(before,after)
        if not interval(opening)[1] <= integer(native_start_ns) <= integer(native_end_ns) <= interval(closing)[0]:
            raise ValueError('native time outside snapshot bracket')
        observed=delta(opening,closing)
        outer=interval(closing)[1]-interval(opening)[0]
        controls=[]
        for before,after in (idle_before,idle_after):
            inner=interval(after)[0]-interval(before)[1]
            if inner<=0:raise ValueError('nonpositive idle inner interval')
            reads=delta(before,after)['Bytes (Read)']
            controls.append(dict(inner_ns=inner,read_bytes=reads,bytes_per_second=float(Fraction(reads*10**9,inner))))
        background=max(Fraction(c['read_bytes'],c['inner_ns']) for c in controls)*outer
        d=observed['Bytes (Read)']
        if background*100>logical or d*100>logical*125:status='background-inconclusive'
        elif d*100<=logical*5:status='little-device-traffic'
        elif logical*90<=d*100<=logical*125:status='consistent-volume'
        else:status='mixed-inconclusive'
        result.update(status=status,device_read_bytes=d,device_delta=observed,outer_ns=outer,idle=controls,
            background_bytes=dict(numerator=background.numerator,denominator=background.denominator),
            logical_bytes=logical,threshold_percent=dict(background=1,little=5,consistent_min=90,consistent_max=125),
            reasons=['raw device-wide traffic; adjacent idle windows cannot exclude a coincident burst'])
    except (ValueError,KeyError,TypeError,IndexError,OverflowError) as exc:
        result['reasons']=[str(exc)]
    return result
