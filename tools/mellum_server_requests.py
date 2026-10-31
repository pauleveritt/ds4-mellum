"""Per-request prefill and decode rates from a ds4-server log.

    python3 tools/mellum_server_requests.py SERVER_LOG

ds4-server writes two lines for every request (ds4_server.c, generate_job_inner):

    ds4-server: chat ctx=C..P:N[ FLAGS] prompt done S.SSSs
    ds4-server: chat ctx=C..P:N gen=G[ FLAGS] finish=F[ error="E"] T.TTTs

ctx is the request's span: C prompt tokens reused from the cache, P in the prompt, N prefilled.  S and T are seconds
since the request started, so prefill runs at N / S and decode at G / (T - S).  A finish line closes the request
opened by the latest prompt-done line with the same span; one without a prompt-done line (a request that failed
before its prefill ended) has no prefill time.  Every other line is ignored; colour escapes are stripped.
"""
from dataclasses import dataclass
import re
import sys
from typing import Iterable

ANSI = re.compile(r'\x1b\[[0-9;]*m')
CTX = r'ds4-server: (?:chat|completion) ctx=(\d+)\.\.(\d+):(\d+)'
DONE = re.compile(CTX + r'(?: [A-Z_ ]+)? prompt done ([0-9.]+)s')
FINISH = re.compile(CTX + r' gen=(\d+)(?: [A-Z_ ]+?)? finish=(\S+)(?: error="(.*)")? ([0-9.]+)s')


@dataclass(frozen=True)
class Request:
    index: int
    cached: int
    prompt: int
    new: int
    prefill_s: float | None
    generated: int
    total_s: float
    finish: str
    error: str | None

    @property
    def prefill_tps(self) -> float | None:
        if self.new == 0 or self.prefill_s is None or self.prefill_s <= 0:
            return None
        return self.new / self.prefill_s

    @property
    def decode_tps(self) -> float | None:
        if self.prefill_s is None or self.generated == 0 or self.total_s - self.prefill_s <= 0:
            return None
        return self.generated / (self.total_s - self.prefill_s)


def parse(lines: Iterable[str]) -> list[Request]:
    open_spans: dict[tuple[int, int, int], float] = {}
    requests: list[Request] = []
    for raw in lines:
        line = ANSI.sub('', raw)
        if m := DONE.search(line):
            open_spans[(int(m[1]), int(m[2]), int(m[3]))] = float(m[4])
        elif m := FINISH.search(line):
            span = (int(m[1]), int(m[2]), int(m[3]))
            requests.append(Request(index=len(requests), cached=span[0], prompt=span[1], new=span[2],
                                    prefill_s=open_spans.pop(span, None), generated=int(m[4]),
                                    total_s=float(m[7]), finish=m[5], error=m[6]))
    return requests


def _fmt(value: float | None) -> str:
    return '' if value is None else f'{value:.1f}'


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1], errors='replace') as f:
        requests = parse(f)
    print('index\tcached\tprompt\tnew\tprefill_tps\tgenerated\tdecode_tps\tfinish')
    for r in requests:
        print(f'{r.index}\t{r.cached}\t{r.prompt}\t{r.new}\t{_fmt(r.prefill_tps)}\t{r.generated}\t'
              f'{_fmt(r.decode_tps)}\t{r.finish}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
