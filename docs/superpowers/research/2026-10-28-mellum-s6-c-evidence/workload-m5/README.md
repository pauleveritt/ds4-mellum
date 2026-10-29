# Phase C's workload, M5 reference

- `run-server-attempt.sh` tried to replay the request through `ds4-server`. The server refused Mellum
  ("ds4-server has no Hermes tool-call syntax": `server-resident.log`, `server-streamed815.log`), and the
  three resident replays failed (`failures-server-attempt.txt`). Their client stdout and stderr were
  overwritten by the later measure runs of the same names (rule 8 breached; the server logs keep the
  refusal).
- `run.sh` is the amended harness: `tools/mellum_s4_measure` on the flattened prompt. It produced
  `resident-*` and `streamed2.5g-*` (`.measure.json`, `.stdout`, `.stderr`) at build `614eb33`.
