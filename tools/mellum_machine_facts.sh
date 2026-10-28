#!/bin/sh
# Machine facts for a streamed-Mellum measurement host (docs/superpowers/plans/2026-10-28-mellum-m1pro-bringup.md,
# step 1).  Read-only: no GPU, no model, no writes except OUT.  Usage: tools/mellum_machine_facts.sh OUT.txt
set -u
out=${1:?usage: tools/mellum_machine_facts.sh OUT.txt}
[ -e "$out" ] && { echo "mellum_machine_facts: $out exists" >&2; exit 1; }
{
    echo "## date"; date -u '+%Y-%m-%dT%H:%M:%SZ'
    echo "## sysctl"; sysctl hw.model hw.memsize hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu \
        machdep.cpu.brand_string vm.user_wire_limit 2>&1
    echo "## sw_vers"; sw_vers 2>&1
    echo "## xcode"; xcode-select -p 2>&1; xcrun --show-sdk-version 2>&1; xcrun -f metal 2>&1
    echo "## gpu"; system_profiler SPDisplaysDataType 2>&1
    echo "## storage"; system_profiler SPNVMeDataType 2>&1
    echo "## disk"; df -h "$HOME" 2>&1
    echo "## memory pressure"; memory_pressure -Q 2>&1 | tail -1
    echo "## swap"; sysctl vm.swapusage 2>&1
    echo "## power"; pmset -g batt 2>&1; pmset -g therm 2>&1
} > "$out"
echo "wrote $out"
