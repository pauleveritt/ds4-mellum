#!/bin/sh
# The scripted background load for S6 phase C's footprint runs (docs/superpowers/plans/2026-10-28-mellum-s6-c-imac.md,
# Task 7): PyCharm on a small Python project, and Safari with the preregistered URLs, one tab each.  It opens
# apps, so only the owner runs it.  It waits for Enter once everything has loaded, then prints a timestamp.
#   tools/mellum_background_load.sh PROJECT_DIR URLS_FILE
set -u
proj=${1:?usage: tools/mellum_background_load.sh PROJECT_DIR URLS_FILE}
urls=${2:?usage: tools/mellum_background_load.sh PROJECT_DIR URLS_FILE}
[ -d "$proj" ] || { echo "mellum_background_load: no directory $proj" >&2; exit 2; }
[ -r "$urls" ] || { echo "mellum_background_load: cannot read $urls" >&2; exit 2; }
open -a "PyCharm" "$proj" || { echo "mellum_background_load: cannot open PyCharm" >&2; exit 1; }
n=0
while IFS= read -r u; do
    case "$u" in ''|'#'*) continue ;; esac
    open -a Safari "$u" || { echo "mellum_background_load: cannot open $u" >&2; exit 1; }
    n=$((n + 1))
done < "$urls"
echo "mellum_background_load: PyCharm on $proj and $n Safari tabs; press Enter once they have loaded"
read -r _
date -u '+%Y-%m-%dT%H:%M:%SZ mellum_background_load: ready'
