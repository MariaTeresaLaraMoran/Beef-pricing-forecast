#!/usr/bin/env bash
# Keep retrying the USDA boxed beef download until the cache is complete.
#
# The USDA LMR Datamart (mpr.datamart.ams.usda.gov) is intermittently unavailable:
# during November 2026 it alternated between serving data normally, returning HTTP 500
# and timing out, within the same hour. fetch_usda_boxed_beef.py caches each date
# window it manages to retrieve and skips cached windows on the next run, so repeated
# passes accumulate progress.
#
# Usage:
#   bash retry_usda.sh            # retry every 10 minutes until complete
#   bash retry_usda.sh 300        # retry every 5 minutes instead
#
# Stop with Ctrl-C at any time; nothing already downloaded is lost.

set -u
SLEEP_SECONDS="${1:-600}"
CACHE_DIR="data/raw/usda_cache"
EXPECTED=216          # 108 quarterly windows x 2 sections (2000 to 2026)
PASS=0

count_cached () { find "$CACHE_DIR" -name '*.csv' 2>/dev/null | wc -l | tr -d ' '; }

while true; do
    PASS=$((PASS + 1))
    echo "=============================================================="
    echo "Pass $PASS  -  $(date '+%Y-%m-%d %H:%M:%S')  -  cached: $(count_cached)/$EXPECTED"
    echo "=============================================================="

    python src/fetch_usda_boxed_beef.py
    STATUS=$?

    CACHED=$(count_cached)
    if [ "$STATUS" -eq 2 ]; then
        echo "Datamart is down; nothing attempted this pass."
    fi
    echo ""
    echo "Pass $PASS finished. Cached windows: $CACHED/$EXPECTED"

    if [ "$CACHED" -ge "$EXPECTED" ]; then
        echo "All windows downloaded. Combined CSVs are in data/raw/. Done."
        break
    fi

    echo "Sleeping ${SLEEP_SECONDS}s before the next pass (Ctrl-C to stop)..."
    sleep "$SLEEP_SECONDS"
done
