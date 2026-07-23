#!/bin/bash
# Poll for OpenRouter credit availability, then launch the finaldata judge sweep.
#
# Why a live call and not /api/v1/credits: that endpoint reported a stale
# 800/800.04 for a while after a top-up, so the only trustworthy readiness signal
# is an actual (tiny) completion succeeding. Polls every 5 min, gives up after 3h.
#
# Once credits are live this runs run_finaldata_judges_all.sh, which is resumable:
# the 63 already-complete (metric, condition) outputs are skipped, so only the
# remaining AITA/FLIP judges and the missing voice conditions are billed.
#
#   bash training_mitigations/slurm_scripts/wait_credits_then_judge.sh
set -uo pipefail
PROJECT=m000254
REPO=/projects/$PROJECT/sycophancy
PY=/projects/$PROJECT/envs/llm126/bin/python
cd "$REPO"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"

DEADLINE=$(( SECONDS + 3*60*60 ))
while (( SECONDS < DEADLINE )); do
  if "$PY" - <<'EOF'
import os, sys
from openai import OpenAI
c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
try:
    c.chat.completions.create(model="openai/gpt-5-mini", max_tokens=2000,
        extra_body={"reasoning": {"effort": "minimal"}},
        messages=[{"role": "user", "content": "Reply with the single digit 1"}])
    sys.exit(0)
except Exception as e:
    print(str(e)[:120], file=sys.stderr)
    sys.exit(1)
EOF
  then
    echo "CREDITS LIVE -- launching judge sweep at $(date -u +%H:%M:%SZ)"
    JOBS=8 WORKERS=10 bash training_mitigations/slurm_scripts/run_finaldata_judges_all.sh all
    echo "JUDGE SWEEP FINISHED"
    exit 0
  fi
  sleep 300
done
echo "GAVE UP: no credits after 3h"
exit 1
