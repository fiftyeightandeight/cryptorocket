#!/bin/bash
curl -s -X POST \
  -H "Authorization: Bearer ${GITHUB_PAT}" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/fiftyeightandeight/cryptorocket/actions/workflows/246102911/dispatches \
  -d '{"ref":"main"}'
