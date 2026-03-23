#!/bin/bash
curl -s -X POST \
  -H "Authorization: Bearer ${GITHUB_PAT}" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/fiftyeightandeight/cryptorocket/actions/workflows/250206308/dispatches \
  -d '{"ref":"main"}'
