#!/usr/bin/env bash
set -euo pipefail

: "${BASE_TAG:?BASE_TAG is required}"
: "${BUILD_DATE:?BUILD_DATE is required}"
: "${ROLLING_KIND:?ROLLING_KIND is required}"
: "${BASE_COMMIT:?BASE_COMMIT is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

daily_tag="${BASE_TAG}-${ROLLING_KIND}-${BUILD_DATE}"
previous_source_sha="$BASE_COMMIT"

if releases_pages="$(gh api --paginate --slurp \
  "repos/${GITHUB_REPOSITORY}/releases?per_page=100" 2>/dev/null)"; then
  releases_json="$(jq -c 'add // []' <<< "$releases_pages")"
  candidate_sha="$(jq -r \
    --arg build_date "$BUILD_DATE" \
    '
      [ .[]
        | select(.tag_name | test("^v[0-9]+\\.[0-9]+\\.[0-9]+-(rolling|native-rolling)-[0-9]{8}$"))
        | .tag_name[-8:] as $date
        | select($date < $build_date)
        | {date: $date, release: .}
      ]
      | sort_by(.date)
      | last // {}
      | .release as $release
      | (($release.body // "")
          | try (capture("source_sha: (?<sha>[0-9a-f]{40})").sha) catch "") as $body_sha
      | if ($body_sha | test("^[0-9a-f]{40}$"))
        then $body_sha
        else ($release.target_commitish // empty)
        end
    ' <<< "$releases_json")"
  if [[ "$candidate_sha" =~ ^[0-9a-f]{40}$ ]]; then
    previous_source_sha="$candidate_sha"
  fi
else
  echo "::warning::Could not list prior rolling releases; using the stable release commit as the range start." >&2
fi

{
  echo "daily_tag=$daily_tag"
  echo "previous_source_sha=$previous_source_sha"
}
