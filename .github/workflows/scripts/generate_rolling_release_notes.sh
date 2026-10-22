#!/usr/bin/env bash
set -euo pipefail

output_file="${1:?Output file is required}"
: "${SOURCE_SHA:?SOURCE_SHA is required}"
: "${PREVIOUS_SOURCE_SHA:?PREVIOUS_SOURCE_SHA is required}"
: "${BUILD_DATE:?BUILD_DATE is required}"
: "${BASE_RELEASE_REPO:?BASE_RELEASE_REPO is required}"
: "${BASE_TAG:?BASE_TAG is required}"
: "${DAILY_TAG:?DAILY_TAG is required}"
: "${ROLLING_TAG:?ROLLING_TAG is required}"
: "${EXPECTED_VERSION:?EXPECTED_VERSION is required}"
: "${WHEEL_KIND:?WHEEL_KIND is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

if [[ ! "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Invalid source SHA: $SOURCE_SHA" >&2
  exit 1
fi
if [[ ! "$PREVIOUS_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Invalid previous source SHA: $PREVIOUS_SOURCE_SHA" >&2
  exit 1
fi

commit_range="${PREVIOUS_SOURCE_SHA}..${SOURCE_SHA}"
mapfile -t commits < <(git log --no-merges --format='%H%x09%s' "$commit_range")
server_url="${GITHUB_SERVER_URL:-https://github.com}"
repo_url="${server_url}/${GITHUB_REPOSITORY}"

print_commits() {
  local requested_category="$1"
  local found=false
  local entry sha subject lower_subject category

  for entry in "${commits[@]}"; do
    sha="${entry%%$'\t'*}"
    subject="${entry#*$'\t'}"
    lower_subject="${subject,,}"
    category="other"
    case "$lower_subject" in
      feat:*|feat\ *|feat\(*|feature:*|feature\ *|support:*|support\ *|enhance:*|enhance\ *|\[feature\]*|\[feat\]*)
        category="features"
        ;;
      fix:*|fix\ *|fix\(*|bugfix:*|bugfix\ *|hotfix:*|hotfix\ *|repair:*|repair\ *|patch:*|patch\ *|\[bugfix\]*|\[bug\]*)
        category="bug_fixes"
        ;;
    esac
    if [[ "$category" == "$requested_category" ]]; then
      printf -- '- [`%s`](%s/commit/%s) %s\n' \
        "${sha:0:9}" "$repo_url" "$sha" "$subject"
      found=true
    fi
  done

  if [[ "$found" != true ]]; then
    echo "- None."
  fi
}

write_release_metadata() {
  local title_prefix
  case "$WHEEL_KIND" in
    rolling)
      title_prefix="Rolling"
      ;;
    native)
      title_prefix="Native Rolling"
      ;;
    *)
      title_prefix="${WHEEL_KIND^} Rolling"
      ;;
  esac
  echo "# ${title_prefix} Wheel ${BUILD_DATE}"
  echo
  echo "<!-- source_sha: ${SOURCE_SHA} -->"
  echo "<!-- previous_source_sha: ${PREVIOUS_SOURCE_SHA} -->"
  echo
  markdown_code='`'
  echo "- Wheel version: ${markdown_code}${EXPECTED_VERSION}${markdown_code}"
  echo "- Source: [${markdown_code}${SOURCE_SHA:0:12}${markdown_code}](${repo_url}/commit/${SOURCE_SHA})"
  echo "- Range: [${markdown_code}${PREVIOUS_SOURCE_SHA:0:12}...${SOURCE_SHA:0:12}${markdown_code}](${repo_url}/compare/${PREVIOUS_SOURCE_SHA}...${SOURCE_SHA})"
  echo "- Stable native baseline: ${markdown_code}${BASE_RELEASE_REPO}@${BASE_TAG}${markdown_code}"
  echo "- Latest alias: [${markdown_code}${ROLLING_TAG}${markdown_code}](${repo_url}/releases/tag/${ROLLING_TAG})"
  echo
}

write_commit_sections() {
  echo "## Features"
  print_commits features
  echo
  echo "## Bug Fixes"
  print_commits bug_fixes
  echo
  echo "## Other Changes"
  print_commits other
}

write_release_footer() {
  echo
  markdown_code='`'
  echo "This prerelease is published under [${markdown_code}${DAILY_TAG}${markdown_code}](${repo_url}/releases/tag/${DAILY_TAG}) and retained for historical installation."
}

commit_manifest="$(
  for entry in "${commits[@]}"; do
    sha="${entry%%$'\t'*}"
    subject="${entry#*$'\t'}"
    printf '%s %s\n' "${sha:0:9}" "$subject"
  done
)"

fixed_notes_file="${output_file}.fixed"
{
  write_release_metadata
  echo "## Summary"
  echo "AI summary unavailable; the fixed template below records every commit in the build range."
  echo
  write_commit_sections
  write_release_footer
} > "$fixed_notes_file"

ai_summary=""
if [[ "${AI_SUMMARY_ENABLED:-true}" == "true" && -n "${GH_TOKEN:-}" && -n "$commit_manifest" ]]; then
  ai_api_url="${AI_API_URL:-https://models.github.ai/inference/chat/completions}"
  ai_model="${AI_MODEL:-openai/gpt-4o-mini}"
  prompt="$(printf '%s\n\nCommits:\n%s' \
    'Summarize the following commits for a software release note. Return Markdown only, with concise sections named Summary, Features, Bug Fixes, and Other Changes. Do not invent changes, and preserve uncertainty when a commit message is unclear.' \
    "$commit_manifest")"
  request_body="$(jq -n \
    --arg model "$ai_model" \
    --arg prompt "$prompt" \
    '{model: $model, messages: [
      {role: "system", content: "You write accurate engineering release notes."},
      {role: "user", content: $prompt}
    ], temperature: 0.2, max_tokens: 1200}')"

  if response="$(curl --fail --silent --show-error --location \
    --max-time "${AI_TIMEOUT_SECONDS:-20}" \
    --header "Authorization: Bearer ${GH_TOKEN}" \
    --header "Content-Type: application/json" \
    --header "X-GitHub-Api-Version: 2022-11-28" \
    --data "$request_body" "$ai_api_url" 2>/dev/null)"; then
    if ! ai_summary="$(jq -r \
      '(.choices[0].message.content // empty) | select(type == "string")' \
      <<< "$response")"; then
      ai_summary=""
    elif [[ -z "$(printf '%s' "$ai_summary" | tr -d '[:space:]')" || \
      "${#ai_summary}" -gt 12000 ]] || \
      ! grep -Eiq '(summary|features|bug fixes|other changes)' <<< "$ai_summary"; then
      ai_summary=""
    fi
  fi
fi

if [[ -n "$ai_summary" ]]; then
  {
    write_release_metadata
    echo "## AI Summary"
    printf '%s\n' "$ai_summary"
    echo
    echo "## Commits"
    echo "The complete commit list is retained below for traceability."
    write_commit_sections
    write_release_footer
  } > "$output_file"
  echo "Generated AI-assisted release notes for ${#commits[@]} commits: ${output_file}"
else
  cp "$fixed_notes_file" "$output_file"
  echo "::warning::AI release-note generation was unavailable; using the fixed commit-list template." >&2
  echo "Generated fallback release notes for ${#commits[@]} commits: ${output_file}"
fi
