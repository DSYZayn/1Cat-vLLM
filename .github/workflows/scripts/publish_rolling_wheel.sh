#!/usr/bin/env bash
set -euo pipefail

wheel_path="${1:?Wheel path is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${SOURCE_SHA:?SOURCE_SHA is required}"
: "${DAILY_TAG:?DAILY_TAG is required}"
: "${ROLLING_TAG:?ROLLING_TAG is required}"
: "${DAILY_TITLE:?DAILY_TITLE is required}"
: "${ROLLING_TITLE:?ROLLING_TITLE is required}"
: "${DAILY_NOTES_FILE:?DAILY_NOTES_FILE is required}"
: "${ROLLING_NOTES_FILE:?ROLLING_NOTES_FILE is required}"

ensure_release() {
  local tag="$1"
  local title="$2"
  local notes_file="$3"

  if gh release view "$tag" --repo "$GITHUB_REPOSITORY" >/dev/null 2>&1; then
    gh release edit "$tag" \
      --repo "$GITHUB_REPOSITORY" \
      --title "$title" \
      --notes-file "$notes_file" \
      --prerelease
  else
    gh release create "$tag" \
      --repo "$GITHUB_REPOSITORY" \
      --target "$SOURCE_SHA" \
      --title "$title" \
      --notes-file "$notes_file" \
      --prerelease
  fi
}

upload_without_overwrite() {
  local tag="$1"
  local asset_name
  asset_name="$(basename "$wheel_path")"

  if gh release view "$tag" --repo "$GITHUB_REPOSITORY" --json assets \
    --jq '.assets[].name' | grep -Fqx "$asset_name"; then
    echo "Asset ${asset_name} already exists in ${tag}; refusing to overwrite it."
    return 0
  fi

  gh release upload "$tag" "$wheel_path" --repo "$GITHUB_REPOSITORY"
}

ensure_release "$DAILY_TAG" "$DAILY_TITLE" "$DAILY_NOTES_FILE"
upload_without_overwrite "$DAILY_TAG"

ensure_release "$ROLLING_TAG" "$ROLLING_TITLE" "$ROLLING_NOTES_FILE"
upload_without_overwrite "$ROLLING_TAG"

echo "Published ${wheel_path} to ${DAILY_TAG} and ${ROLLING_TAG}."
