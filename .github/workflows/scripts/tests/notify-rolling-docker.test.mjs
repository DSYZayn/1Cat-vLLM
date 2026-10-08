import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

const workflow = readFileSync(new URL('../../full-native-wheel-publish.yml', import.meta.url), 'utf8');
const step = workflow.split('      - name: Notify rolling Docker image workflow')[1];
const script = step.split('        run: |')[1].split(/\r?\n/)
  .map(line => line.replace(/^          /, '')).join('\n');

function runNotification(overrides = {}) {
  const cwd = mkdtempSync(join(tmpdir(), 'rolling-notify-'));
  const mocks = `
gh() {
  printf '%s %s\n' "$GH_TOKEN" "$*" >> calls.log
  if [[ "$*" == *'/dispatches'* ]]; then
    if [[ "$MOCK_DISPATCH_EXIT" != 0 ]]; then
      echo 'gh: Bad credentials (HTTP 401)' >&2
    fi
    return "$MOCK_DISPATCH_EXIT"
  fi
  echo '2026-10-05T00:54:50Z'
  return "$MOCK_LOOKUP_EXIT"
}
jq() { printf '%s\n' '{"event_type":"rolling-wheel-published"}'; }
`;
  try {
    const result = spawnSync(process.env.BASH_BINARY || 'bash', ['-c', mocks + script], {
      cwd, encoding: 'utf8', timeout: 10000,
      env: {
        ...process.env,
        GH_TOKEN: 'source-token', DOCKER_DISPATCH_TOKEN: 'dispatch-token',
        GITHUB_REPOSITORY: 'DSYZayn/1Cat-vLLM',
        DOCKER_REPOSITORY: 'DSYZayn/1cat-vllm-v100-docker',
        RELEASE_TAG: 'v1.5.1-native-rolling-20261005', SOURCE_SHA: 'a'.repeat(40),
        WHEEL_VERSION: '1.5.1.post20261005', RUNNER_TEMP: '.',
        GITHUB_STEP_SUMMARY: 'summary.md', MOCK_DISPATCH_EXIT: '0',
        MOCK_LOOKUP_EXIT: '0', ...overrides,
      },
    });
    const read = name => existsSync(join(cwd, name)) ? readFileSync(join(cwd, name), 'utf8') : '';
    return { ...result, calls: read('calls.log'), summary: read('summary.md') };
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
}

test('reads with the repository token and dispatches with the optional token', () => {
  const result = runNotification();
  assert.equal(result.status, 0, result.error?.message || result.stderr);
  const calls = result.calls.trim().split('\n');
  assert.match(calls[0], /^source-token api repos\/DSYZayn\/1Cat-vLLM\/releases/);
  assert.match(calls[1], /^dispatch-token api repos\/DSYZayn\/1cat-vllm-v100-docker\/dispatches/);
  assert.match(result.stdout, /Triggered rolling Docker build/);
});
test('a missing dispatch secret leaves the published wheel successful', () => {
  const result = runNotification({ DOCKER_DISPATCH_TOKEN: '' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.calls, '');
  assert.match(result.stdout, /::warning::/);
});
test('a 401 dispatch failure leaves the wheel successful and reports a warning', () => {
  const result = runNotification({ MOCK_DISPATCH_EXIT: '1' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stderr, /HTTP 401/);
  assert.match(result.stdout, /::warning::Docker dispatch failed/);
  assert.match(result.summary, /scheduled discovery remains enabled/);
});
test('failed metadata lookup does not attempt a dispatch', () => {
  const result = runNotification({ MOCK_LOOKUP_EXIT: '1' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.calls.trim().split('\n').length, 1);
  assert.match(result.stdout, /::warning::Could not resolve dispatch metadata/);
});
