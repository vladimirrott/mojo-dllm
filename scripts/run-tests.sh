#!/usr/bin/env bash
# Build and run every tests/test_*.mojo, then check the total against the pin.
#
#   scripts/run-tests.sh            run all, compare with tests/test-count.txt
#   UPDATE_TEST_COUNT=1 scripts/run-tests.sh   rewrite the pin after adding tests
#
# Why the pin: Mojo's TestSuite exits 0 when it runs zero tests, and a renamed
# or mis-imported test file can silently drop out. A suite that passed and a
# suite that never ran look identical unless something counts.
#
# Tests are compiled with `mojo build` into build/tests/, so a failing file can
# be re-run without recompiling. TestSuite reports per-test times in
# milliseconds.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
if [ -z "${MOJO:-}" ]; then
    if command -v mojo >/dev/null 2>&1; then MOJO=mojo; else MOJO="pixi run mojo"; fi
fi
mkdir -p build/tests

files=(tests/test_*.mojo)
if [ "${#files[@]}" -eq 0 ] || [ ! -e "${files[0]}" ]; then
    echo "FAIL: no test files found under tests/" >&2
    exit 1
fi

total_passed=0
failed_files=()
for f in "${files[@]}"; do
    name="$(basename "$f" .mojo)"
    if ! build_out="$($MOJO build -I src "$f" -o "build/tests/$name" 2>&1)"; then
        printf '%s\n' "$build_out" | grep -v Crashpad >&2
        failed_files+=("$name (build)")
        continue
    fi
    run_out="$("build/tests/$name" 2>&1)"
    rc=$?
    summary="$(grep -E '^Summary' <<<"$run_out" | tail -1)"
    passed="$(sed -n 's/.* \([0-9][0-9]*\) passed .*/\1/p' <<<"$summary")"
    failed="$(sed -n 's/.* \([0-9][0-9]*\) failed .*/\1/p' <<<"$summary")"
    if [ "$rc" -ne 0 ] || [ -z "$summary" ] || [ "${failed:-1}" != "0" ] || [ "${passed:-0}" -eq 0 ]; then
        grep -E 'FAIL|Error|error' <<<"$run_out" | head -20 >&2
        failed_files+=("$name (rc=$rc, passed=${passed:-?}, failed=${failed:-?})")
        continue
    fi
    printf '  %-22s %3d passed\n' "$name" "$passed"
    total_passed=$((total_passed + passed))
done

if [ "${#failed_files[@]}" -gt 0 ]; then
    printf 'FAIL: %s\n' "${failed_files[@]}" >&2
    exit 1
fi

pin_file=tests/test-count.txt
if [ "${UPDATE_TEST_COUNT:-0}" = "1" ]; then
    echo "$total_passed" >"$pin_file"
    echo "pinned test count: $total_passed"
    exit 0
fi
pinned="$(cat "$pin_file" 2>/dev/null || echo missing)"
if [ "$pinned" != "$total_passed" ]; then
    echo "FAIL: $total_passed tests passed but $pin_file pins $pinned." >&2
    echo "      If you added or removed tests on purpose: UPDATE_TEST_COUNT=1 $0" >&2
    exit 1
fi
echo "OK: $total_passed tests passed (pinned $pinned)"
