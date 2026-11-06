# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared torchtitan revision resolution and checkout for CI scripts.
# Usage: source .ci/setup_torchtitan.sh; _setup_torchtitan <target_dir>

TORCHTITAN_REPOSITORY="${TORCHTITAN_REPOSITORY:-https://gitcode.com/GitHub_Trending/to/torchtitan.git}"

_torchtitan_revision() {
    if [[ -n "${TORCHTITAN_COMMIT:-}" ]]; then
        printf '%s\n' "${TORCHTITAN_COMMIT}"
        return
    fi
    local requirements_file="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/requirements.txt}"
    local line requirement revision="" count=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        requirement="${line//[[:space:]]/}"
        if [[ "$requirement" =~ ^torchtitan==([0-9][0-9A-Za-z.+-]*)$ ]]; then
            revision="v${BASH_REMATCH[1]}"
            count=$((count + 1))
        elif [[ "$requirement" =~ ^torchtitan@git\+.+@([[:xdigit:]]{40})$ ]]; then
            revision="${BASH_REMATCH[1]}"
            count=$((count + 1))
        fi
    done < "$requirements_file"
    if [[ "$count" != 1 ]]; then
        echo "Expected exactly one TorchTitan release or full Git SHA requirement in ${requirements_file}." >&2
        return 1
    fi
    printf '%s\n' "$revision"
}

_setup_torchtitan() {
    local target_dir="${1:-/tmp/torchtitan}"
    local revision repository
    revision="$(_torchtitan_revision)" || return
    repository="${TORCHTITAN_REPO:-${TORCHTITAN_REPOSITORY}}"

    # Workflows may run from the target branch while sourcing this PR's script.
    # Let subsequent pip subprocesses fetch the pinned remat SHA from GitCode.
    local git_config_index="${GIT_CONFIG_COUNT:-0}"
    export "GIT_CONFIG_KEY_${git_config_index}=url.https://gitcode.com/hitwdy/remat.git.insteadOf"
    export "GIT_CONFIG_VALUE_${git_config_index}=https://github.com/meta-pytorch/remat.git"
    export GIT_CONFIG_COUNT="$((git_config_index + 1))"

    echo "Preparing torchtitan at ${revision}..."
    if [[ ! -e "$target_dir" ]]; then
        mkdir -p "$(dirname "$target_dir")"
        git clone --filter=blob:none --no-checkout "$repository" "$target_dir" || return
    elif ! git -C "$target_dir" rev-parse --git-dir >/dev/null 2>&1 \
        || [[ "$(git -C "$target_dir" rev-parse --show-toplevel)" != "$(cd "$target_dir" && pwd -P)" ]]; then
        echo "TorchTitan target exists and is not a Git checkout: ${target_dir}" >&2
        return 1
    elif [[ -n "$(git -C "$target_dir" status --porcelain)" ]]; then
        echo "Refusing to change a dirty TorchTitan checkout: ${target_dir}" >&2
        return 1
    fi

    # Mirrors may reject fetch-by-SHA while retaining the object in branch history.
    if ! git -C "$target_dir" cat-file -e "${revision}^{commit}" 2>/dev/null; then
        git -C "$target_dir" fetch --filter=blob:none origin \
            '+refs/heads/*:refs/remotes/origin/*' 'refs/tags/*:refs/tags/*' || return
        if ! git -C "$target_dir" cat-file -e "${revision}^{commit}" 2>/dev/null; then
            git -C "$target_dir" fetch --depth 1 origin "$revision" || return
        fi
    fi
    git -C "$target_dir" checkout --detach --quiet "$revision" || return
    git -C "$target_dir" rev-parse HEAD
}
