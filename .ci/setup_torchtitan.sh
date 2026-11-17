# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared torchtitan setup for CI scripts.
# Usage: source .ci/setup_torchtitan.sh

TORCHTITAN_BRANCH="main"
TORCHTITAN_REPOSITORY="https://gitcode.com/GitHub_Trending/to/torchtitan.git"

_setup_torchtitan() {
    local target_dir="${1:-/tmp/torchtitan}"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local requirements_file="${script_dir}/../requirements.txt"
    local requirement_pattern='^[[:space:]]*torchtitan==[0-9][0-9A-Za-z.+-]*[[:space:]]*$'
    local torchtitan_requirement
    torchtitan_requirement="$(grep -E "${requirement_pattern}" "${requirements_file}")"
    local torchtitan_version="${torchtitan_requirement##*==}"
    torchtitan_version="${torchtitan_version//[[:space:]]/}"
    local torchtitan_commit="v${torchtitan_version}"

    echo "Preparing torchtitan at ${torchtitan_commit}..."

    echo "Cloning torchtitan source..."
    mkdir -p "$(dirname "$target_dir")"
    git clone --branch "$TORCHTITAN_BRANCH" \
        "$TORCHTITAN_REPOSITORY" "$target_dir"

    git -C "$target_dir" checkout "$torchtitan_commit"
}
