#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# 不带参数时安装的默认远程 skills。
DEFAULT_SKILLS=("gitcode-pr" "gitcode-pipeline")

# 各来源仓库地址（可用环境变量覆盖）。
GITCODE_REPO_URL="${DEFAULT_SKILLS_REPO_URL:-https://gitcode.com/cann-agent/skills.git}"
ASCEND_REPO_URL="${ASCEND_SKILLS_REPO_URL:-https://github.com/Ascend/agent-skills.git}"

CLONE_TIMEOUT="${CLONE_TIMEOUT:-30}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SKILLS_DIR}/../.." && pwd)"
REMOTE_DIR="${SKILLS_DIR}/_remote"
GITIGNORE="${PROJECT_ROOT}/.gitignore"

# 选择要安装的 skills：命令行参数优先，否则用默认集。
# 用法: install-default-skills.sh [skill-name ...]
#   install-default-skills.sh                          # 安装默认集（gitcode-pr、gitcode-pipeline）
#   install-default-skills.sh cann-operator-env-config # 按需安装 CANN 环境配置 skill（来自 GitHub Ascend 仓）
if [[ $# -gt 0 ]]; then
  REQUESTED_SKILLS=("$@")
else
  REQUESTED_SKILLS=("${DEFAULT_SKILLS[@]}")
fi

# skill -> 源仓库地址映射。新增非 GitCode 来源的 skill 时在此登记。
skill_repo_url() {
  case "$1" in
    cann-operator-env-config) printf '%s' "${ASCEND_REPO_URL}" ;;
    *) printf '%s' "${GITCODE_REPO_URL}" ;;
  esac
}

mkdir -p "${REMOTE_DIR}"

if command -v mktemp >/dev/null 2>&1; then
  TEMP_DIR="$(mktemp -d)"
else
  TEMP_DIR="/tmp/torchtitan_npu_skills_install_$$"
  mkdir -p "${TEMP_DIR}"
fi
trap 'rm -rf "${TEMP_DIR}"' EXIT

append_gitignore_once() {
  local entry="$1"
  if [[ -f "${GITIGNORE}" ]] && ! grep -qxF "${entry}" "${GITIGNORE}"; then
    printf '%s\n' "${entry}" >> "${GITIGNORE}"
  fi
}

# 把仓库地址映射成 TEMP_DIR 下唯一的克隆目录，保证同一仓库只克隆一次。
clone_dir_for() {
  printf '%s/repo-%s' "${TEMP_DIR}" "$(printf '%s' "$1" | tr -c 'A-Za-z0-9' '_')"
}

ensure_repo_cloned() {
  local url="$1"
  local dir
  dir="$(clone_dir_for "${url}")"

  if [[ -d "${dir}/skills" ]]; then
    return 0
  fi

  echo "Cloning skills repository: ${url}"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${CLONE_TIMEOUT}" git clone --depth 1 "${url}" "${dir}" || {
      echo "Error: failed to clone ${url} (check network/access)." >&2
      return 1
    }
  else
    GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME="${CLONE_TIMEOUT}" \
      git clone --depth 1 "${url}" "${dir}" || {
      echo "Error: failed to clone ${url} (check network/access)." >&2
      return 1
    }
  fi

  if [[ ! -d "${dir}/skills" ]]; then
    echo "Error: skills directory not found in ${url}" >&2
    return 1
  fi
}

install_skill() {
  local skill="$1"
  local url
  url="$(skill_repo_url "${skill}")"

  if ! ensure_repo_cloned "${url}"; then
    echo "Error: cannot install ${skill}; repository unavailable: ${url}" >&2
    exit 1
  fi

  local repo_dir
  repo_dir="$(clone_dir_for "${url}")"
  local source_dir="${repo_dir}/skills/${skill}"
  local target_dir="${REMOTE_DIR}/${skill}"
  local link_path="${SKILLS_DIR}/${skill}"

  if [[ ! -d "${source_dir}" ]]; then
    echo "Warning: remote skill not found in ${url}: ${skill}" >&2
    return
  fi

  if [[ -e "${link_path}" && ! -L "${link_path}" ]]; then
    echo "Error: ${link_path} exists and is not a symlink. Refusing to overwrite." >&2
    exit 1
  fi

  rm -rf "${target_dir}"
  cp -R "${source_dir}" "${target_dir}"
  ln -sfn "_remote/${skill}" "${link_path}"
  append_gitignore_once ".agents/skills/${skill}"
  echo "Installed skill: ${skill}"
}

append_gitignore_once ".agents/skills/_remote/"

for skill in "${REQUESTED_SKILLS[@]}"; do
  install_skill "${skill}"
done

echo "Skills installed successfully: ${REQUESTED_SKILLS[*]}"
