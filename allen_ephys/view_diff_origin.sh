#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
local_dir="${repo_root}/allen_ephys"

if [[ ! -d "${local_dir}" ]]; then
  echo "Error: local directory not found: ${local_dir}" >&2
  exit 1
fi

tmp="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp}"
}
trap cleanup EXIT

git clone \
  --branch master \
  --single-branch \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  https://github.com/AllenInstitute/AllenSDK.git \
  "${tmp}/AllenSDK"

git -C "${tmp}/AllenSDK" sparse-checkout set allensdk/ephys

upstream_dir="${tmp}/AllenSDK/allensdk/ephys"
local_subset_dir="${tmp}/local_ephys_subset"
formatted_upstream_dir="${tmp}/formatted_upstream"
formatted_local_dir="${tmp}/formatted_local"

mkdir -p "${local_subset_dir}"

while IFS= read -r upstream_path; do
  rel_path="${upstream_path#allensdk/ephys/}"
  local_path="${local_dir}/${rel_path}"
  subset_path="${local_subset_dir}/${rel_path}"

  if [[ -f "${local_path}" ]]; then
    mkdir -p "$(dirname "${subset_path}")"
    cp -a "${local_path}" "${subset_path}"
  fi
done < <(git -C "${tmp}/AllenSDK" ls-files allensdk/ephys)

mkdir -p "${formatted_upstream_dir}" "${formatted_local_dir}"
cp -a "${upstream_dir}/." "${formatted_upstream_dir}/"
cp -a "${local_subset_dir}/." "${formatted_local_dir}/"

if command -v uv >/dev/null 2>&1; then
  uv run ruff format "${formatted_upstream_dir}" "${formatted_local_dir}" >/dev/null
elif command -v ruff >/dev/null 2>&1; then
  ruff format "${formatted_upstream_dir}" "${formatted_local_dir}" >/dev/null
else
  echo "Error: ruff is required for formatted diff but was not found." >&2
  exit 1
fi

git diff --no-index "$@" -- "${formatted_upstream_dir}/" "${formatted_local_dir}/"
