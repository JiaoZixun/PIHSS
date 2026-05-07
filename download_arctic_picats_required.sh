#!/usr/bin/env bash
set -euo pipefail

# Download only the ARCTIC files required by the PI-CATS / PINN-HOI project:
#   1) raw_seqs.zip      -> MANO/object/egocam raw GT sequences
#   2) meta.zip          -> object_vtemplates, camera/misc metadata
#   3) splits_json.zip   -> official sequence split definitions, optional but useful
#   4) mano_v1_2.zip     -> MANO_RIGHT/LEFT model files used by the unified geometry engine
# It intentionally skips full images, cropped_images, processed splits, features, baselines, mocap, and SMPL-X.

PROJECT_ROOT="${PROJECT_ROOT:-/home/data/pihss_arctic_topconf}"
DATA_ROOT="${DATA_ROOT:-/home/data/arctic_picats_data}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT}/downloads}"
BUILD_OUT_DIR="${BUILD_OUT_DIR:-${PROJECT_ROOT}/outputs/picats_arctic/sequences}"
SPLIT_OUT_DIR="${SPLIT_OUT_DIR:-${PROJECT_ROOT}/outputs/picats_arctic/splits}"
NUM_OBJECT_POINTS="${NUM_OBJECT_POINTS:-1024}"
CONTACT_THRESH_M="${CONTACT_THRESH_M:-0.015}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SEQS="${MAX_SEQS:--1}"
RUN_BUILD="${RUN_BUILD:-1}"
FORCE_BUILD="${FORCE_BUILD:-0}"

ARCTIC_DATA_ROOT="${DATA_ROOT}/arctic_data/data"
MANO_ROOT="${DATA_ROOT}/body_models/mano"

RAW_SEQS_URL='https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile=arctic_release/c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/v1_0/data/raw_seqs.zip'
META_URL='https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile=arctic_release/c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/v1_0/data/meta.zip'
SPLITS_JSON_URL='https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile=arctic_release/c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/v1_0/data/splits_json.zip'
MANO_URL='https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip'

need_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] Missing environment variable: ${name}" >&2
    return 1
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 1; }
}

fetch_post() {
  local url="$1"
  local out="$2"
  local username="$3"
  local password="$4"
  local label="$5"

  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    echo "[SKIP] ${label}: ${out} already exists ($(du -h "$out" | awk '{print $1}'))"
    return 0
  fi

  echo "[DOWNLOAD] ${label} -> ${out}"
  curl -fLk --retry 5 --retry-delay 5 --continue-at - \
    --data-urlencode "username=${username}" \
    --data-urlencode "password=${password}" \
    -o "${out}" \
    "${url}"

  if [[ ! -s "$out" ]]; then
    echo "[ERROR] Downloaded file is empty: ${out}" >&2
    exit 1
  fi
}

unzip_if_needed() {
  local zip_p="$1"
  local dst="$2"
  local sentinel="$3"
  local label="$4"

  if [[ -e "$sentinel" ]]; then
    echo "[SKIP] ${label}: found ${sentinel}"
    return 0
  fi
  echo "[UNZIP] ${label}: ${zip_p} -> ${dst}"
  mkdir -p "$dst"
  unzip -q -o "$zip_p" -d "$dst"
}

verify_zip() {
  local zip_p="$1"
  echo "[CHECK] unzip -t ${zip_p}"
  unzip -tq "$zip_p" >/dev/null
}

main() {
  require_cmd curl
  require_cmd unzip
  require_cmd python

  need_env ARCTIC_USERNAME
  need_env ARCTIC_PASSWORD
  need_env MANO_USERNAME
  need_env MANO_PASSWORD

  echo "================ ARCTIC PI-CATS required downloader ================"
  echo "PROJECT_ROOT      = ${PROJECT_ROOT}"
  echo "DATA_ROOT         = ${DATA_ROOT}"
  echo "ARCTIC_DATA_ROOT  = ${ARCTIC_DATA_ROOT}"
  echo "MANO_ROOT         = ${MANO_ROOT}"
  echo "RUN_BUILD         = ${RUN_BUILD}"
  echo "MAX_SEQS          = ${MAX_SEQS}"
  echo "===================================================================="

  mkdir -p "${DOWNLOAD_DIR}" "${ARCTIC_DATA_ROOT}" "${MANO_ROOT}"

  fetch_post "$RAW_SEQS_URL"     "${DOWNLOAD_DIR}/raw_seqs.zip"     "$ARCTIC_USERNAME" "$ARCTIC_PASSWORD" "ARCTIC raw_seqs"
  fetch_post "$META_URL"         "${DOWNLOAD_DIR}/meta.zip"         "$ARCTIC_USERNAME" "$ARCTIC_PASSWORD" "ARCTIC meta"
  fetch_post "$SPLITS_JSON_URL"  "${DOWNLOAD_DIR}/splits_json.zip"  "$ARCTIC_USERNAME" "$ARCTIC_PASSWORD" "ARCTIC splits_json"
  fetch_post "$MANO_URL"         "${DOWNLOAD_DIR}/mano_v1_2.zip"    "$MANO_USERNAME"   "$MANO_PASSWORD"   "MANO v1.2"

  verify_zip "${DOWNLOAD_DIR}/raw_seqs.zip"
  verify_zip "${DOWNLOAD_DIR}/meta.zip"
  verify_zip "${DOWNLOAD_DIR}/splits_json.zip"
  verify_zip "${DOWNLOAD_DIR}/mano_v1_2.zip"

  unzip_if_needed "${DOWNLOAD_DIR}/raw_seqs.zip"    "${ARCTIC_DATA_ROOT}" "${ARCTIC_DATA_ROOT}/raw_seqs" "raw_seqs"
  unzip_if_needed "${DOWNLOAD_DIR}/meta.zip"        "${ARCTIC_DATA_ROOT}" "${ARCTIC_DATA_ROOT}/meta" "meta"
  unzip_if_needed "${DOWNLOAD_DIR}/splits_json.zip" "${ARCTIC_DATA_ROOT}" "${ARCTIC_DATA_ROOT}/splits_json" "splits_json"

  if [[ ! -e "${MANO_ROOT}/MANO_RIGHT.pkl" && ! -e "${MANO_ROOT}/MANO_RIGHT.npz" ]]; then
    echo "[UNZIP] MANO v1.2 -> ${MANO_ROOT}"
    tmp_mano="${DATA_ROOT}/_tmp_mano_unpack"
    rm -rf "$tmp_mano"
    mkdir -p "$tmp_mano"
    unzip -q -o "${DOWNLOAD_DIR}/mano_v1_2.zip" -d "$tmp_mano"
    mano_models_dir="$(find "$tmp_mano" -type d -name models | head -n 1 || true)"
    if [[ -z "$mano_models_dir" ]]; then
      echo "[ERROR] Cannot find MANO models directory after unzipping ${DOWNLOAD_DIR}/mano_v1_2.zip" >&2
      find "$tmp_mano" -maxdepth 3 -type f | head -50 >&2
      exit 1
    fi
    mkdir -p "${MANO_ROOT}"
    cp -a "${mano_models_dir}/." "${MANO_ROOT}/"
    rm -rf "$tmp_mano"
  else
    echo "[SKIP] MANO models: found ${MANO_ROOT}/MANO_RIGHT.*"
  fi

  echo "[VERIFY] required local structure"
  test -d "${ARCTIC_DATA_ROOT}/raw_seqs"
  test -d "${ARCTIC_DATA_ROOT}/meta/object_vtemplates"
  test -d "${ARCTIC_DATA_ROOT}/splits_json"
  if [[ ! -e "${MANO_ROOT}/MANO_RIGHT.pkl" && ! -e "${MANO_ROOT}/MANO_RIGHT.npz" ]]; then
    echo "[ERROR] MANO_RIGHT model not found under ${MANO_ROOT}" >&2
    ls -la "${MANO_ROOT}" >&2 || true
    exit 1
  fi
  if [[ ! -e "${MANO_ROOT}/MANO_LEFT.pkl" && ! -e "${MANO_ROOT}/MANO_LEFT.npz" ]]; then
    echo "[ERROR] MANO_LEFT model not found under ${MANO_ROOT}" >&2
    ls -la "${MANO_ROOT}" >&2 || true
    exit 1
  fi

  echo "[OK] Downloaded raw data needed by PI-CATS."
  echo "     ARCTIC root for 00 script: ${ARCTIC_DATA_ROOT}"
  echo "     MANO root for 00 script:   ${MANO_ROOT}"

  if [[ "${RUN_BUILD}" == "1" ]]; then
    echo "[BUILD] Running 00_build_arctic_picats_dataset.py"
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
    force_arg=()
    if [[ "${FORCE_BUILD}" == "1" ]]; then
      force_arg=(--force)
    fi
    python scripts/00_build_arctic_picats_dataset.py \
      --arctic-root "${ARCTIC_DATA_ROOT}" \
      --mano-root "${MANO_ROOT}" \
      --out-dir "${BUILD_OUT_DIR}" \
      --split-out "${SPLIT_OUT_DIR}" \
      --num-object-points "${NUM_OBJECT_POINTS}" \
      --contact-thresh-m "${CONTACT_THRESH_M}" \
      --device "${DEVICE}" \
      --max-seqs "${MAX_SEQS}" \
      "${force_arg[@]}"
  else
    echo "[INFO] RUN_BUILD=0, skip 00 script. You can run:"
    echo "cd ${PROJECT_ROOT}"
    echo "export PYTHONPATH=${PROJECT_ROOT}/src:\$PYTHONPATH"
    echo "python scripts/00_build_arctic_picats_dataset.py --arctic-root ${ARCTIC_DATA_ROOT} --mano-root ${MANO_ROOT} --out-dir ${BUILD_OUT_DIR} --split-out ${SPLIT_OUT_DIR} --num-object-points ${NUM_OBJECT_POINTS} --contact-thresh-m ${CONTACT_THRESH_M} --device ${DEVICE} --max-seqs ${MAX_SEQS}"
  fi
}

main "$@"
