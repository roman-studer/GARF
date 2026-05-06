#!/bin/bash
#SBATCH -t 1-0:00:00
#SBATCH -p performance
#SBATCH --gres=gpu:rtxA4500
#SBATCH --mem=48GB
#SBATCH --job-name=garf
#SBATCH --output=garf_%j.out
#SBATCH --error=garf_%j.err

set -o pipefail

hparams=""
experiment=""
params=""
cv=0

GITLAB_REGISTRY="cr.gitlab.fhnw.ch/i4ds/wristfracture:latest"
SIF_FILE="wristfracture_latest.sif"
REGISTRY_AUTH_URL="https://gitlab.fhnw.ch/jwt/auth?service=container_registry&scope=repository:i4ds/wristfracture:pull"

log_section() {
    echo
    echo "==== $1 ===="
}

fail() {
    echo "Error: $1" >&2
    exit 1
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        fail "Required command '$1' not found in PATH"
    fi
}

trim_cr() {
    printf '%s' "${1%$'\r'}"
}

cleanup() {
    rm -f "${HOME}/.docker/config.json"
}

trap cleanup EXIT

while getopts e:h:p:c: flag
do
    case "${flag}" in
        e) experiment=${OPTARG};;
        h) hparams=${OPTARG};;
        p) params=${OPTARG};;
        c) cv=${OPTARG};;
        *)
            fail "Invalid option. Usage: sbatch $0 -e EXPERIMENT [-h HPARAMS] [-p PARAMS] [-c CV]"
            ;;
    esac
done

log_section "Job context"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Started at: $(date -Is)"
echo "Experiment: ${experiment}"
echo "Hparams: ${hparams}"
echo "Params: ${params}"
echo "CV fold: ${cv}"

require_cmd git
require_cmd singularity
require_cmd base64

log_section "Repository update"
cd /cluster/group/wristfractures/GARF/ || fail "Could not cd into repository directory"
echo "Repository directory: $(pwd)"

git status
git fetch
git pull
echo "Pull successful"

log_section "Credential loading"

if [ -f .env ]; then
    echo "Loading credentials from $(pwd)/.env"
    echo "Credential entries found in .env:"
    grep -nE '^(GITLAB_USERNAME|GITLAB_TOKEN|GITLAB_PASSWORD)=' .env | sed -E 's/=(.*)$/=<redacted>/' || true

    if LC_ALL=C grep -q $'\r' .env; then
        echo "Warning: .env contains CRLF line endings. Credentials will be trimmed."
    fi

    # shellcheck disable=SC1091
    source .env
else
    fail ".env file not found. Create one with GITLAB_USERNAME and GITLAB_TOKEN"
fi

GITLAB_USERNAME="$(trim_cr "${GITLAB_USERNAME:-}")"
GITLAB_TOKEN="$(trim_cr "${GITLAB_TOKEN:-}")"
GITLAB_PASSWORD="$(trim_cr "${GITLAB_PASSWORD:-}")"

if [ -z "$GITLAB_TOKEN" ] && [ -n "$GITLAB_PASSWORD" ]; then
    GITLAB_TOKEN="$GITLAB_PASSWORD"
    echo "Notice: GITLAB_TOKEN is not set; using GITLAB_PASSWORD as registry token/password."
fi

if [ -z "$GITLAB_USERNAME" ] || [ -z "$GITLAB_TOKEN" ]; then
    fail "GITLAB_USERNAME and either GITLAB_TOKEN or GITLAB_PASSWORD must be set in .env"
fi

log_section "Credential diagnostics"
echo "Registry image: ${GITLAB_REGISTRY}"
echo "Singularity version: $(singularity --version 2>/dev/null || echo 'unknown')"
echo "GITLAB_USERNAME length: ${#GITLAB_USERNAME}"
echo "Registry token/password length: ${#GITLAB_TOKEN}"

if [[ "$GITLAB_USERNAME" =~ [[:space:]] ]]; then
    echo "Warning: GITLAB_USERNAME contains whitespace."
fi

if [[ "$GITLAB_TOKEN" =~ [[:space:]] ]]; then
    echo "Warning: GITLAB_TOKEN contains whitespace."
fi

if [[ "$GITLAB_TOKEN" == glpat-* ]]; then
    echo "Token looks like a personal access token. Ensure it has read_registry scope."
else
    echo "Token does not start with glpat-. If this is a deploy/project/group token, use its token-specific username."
fi

if [ -n "$GITLAB_PASSWORD" ] && [ "$GITLAB_PASSWORD" != "$GITLAB_TOKEN" ]; then
    echo "Notice: both GITLAB_PASSWORD and GITLAB_TOKEN are set and differ. This script uses GITLAB_TOKEN."
fi

log_section "Registry auth preflight"

if command -v curl >/dev/null 2>&1; then
    auth_probe="$(curl -sS -u "${GITLAB_USERNAME}:${GITLAB_TOKEN}" -w '\nHTTP_STATUS:%{http_code}' "${REGISTRY_AUTH_URL}" 2>&1)"
    auth_probe_status=$?

    if [ ${auth_probe_status} -ne 0 ]; then
        echo "Registry auth probe failed to execute. curl exit code: ${auth_probe_status}"
        echo "${auth_probe}"
    else
        auth_http_status="${auth_probe##*HTTP_STATUS:}"
        auth_body="${auth_probe%HTTP_STATUS:*}"
        auth_body="${auth_body%$'\n'}"

        echo "Registry auth HTTP status: ${auth_http_status}"

        if [[ "${auth_body}" == *'"token"'* ]]; then
            echo "Registry auth probe returned a JWT token."
        else
            echo "Registry auth response body: ${auth_body}"
        fi

        if [ "${auth_http_status}" != "200" ]; then
            echo "Warning: registry auth preflight failed."
            echo "Likely causes: wrong credentials, missing read_registry scope, wrong username for token type, or no image access."
        fi
    fi
else
    echo "curl not found; skipping registry auth preflight."
fi

log_section "Docker auth setup"

mkdir -p "${HOME}/.docker"

docker_auth="$(printf '%s' "${GITLAB_USERNAME}:${GITLAB_TOKEN}" | base64 | tr -d '\n')"

cat > "${HOME}/.docker/config.json" << EOF
{
    "auths": {
        "cr.gitlab.fhnw.ch": {
            "auth": "${docker_auth}"
        }
    }
}
EOF

chmod 600 "${HOME}/.docker/config.json"

echo "Docker auth config written to ${HOME}/.docker/config.json"
echo "Docker auth payload length: ${#docker_auth}"

log_section "Container pull"

echo "Pulling container image from GitLab registry"

SINGULARITY_DOCKER_USERNAME="${GITLAB_USERNAME}" \
SINGULARITY_DOCKER_PASSWORD="${GITLAB_TOKEN}" \
singularity pull --force "${SIF_FILE}" "docker://${GITLAB_REGISTRY}"

pull_status=$?

if [ ${pull_status} -ne 0 ]; then
    echo "Initial singularity pull failed with exit code ${pull_status}."
    echo "Retrying with singularity debug output."

    SINGULARITY_DOCKER_USERNAME="${GITLAB_USERNAME}" \
    SINGULARITY_DOCKER_PASSWORD="${GITLAB_TOKEN}" \
    singularity -d pull --force "${SIF_FILE}" "docker://${GITLAB_REGISTRY}"

    debug_pull_status=$?
    echo "Debug pull exit code: ${debug_pull_status}"

    fail "Failed to pull image from registry."
fi

echo "Image pull successful"

log_section "Training"

echo "Starting preprocessing/training pipeline"
echo "SIF file: ${SIF_FILE}"

SINGULARITYENV_LC_ALL=C.UTF-8 \
SINGULARITYENV_LANG=C.UTF-8 \
SINGULARITYENV_HPARAMS="${hparams}" \
SINGULARITYENV_EXPERIMENT="${experiment}" \
SINGULARITYENV_PARAMS="${params}" \
SINGULARITYENV_CV="${cv}" \
singularity exec \
    --bind /lib:/host_lib \
    --bind /lib64:/host_lib64 \
    --env LD_LIBRARY_PATH="/host_lib:/host_lib64:${LD_LIBRARY_PATH}" \
    -B /cluster/group/wristfractures/GARF:/workspace \
    --nv "./${SIF_FILE}" \
    bash -lc '
        set -euo pipefail

        cd /workspace

        echo "Inside container"
        echo "Working directory: $(pwd)"
        echo "Python: $(command -v python || true)"
        echo "Python version: $(python --version 2>&1 || true)"
        echo "Initial PATH: ${PATH}"

        if ! command -v uv >/dev/null 2>&1; then
            echo "uv not found in container. Installing uv into user environment."

            if command -v python >/dev/null 2>&1; then
                python -m pip install --user --no-cache-dir uv
            elif command -v python3 >/dev/null 2>&1; then
                python3 -m pip install --user --no-cache-dir uv
            else
                echo "Error: neither python nor python3 is available to install uv" >&2
                exit 1
            fi

            export PATH="${HOME}/.local/bin:${PATH}"
        fi

        if ! command -v uv >/dev/null 2>&1; then
            echo "Error: uv installation failed or uv is still not on PATH" >&2
            echo "PATH: ${PATH}" >&2
            exit 1
        fi

        echo "uv found at: $(command -v uv)"
        echo "uv version: $(uv --version)"

        echo "Running uv sync"
        uv sync --locked --extra post --no-dev

        echo "Starting training"

        if [ -z "${HPARAMS}" ]; then
            .venv/bin/python src/train.py \
                experiment="${EXPERIMENT}" \
                data.cfg.cv_fold="${CV}" \
                ${PARAMS}
        else
            .venv/bin/python src/train.py \
                hparams_search="${HPARAMS}" \
                experiment="${EXPERIMENT}" \
                data.cfg.cv_fold="${CV}" \
                ${PARAMS}
        fi
    '

train_status=$?

if [ ${train_status} -ne 0 ]; then
    fail "Training failed with exit code ${train_status}"
fi

log_section "Finished"
echo "Training finished"
echo "Finished at: $(date -Is)"