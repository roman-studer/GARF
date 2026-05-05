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
    esac
done

log_section "Job context"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Started at: $(date -Is)"

require_cmd git
require_cmd singularity
require_cmd base64

echo "Pulling latest version of repository"
cd /cluster/group/wristfractures/GARF/ || exit
echo "Repository directory: $(pwd)"
git status
git fetch
git pull
echo "Pull successful"

# Load credentials from .env file
if [ -f .env ]; then
    log_section "Credential loading"
    echo "Loading credentials from $(pwd)/.env"
    echo "Credential entries found in .env:"
    grep -nE '^(GITLAB_USERNAME|GITLAB_TOKEN|GITLAB_PASSWORD)=' .env | sed -E 's/=(.*)$/=<redacted>/' || true
    if LC_ALL=C grep -q $'\r' .env; then
        echo "Warning: .env contains CRLF line endings. The script will trim trailing carriage returns from loaded credentials."
    fi
    # shellcheck disable=SC1091
    source .env
else
    echo "Error: .env file not found. Please create a .env file with GITLAB_USERNAME and GITLAB_TOKEN"
    exit 1
fi

GITLAB_USERNAME="$(trim_cr "${GITLAB_USERNAME:-}")"
GITLAB_TOKEN="$(trim_cr "${GITLAB_TOKEN:-}")"
GITLAB_PASSWORD="$(trim_cr "${GITLAB_PASSWORD:-}")"

# Check if credentials are loaded
if [ -z "$GITLAB_USERNAME" ] || [ -z "$GITLAB_TOKEN" ]; then
    echo "Error: GITLAB_USERNAME or GITLAB_TOKEN not set in .env file"
    if [ -n "$GITLAB_PASSWORD" ]; then
        echo "Warning: GITLAB_PASSWORD is set, but run_experiment.sh authenticates with GITLAB_TOKEN."
    fi
    exit 1
fi

log_section "Credential diagnostics"
echo "Registry image: ${GITLAB_REGISTRY}"
echo "Singularity version: $(singularity --version 2>/dev/null || echo 'unknown')"
echo "GITLAB_USERNAME length: ${#GITLAB_USERNAME}"
echo "GITLAB_TOKEN length: ${#GITLAB_TOKEN}"

if [[ "$GITLAB_USERNAME" =~ [[:space:]] ]]; then
    echo "Warning: GITLAB_USERNAME contains whitespace."
fi

if [[ "$GITLAB_TOKEN" =~ [[:space:]] ]]; then
    echo "Warning: GITLAB_TOKEN contains whitespace."
fi

if [[ "$GITLAB_TOKEN" == glpat-* ]]; then
    echo "Token prefix indicates a personal access token. Ensure it has read_registry and that GITLAB_USERNAME is your GitLab username."
else
    echo "Token does not start with glpat-. If this is a deploy, project, or group token, GITLAB_USERNAME must be the token-specific username."
fi

if [ -n "$GITLAB_PASSWORD" ] && [ "$GITLAB_PASSWORD" != "$GITLAB_TOKEN" ]; then
    echo "Notice: both GITLAB_PASSWORD and GITLAB_TOKEN are set and differ. This script uses GITLAB_TOKEN."
fi

log_section "Registry auth preflight"
if command -v curl >/dev/null 2>&1; then
    auth_probe="$(curl -sS -u "${GITLAB_USERNAME}:${GITLAB_TOKEN}" -w '\nHTTP_STATUS:%{http_code}' "${REGISTRY_AUTH_URL}" 2>&1)"
    auth_probe_status=$?
    if [ ${auth_probe_status} -ne 0 ]; then
        echo "Registry auth probe failed to execute (curl exit ${auth_probe_status})."
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
            echo "Warning: registry auth preflight failed. Likely causes are wrong credentials, missing read_registry scope, wrong username for the token type, or missing access to ${GITLAB_REGISTRY}."
        fi
    fi
else
    echo "curl not found; skipping registry auth preflight."
fi

# Create Docker authentication configuration
echo "Setting up Docker authentication"
mkdir -p ~/.docker
docker_auth="$(printf '%s' "${GITLAB_USERNAME}:${GITLAB_TOKEN}" | base64 | tr -d '\n')"
cat > ~/.docker/config.json << EOF
{
    "auths": {
        "cr.gitlab.fhnw.ch": {
            "auth": "${docker_auth}"
        }
    }
}
EOF
chmod 600 ~/.docker/config.json
echo "Docker auth config written to ${HOME}/.docker/config.json"
echo "Docker auth payload length: ${#docker_auth}"

echo "Pulling container image from GitLab registry"
SINGULARITY_DOCKER_USERNAME=${GITLAB_USERNAME} SINGULARITY_DOCKER_PASSWORD=${GITLAB_TOKEN} \
singularity pull --force ${SIF_FILE} docker://${GITLAB_REGISTRY}
pull_status=$?
if [ ${pull_status} -ne 0 ]; then
    echo "Initial singularity pull failed with exit code ${pull_status}."
    echo "Retrying with 'singularity -d pull' for additional diagnostics."
    SINGULARITY_DOCKER_USERNAME=${GITLAB_USERNAME} SINGULARITY_DOCKER_PASSWORD=${GITLAB_TOKEN} \
    singularity -d pull --force ${SIF_FILE} docker://${GITLAB_REGISTRY}
    debug_pull_status=$?
    echo "Debug pull exit code: ${debug_pull_status}"
    echo "Failed to pull image from registry. Common causes: invalid token, missing read_registry scope, wrong username for the token type, or no access to ${GITLAB_REGISTRY}."
    exit 1
fi
echo "Image pull successful"

echo "in dir is $in_dir"
echo "out dir is $out_dir"

echo "Starting preprocessing pipe"
SINGULARITYENV_LC_ALL=C.UTF-8 \
SINGULARITYENV_LANG=C.UTF-8 \
singularity exec --bind /lib:/host_lib --bind /lib64:/host_lib64 --env LD_LIBRARY_PATH=/host_lib:/host_lib64:$LD_LIBRARY_PATH \
-B /cluster/group/wristfractures/wristfracture:/workspace \
--nv ./${SIF_FILE} \
bash -c "cd /workspace && python3 -m pip install -r requirements.txt && \
if [ -z "$hparams" ]
then
    python3 src/train.py experiment=$experiment data.cfg.cv_fold=$cv $params
else
    python3 src/train.py hparams_search=$hparams experiment=$experiment data.cfg.cv_fold=$cv $params
fi"
echo "Training finished"
