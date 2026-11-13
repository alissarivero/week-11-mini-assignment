#!/usr/bin/env bash
set -euo pipefail

export AIRFLOW_HOME="${AIRFLOW_HOME:-/opt/airflow}"

if [[ "${1:-}" == "webserver" || "${1:-}" == "" ]]; then
  airflow db migrate
  airflow users create \
    --username "${_AIRFLOW_WWW_USER_USERNAME:-admin}" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password "${_AIRFLOW_WWW_USER_PASSWORD:-admin}" \
    || true
  airflow scheduler &
  exec airflow webserver --port 8080
fi

exec "$@"
