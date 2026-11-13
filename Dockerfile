FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-airflow.txt /requirements-airflow.txt
COPY docker/airflow-entrypoint.sh /airflow-entrypoint.sh
RUN chmod +x /airflow-entrypoint.sh

USER airflow
RUN pip install --no-cache-dir -r /requirements-airflow.txt

ENV PYTHONPATH=/opt/airflow/src \
    PROJECT_ROOT=/opt/airflow \
    AIRFLOW_HOME=/opt/airflow

WORKDIR /opt/airflow
ENTRYPOINT ["/airflow-entrypoint.sh"]
