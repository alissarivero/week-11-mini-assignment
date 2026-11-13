.PHONY: data up down pipeline agents adaptive test reports

data:
	python3 scripts/generate_data.py

up:
	docker compose up --build -d

down:
	docker compose down

pipeline:
	python3 scripts/run_part1.py

agents:
	python3 scripts/run_part2.py

adaptive:
	python3 scripts/run_part3.py

test:
	PYTHONPATH=src python3 -m pytest tests -q

reports: pipeline agents adaptive

trigger-dag:
	docker compose exec airflow airflow dags trigger bike_demand_etl
