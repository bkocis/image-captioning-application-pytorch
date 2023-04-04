port=8081
app_name=image-captioning

up:
	docker-compose up

down:
	docker-compose down

build:
	flake8 --config=.flake8
	@if [ $$? -eq 0 ]; then \
		echo "Linting passed"; \
	else \
		echo "Linting failed"; \
	fi

	pytest
	@if [ $$? -eq 0 ]; then \
		echo "Tests passed"; \
	else \
		echo "Tests failed"; \
	fi
	docker-compose build

run_local:
	docker run -p 8081:8081 --rm image-captioning-app_python_build

run_as_gunicorn:
	gunicorn "application.main:app" -c application/config/gunicorn_config.py -k uvicorn.workers.UvicornWorker

deploy:
	docker build --tag=${app_name} .
	docker run -dit -p ${port}:${port} ${app_name}

git_push:
	flake8
	pytest
	git add .
	git commit -m "update"
	git push