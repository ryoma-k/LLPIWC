singularity-bash:
		singularity shell docker://ryomak333/llpiwc:latest

docker-pull:
		docker pull ryomak333/llpiwc:latest

docker-build:
		docker build -t llpiwc -f docker_files/Dockerfile .

docker-up:
		docker compose -f docker_files/docker-compose.yml up -d

docker-stop:
		docker compose -f docker_files/docker-compose.yml stop

docker-bash:
		docker compose -f docker_files/docker-compose.yml exec work bash
