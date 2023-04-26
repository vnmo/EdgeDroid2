#BUILD_CMD = docker buildx build --push --platform linux/arm64,linux/amd64
BUILD_CMD = docker buildx build --platform linux/arm64
DOCKER_USER = molguin
IMG_REPO = $(DOCKER_USER)/edgedroid2

all: edgedroid-experiment-server edgedroid-experiment-client
.PHONY: all

edgedroid-experiment-client: Dockerfile
	$(BUILD_CMD) --target client -t $(IMG_REPO):experiment-client -f $< .

edgedroid-experiment-server: Dockerfile
	$(BUILD_CMD) --target server -t $(IMG_REPO):experiment-server -f $< .

clean:
	docker image rm $(IMG_REPO):experiment-client
	docker image rm $(IMG_REPO):experiment-server