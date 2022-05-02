BUILD_CMD = docker buildx build --push --platform linux/arm64,linux/amd64
DOCKER_USER = molguin
IMG_REPO = $(DOCKER_USER)/edgedroid2

all: server client login
.PHONY: all clean

login:
	docker login -u $(DOCKER_USER)

server: Dockerfile.server login
	$(BUILD_CMD) -t $(IMG_REPO):server -f $< .

client: Dockerfile.client login
	$(BUILD_CMD) -t $(IMG_REPO):client -f $< .

clean:
	docker image rm $(IMG_REPO):client
	docker image rm $(IMG_REPO):server
	docker logout
