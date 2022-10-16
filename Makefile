BUILD_CMD = docker buildx build --push --platform linux/arm64,linux/amd64
DOCKER_USER = molguin
IMG_REPO = $(DOCKER_USER)/edgedroid2

all: edgedroid
.PHONY: all clean dlib

#login:
#	docker login -u $(DOCKER_USER)

#server: Dockerfile
#	$(BUILD_CMD) --target server -t $(IMG_REPO):server -f $< .
#
#client: Dockerfile
#	$(BUILD_CMD) --target client -t $(IMG_REPO):client -f $< .

edgedroid: Dockerfile
	$(BUILD_CMD) -t $(IMG_REPO):latest -f $< .

dlib: Dockerfile.dlib
	$(BUILD_CMD) -t $(IMG_REPO):dlib-base -f $< .

clean:
	docker image rm $(IMG_REPO):latest
