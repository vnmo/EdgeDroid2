BUILD_CMD = docker build
IMG_TAG = molguin/edgedroid2

all: server client
.PHONY: all

server: Dockerfile.server
	$(BUILD_CMD) -t $(IMG_TAG):server -f $^ .

client: Dockerfile.client
	$(BUILD_CMD) -t $(IMG_TAG):client -f $^ .