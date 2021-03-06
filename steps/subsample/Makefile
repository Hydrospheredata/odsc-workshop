IMAGE_NAME = mnist-pipeline-subsample
DOCKER_REGISTRY ?= hydrosphere
TAG ?= latest

all: test-local prepare build-container clean test-container push-container
release: test-local build-container clean push-container
release-raw: build-container clean push-container

test-local: 
	@echo Performing local run
	python3 subsample.py \
		--output-data-path s3://workshop-hydrosphere/mnist/data \
		--application-name kubeflow-mnist-app \
		--dev

# DOCKER_BUILD_OPTS env var can be used to configure build step
build-container:
	@echo Started building new image
	docker build ${DOCKER_BUILD_OPTS} -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

test-container:
	@echo Performing container run
	docker run -v ~/.aws:/root/.aws \
		$(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) \
		--output-data-path s3://workshop-hydrosphere/mnist/data \
		--application-name kubeflow-mnist-app \
		--dev

push-container: 
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf data cloud.py *.log output_data_path
