BASE_IMG = workshop-base
TAG = latest
DOCKER_REGISTRY = 943173312784.dkr.ecr.eu-central-1.amazonaws.com
AWS_REGION = eu-central-1

# List any changed  files. We only include files in the notebooks directory.
# because that is the code in the docker image.
# In particular we exclude changes to the ksonnet configs.
CHANGED_FILES := $(shell git diff-files --relative=steps/)

ifeq ($(strip $(CHANGED_FILES)),)
# Changed files is empty; not dirty
# Don't include --dirty because it could be dirty if files outside the ones we care
# about changed.
GIT_VERSION := $(shell git describe --always)
else
GIT_VERSION := $(shell git describe --always)-dirty-$(shell git diff | shasum -a256 | cut -c -6)
endif

UNIQUE_TAG := $(shell date +v%Y%m%d)-$(GIT_VERSION)
all: latest

# Operations to work with the base image
create-base-repository:
	aws ecr create-repository --repository-name $(BASE_IMG)

build-base:
	docker build ${DOCKER_BUILD_PARAMS} -t $(DOCKER_REGISTRY)/$(BASE_IMG):$(UNIQUE_TAG) -f Dockerfile . --label=git-versions=$(GIT_VERSION)
	docker tag $(DOCKER_REGISTRY)/$(BASE_IMG):$(UNIQUE_TAG) $(DOCKER_REGISTRY)/$(BASE_IMG):latest

push-base:
	docker push $(DOCKER_REGISTRY)/$(BASE_IMG):latest

# Operations to work with worker steps
create-workers-repositories: 
	@for path in $(wildcard steps/*); do \
		aws ecr create-repository --repository-name mnist-pipeline-`basename $$path`; \
	done 

build-workers: 
	@for path in $(wildcard steps/*); do \
		if [[ ! -z "$$TARGET" ]] && [[ ${TARGET} != `basename $$path` ]]; then continue; fi; \
		cp utils/cloud.py $$path; \
		TAG=$(TAG) BASE_IMG=$(BASE_IMG) DOCKER_REGISTRY=$(DOCKER_REGISTRY) envsubst \
			'$$TAG,$$BASE_IMG,$$DOCKER_REGISTRY' < "$$path/Dockerfile" > "$$path/envsubDockerfile"; \
		docker build -t $(DOCKER_REGISTRY)/mnist-pipeline-`basename $$path`:$(UNIQUE_TAG) \
			-f $$path/envsubDockerfile --no-cache $$path; \
		docker tag $(DOCKER_REGISTRY)/mnist-pipeline-`basename $$path`:$(UNIQUE_TAG) \
			$(DOCKER_REGISTRY)/mnist-pipeline-`basename $$path`:$(TAG); \
		rm $$path/envsubDockerfile; \
	done

push-workers:
	@for path in $(wildcard steps/*); do \
		if [[ ! -z "$$TARGET" ]] && [[ ${TARGET} != `basename $$path` ]]; then continue; fi; \
		docker push $(DOCKER_REGISTRY)/mnist-pipeline-`basename $$path`:$(TAG); \
	done

# Combined operations
base: build-base push-base
workers: build-workers push-workers

build-test-workflow:
	@echo Compiling test workflow
	python3 workflows/kubeflow/aws/test.py 

run-test:
	@echo Sumbitting pipeline to the cluster
	python3 utils/kubeflow.py -f pipeline.tar.gz -k http://d4eeffac.kubeflow.odsc.k8s.hydrosphere.io 

run: build-test-workflow run-test