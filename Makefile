all: compile submit 

compile:
	python3 workflows/origin.py 
submit:
	python3 utils/kubeflow.py --file pipeline.tar.gz --kubeflow ml-pipeline-ui.k8s.hydrosphere.io 

release-workers:
	@for path in download train-drift-detector train-model; do \
		cd steps/$$path && make release; \
		cd ../../; \
	done
