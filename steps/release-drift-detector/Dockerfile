FROM $DOCKER_REGISTRY/$BASE_IMG:$TAG

COPY *.py /src/
WORKDIR /src/

ENTRYPOINT [ "python", "release_drift_detector.py" ]
