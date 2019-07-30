FROM $DOCKER_REGISTRY/$BASE_IMG:$TAG

COPY *.py /src/
COPY *.npz /src/
WORKDIR /src/

ENTRYPOINT [ "python", "train_model.py" ]
