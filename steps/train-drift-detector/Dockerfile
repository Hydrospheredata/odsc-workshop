FROM hydrosphere/odsc-workshop-base:$TAG

COPY *.py /src/
WORKDIR /src/

ENTRYPOINT [ "python", "train_drift_detector.py" ]
