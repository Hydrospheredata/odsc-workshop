FROM hydrosphere/odsc-workshop-base:$TAG

ADD ./release_drift_detector.py /src/

WORKDIR /src/
ENTRYPOINT [ "python", "release_drift_detector.py" ]
