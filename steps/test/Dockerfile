FROM hydrosphere/odsc-workshop-base:$TAG

ADD ./test.py /src/

WORKDIR /src/
ENTRYPOINT [ "python", "test.py" ]
