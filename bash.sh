# check arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --docker) 
    BUILD_DOCKER=true;
    shift # past argument
    ;;
    --aws)
    BUILD_AWS=true;
    shift # past argument
    ;;
    --build-base)
    BUILD_BASE=true;
    shift # past argument
    ;;
    --build-workers)
    BUILD_WORKERS=true;
    shift # past argument
    ;;
    --no-cache)
    cache="--no-cache";
    shift # past argument
    ;;
    --compile-origin)
    COMPILE_ORIGIN_PIPELINE=true;
    shift # past argument
    ;;
    --run-origin)
    RUN_ORIGIN_PIPELINE=true;
    shift # past argument
    ;;
    --compile-subsample)
    COMPILE_SUBSAMPLE_PIPELINE=true;
    shift # past argument
    ;;
    --run-subsample)
    RUN_SUBSAMPLE_PIPELINE=true;
    shift # past argument
    ;;
    --namespace)
    NAMESPACE=$2
    shift # past argument
    shift # past value
    ;;
  esac
done

# Check environment 
if !([ -z ${BUILD_BASE+x} ] || [ -z ${BUILD_WORKERS+x} ]); then
  if [ -z "$BUILD_DOCKER" ] && [ -z "$BUILD_AWS" ]; then
    echo "Either --aws/-a or --docker/-d flags should be passed"
    exit 1
  fi
fi

# Add default environment variables
[ -z "$DOCKER_ACCOUNT" ] && DOCKER_ACCOUNT="hydrosphere"
[ -z "$TAG" ] && TAG="latest"
[ -z "$DIRECTORY" ] && DIRECTORY="."

# Build base if specified
if [[ $BUILD_BASE && $BUILD_DOCKER ]]; then
  echo "Building base image for Docker"
  docker build -t $DOCKER_ACCOUNT/odsc-workshop-base:$TAG -f Dockerfile $cache .
  docker push $DOCKER_ACCOUNT/odsc-workshop-base:$TAG
fi 

# Build workers if specified
if [[ $BUILD_WORKERS && $BUILD_DOCKER ]]; then
  echo "Building stage images for Docker"
  for path in steps/*/; do 
    IFS=$'_'; arr=($path); unset IFS;
    TAG=$TAG envsubst '$TAG' < "$path/Dockerfile" > "$path/envsubDockerfile"
    docker build -t $DOCKER_ACCOUNT/mnist-pipeline-${arr[1]}:$TAG \
      -f "$path/envsubDockerfile" $cache $path
    docker push $DOCKER_ACCOUNT/mnist-pipeline-${arr[1]}:$TAG
    rm "$path/envsubDockerfile"
  done
fi 

# Package files for AWS Lambda
if [[ $BUILD_WORKERS && $BUILD_AWS ]]; then
  echo "Copying functions for packaging"
  for path in steps/*/; do 
    echo $path
    cp $path/*.py serverless
  done
fi 

# Compile origin and subsample piplines
if [[ $COMPILE_ORIGIN_PIPELINE ]]; then
  echo "Compiling origin pipeline"
  python3 workflows/origin.py -n $NAMESPACE
  rm pipeline.tar.gz pipeline.yaml\'\'
fi

if [[ $COMPILE_SUBSAMPLE_PIPELINE ]]; then
  echo "Compiling subsample pipeline"
  python3 workflows/subsample.py -n $NAMESPACE
  rm pipeline.tar.gz pipeline.yaml\'\'
fi

# Run origin and subsample pipelines
if [[ $RUN_ORIGIN_PIPELINE ]]; then
  echo "Running origin pipeline"
  python3 kubeflow_client.py -n $NAMESPACE -f pipeline.yaml
fi

if [[ $RUN_SUBSAMPLE_PIPELINE ]]; then
  echo "Running subsample pipeline"
  python3 kubeflow_client.py -n $NAMESPACE -f pipeline.yaml
fi