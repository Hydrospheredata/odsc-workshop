# check arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -d|--docker) 
    BUILD_DOCKER=true;
    shift # past argument
    ;;
    -a|--aws)
    BUILD_AWS=true;
    shift # past argument
    ;;
    -b|--build-base)
    BUILD_BASE=true;
    shift # past argument
    ;;
    -w|--build-workers)
    BUILD_WORKERS=true;
    shift # past argument
    ;;
    --no-cache)
    cache="--no-cache";
    shift # past argument
    ;;
    -oc|--origin-compile)
    COMPILE_ORIGIN_PIPELINE=true;
    shift # past argument
    ;;
    -or|--origin-run)
    RUN_ORIGIN_PIPELINE=true;
    shift # past argument
    ;;
    -sc|--sampling-compile)
    COMPILE_SAMPLING_PIPELINE=true;
    shift # past argument
    ;;
    -sr|--sampling-run)
    RUN_SAMPLING_PIPELINE=true;
    shift # past argument
    ;;
    -n|--namespace)
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
  docker build -t $DOCKER_ACCOUNT/odsc-workshop-base:$TAG -f baseDockerfile $cache .
  docker push $DOCKER_ACCOUNT/odsc-workshop-base:$TAG
fi 

# Build workers if specified
if [[ $BUILD_WORKERS && $BUILD_DOCKER ]]; then
  echo "Building stage images for Docker"
  for path in 01_download 01_sample 02_train-model 02_train-autoencoder 03_release-model 03_release-autoencoder 04_deploy 05_test; do 
    IFS=$'_'; arr=($path); unset IFS;
    TAG=$TAG envsubst '$TAG' < "$path/Dockerfile" > "$path/SubsDockerfile"
    docker build -t $DOCKER_ACCOUNT/mnist-pipeline-${arr[1]}:$TAG \
      -f "$path/SubsDockerfile" $cache $path
    docker push $DOCKER_ACCOUNT/mnist-pipeline-${arr[1]}:$TAG
    rm "$path/SubsDockerfile"
  done
fi 

# Package files for AWS Lambda
if [[ $BUILD_WORKERS && $BUILD_AWS ]]; then
  echo "Zipping stage steps for AWS Lambda"
  for path in 01_download 01_sample 02_train-model 02_train-autoencoder 03_release-model 03_release-autoencoder 04_deploy 05_test; do 
    cd $path
    unzip ./aws.zip -d ./aws
    cp *.py ./aws
    cd ./aws
    zip -r ../compiled.zip .
    cd ../
    rm -rf ./aws
    cd ../
    exit 0
  done
fi 

# Compile and run origin if needed
if [[ $COMPILE_ORIGIN_PIPELINE ]]; then
  echo "Compiling origin pipeline"
  python3 workflows/origin.py -n $NAMESPACE
  rm pipeline.tar.gz pipeline.yaml\'\'
fi

if [[ $RUN_ORIGIN_PIPELINE ]]; then
  echo "Running origin pipeline"
  python3 kubeflow_client.py -n $NAMESPACE -f pipeline.yaml
fi

# Compile and run sampling if needed
if [[ $COMPILE_SAMPLING_PIPELINE ]]; then
  echo "Compiling sampling pipeline"
  python3 workflows/sampling.py -n $NAMESPACE
  rm pipeline.tar.gz pipeline.yaml\'\'
fi

if [[ $RUN_SAMPLING_PIPELINE ]]; then
  echo "Running sampling pipeline"
  python3 kubeflow_client.py -n $NAMESPACE -f pipeline.yaml
fi