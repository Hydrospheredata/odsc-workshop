import kfp.dsl as dsl 
from kfp.aws import use_aws_secret
from kfp.gcp import use_gcp_secret
import kubernetes.client.models as k8s
import argparse, os

tag = os.environ.get("TAG", "latest")


def parametrise_pipeline(cloud, secret_fn):
    """ Parametrise pipeline definition. """

    def pipeline_definition(
        hydrosphere_address,
        experiment_name, 
        model_learning_rate="0.01",
        model_epochs="10",
        model_batch_size="256",
        drift_detector_learning_rate="0.01",
        drift_detector_steps="3500",
        drift_detector_batch_size="256",
        model_name="mnist",
        model_drift_detector_name="mnist_drift_detector",
        acceptable_accuracy="0.90",
    ):
        """ Pipeline describes structure in which steps should be executed. """

        # 1. Download MNIST dataset
        download = dsl.ContainerOp(
            name="download",
            image=f"hydrosphere/mnist-pipeline-download:{tag}",
            file_outputs={"data_path": "/data_path.txt"},
            arguments=[
                "--cloud", cloud,
                "--hydrosphere-address", hydrosphere_address,
                "--orchestrator", "kubeflow",
            ]
        ).apply(secret_fn())

        # 2. Train MNIST classifier
        train_model = dsl.ContainerOp(
            name="train_model",
            image=f"hydrosphere/mnist-pipeline-train-model:{tag}",
            file_outputs={
                "model_path": "/model_path.txt",
                "classes": "/classes.txt",
                "accuracy": "/accuracy.txt",
                "average_loss": "/average_loss.txt",
                "global_step": "/global_step.txt",
                "loss": "/loss.txt",
                "mlflow_link": "/mlflow_link.txt",
            },
            arguments=[
                "--data-path", download.outputs["data_path"],
                "--learning-rate", model_learning_rate,
                "--batch-size", model_batch_size,
                "--epochs", model_epochs,
                "--hydrosphere-address", hydrosphere_address,
                "--cloud", cloud,
                "--orchestrator", "kubeflow",
                "--experiment", experiment_name,
                "--model-name", model_name, 
            ]
        ).apply(secret_fn())
        train_model.set_memory_request('1G')
        train_model.set_cpu_request('1')

        # 3. Train Drift Detector on MNIST dataset
        train_drift_detector = dsl.ContainerOp(
            name="train_drift_detector",
            image=f"hydrosphere/mnist-pipeline-train-drift-detector:{tag}",
            file_outputs={
                "model_path": "/model_path.txt",
                "classes": "/classes.txt",
                "loss": "/loss.txt",
                "mlflow_link": "/mlflow_link.txt",
            },
            arguments=[
                "--data-path", download.outputs["data_path"],
                "--learning-rate", drift_detector_learning_rate,
                "--batch-size", drift_detector_batch_size,
                "--steps", drift_detector_steps,
                "--hydrosphere-address", hydrosphere_address,
                "--cloud", cloud,
                "--orchestrator", "kubeflow",
                "--experiment", experiment_name,
                "--model-name", model_drift_detector_name,
            ]
        ).apply(secret_fn())
        train_drift_detector.set_memory_request('2G')
        train_drift_detector.set_cpu_request('1')

        # # 4. Release Drift Detector to Hydrosphere.io platform 
        # release_drift_detector = dsl.ContainerOp(
        #     name="release_drift_detector",
        #     image=f"hydrosphere/mnist-pipeline-release-drift-detector:{tag}", 
        #     file_outputs={
        #         "model_version": "/model_version.txt",
        #         "model_link": "/model_link.txt"
        #     },
        #     arguments=[
        #         "--data-path", download.outputs["data_path"],
        #         "--model-name", model_drift_detector_name,
        #         "--models-path", train_drift_detector.outputs["model_path"],
        #         "--classes", train_drift_detector.outputs["classes"],
        #         "--loss", train_drift_detector.outputs["loss"],
        #         "--hydrosphere-address", hydrosphere_address,
        #         "--learning-rate", drift_detector_learning_rate,
        #         "--batch-size", drift_detector_batch_size,
        #         "--steps", drift_detector_steps, 
        #         "--cloud", cloud,
        #         "--orchestrator", "kubeflow",
        #     ]
        # ).apply(secret_fn())

        # # 5. Deploy Drift Detector model as endpoint application 
        # deploy_drift_detector_to_prod = dsl.ContainerOp(
        #     name="deploy_drift_detector_to_prod",
        #     image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        #     file_outputs={
        #         "application_name": "/application_name.txt",
        #         "application_link": "/application_link.txt"
        #     },
        #     arguments=[
        #         "--model-version", release_drift_detector.outputs["model_version"],
        #         "--application-name-postfix", "_app", 
        #         "--hydrosphere-address", hydrosphere_address,
        #         "--model-name", model_drift_detector_name,
        #         "--cloud", cloud,
        #         "--orchestrator", "kubeflow",
        #     ],
        # ).apply(secret_fn())

        # # 6. Release MNIST classifier with assigned metrics to Hydrosphere.io platform
        # release_model = dsl.ContainerOp(
        #     name="release_model",
        #     image=f"hydrosphere/mnist-pipeline-release-model:{tag}", 
        #     file_outputs={
        #         "model_version": "/model_version.txt",
        #         "model_link": "/model_link.txt"
        #     },
        #     arguments=[
        #         "--data-path", download.outputs["data_path"],
        #         "--model-name", model_name,
        #         "--models-path", train_model.outputs["model_path"],
        #         "--drift-detector-app", deploy_drift_detector_to_prod.outputs["application_name"],
        #         "--classes", train_model.outputs["classes"],
        #         "--accuracy", train_model.outputs["accuracy"],
        #         "--hydrosphere-address", hydrosphere_address,
        #         "--learning-rate", model_learning_rate,
        #         "--epochs", model_epochs,
        #         "--batch-size", model_batch_size,
        #         "--cloud", cloud,
        #         "--orchestrator", "kubeflow",
        #     ]
        # ).apply(secret_fn())

        # # 7. Deploy MNIST classifier model as endpoint application on stage for testing purposes
        # deploy_model_to_stage = dsl.ContainerOp(
        #     name="deploy_model_to_stage",
        #     image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        #     file_outputs={
        #         "application_name": "/application_name.txt",
        #         "application_link": "/application_link.txt"
        #     },
        #     arguments=[
        #         "--model-version", release_model.outputs["model_version"],
        #         "--application-name-postfix", "_stage_app", 
        #         "--hydrosphere-address", hydrosphere_address,
        #         "--model-name", model_name,
        #         "--cloud", cloud,
        #         "--orchestrator", "kubeflow",
        #     ],
        # ).apply(secret_fn())

        # # 8. Perform integration testing on the deployed staged application
        # test_model = dsl.ContainerOp(
        #     name="test_model",
        #     image=f"hydrosphere/mnist-pipeline-test:{tag}", 
        #     arguments=[
        #         "--data-path", download.outputs["data_path"],
        #         "--hydrosphere-address", hydrosphere_address,
        #         "--application-name", deploy_model_to_stage.outputs["application_name"], 
        #         "--acceptable-accuracy", acceptable_accuracy,
        #         "--cloud", cloud,
        #         "--orchestrator", "kubeflow",
        #     ],
        # ).apply(secret_fn())
        # test_model.set_retry(3)

        # # 9. Deploy MNIST classifier model as endpoint application to production
        # deploy_model_to_prod = dsl.ContainerOp(
        #     name="deploy_model_to_prod",
        #     image=f"hydrosphere/mnist-pipeline-deploy:{tag}",  
        #     file_outputs={
        #         "application_name": "/application_name.txt",
        #         "application_link": "/application_link.txt"
        #     },
        #     arguments=[
        #         "--model-version", release_model.outputs["model_version"],
        #         "--application-name-postfix", "_app", 
        #         "--hydrosphere-address", hydrosphere_address,
        #         "--model-name", model_name,
        #         "--cloud", cloud,
        #         "--orchestrator", "kubeflow",
        #     ],
        # ).apply(secret_fn())
        # deploy_model_to_prod.after(test_model)

    return pipeline_definition


def cloud_specific_pipeline_definition(is_aws=False, is_gcp=False):
    if is_aws:
        return dsl.pipeline(name="MNIST", description="MNIST Workflow Example") \
            (parametrise_pipeline(cloud="aws", secret_fn=use_aws_secret))
    if is_gcp:
        return dsl.pipeline(name="MNIST", description="MNIST Workflow Example") \
            (parametrise_pipeline(cloud="gcp", secret_fn=use_gcp_secret))

    raise NotImplementedError("Only AWS and GCP are supported at the moment")


if __name__ == "__main__":
    import kfp.compiler as compiler
    import subprocess, sys, argparse 

    # Acquire parameters	
    parser = argparse.ArgumentParser()	
    parser.add_argument('--aws', action="store_true")
    parser.add_argument('--gcp', action="store_true")
    parser.add_argument(	
        '-n', '--namespace', help="Namespace, where kubeflow and serving are running")	
    args = parser.parse_args()

    # Compile pipeline
    assert args.aws or args.gcp, "Either --aws or --gcp should be provided"
    if args.aws: 
        compiler.Compiler().compile(cloud_specific_pipeline_definition(is_aws=True), "pipeline.tar.gz")
    if args.gcp:
        compiler.Compiler().compile(cloud_specific_pipeline_definition(is_gcp=True), "pipeline.tar.gz")
    
    process = subprocess.run("tar -xvf pipeline.tar.gz".split())	

    # Replace hardcoded namespaces	
    if args.namespace: 
        process = subprocess.run(f"sed -i \"s/minio-service.kubeflow/minio-service.{args.namespace}/g\" pipeline.yaml".split())	
