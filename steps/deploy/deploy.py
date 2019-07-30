import argparse, csv, datetime
import os, urllib.parse
from hydrosdk import sdk
from cloud import CloudHelper


def main(model_version, model_name, application_name_postfix, bucket_name, **kwargs):
    
    # Define helper class
    cloud = CloudHelper().set_bucket(bucket_name)
    config = cloud.get_kube_config_map()
    
    # Create and deploy endpoint application
    application_name = f"{model_name}{application_name_postfix}"
    model = sdk.Model.from_existing(model_name, model_version)
    
    application = sdk.Application.singular(application_name, model)
    application.apply(config["uri.hydrosphere"])

    # Export meta to the orchestrator
    application_uri = urllib.parse.urljoin(config["uri.hydrosphere"], f"applications/{application_name}")
    cloud.export_metas({
        "application_name": application_name,
        "application_uri": application_uri
    })
    
    # In the last workflow stage return all outputs as an artifact for Kubeflow
    if kwargs.get("mlflow_model_run_uri"):
        with open('output.csv', 'w+', newline='') as file:
            fieldnames = ['key', 'value']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({'key': 'mlflow-model-run-uri', 'value': kwargs["mlflow_model_run_uri"]})
            writer.writerow({'key': 'mlflow-drift-detector-run-uri', 'value': kwargs["mlflow_drift_detector_run_uri"]})
            writer.writerow({'key': 'application-uri', 'value': application_uri})
            writer.writerow({'key': 'data-path', 'value': kwargs["data_path"]})
            writer.writerow({'key': 'model-path', 'value': kwargs["model_path"]})
            writer.writerow({'key': 'model-drift-detector-path', 'value': kwargs["model_drift_detector_path"]})

        run_path = os.path.join("run", str(round(datetime.datetime.now().timestamp())))
        output_cloud_path = cloud.upload_file('output.csv', os.path.join(run_path, 'output.csv'))
        cloud.export_meta(
            key="mlpipeline-ui-metadata",
            value={
                'outputs': [{
                    'type': 'table',
                    'storage': cloud.scheme,
                    'format': 'csv',
                    'source': output_cloud_path,
                    'header': ['key', 'value']
                }]
            }, 
            extension="json"
        )


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--application-name-postfix', required=True)
    parser.add_argument('--bucket-name', required=True)
    parser.add_argument('--mlflow-model-run-uri')
    parser.add_argument('--mlflow-drift-detector-run-uri')
    parser.add_argument('--data-path')
    parser.add_argument('--model-path')
    parser.add_argument('--model-drift-detector-path')
    
    args = parser.parse_args()
    main(
        model_version=args.model_version,
        model_name=args.model_name,
        application_name_postfix=args.application_name_postfix, 
        bucket_name=args.bucket_name,
        mlflow_model_run_uri=args.mlflow_model_run_uri,
        mlflow_drift_detector_run_uri=args.mlflow_drift_detector_run_uri,
        data_path=args.data_path,
        model_path=args.model_path,
        model_drift_detector_path=args.model_drift_detector_path,
    )
