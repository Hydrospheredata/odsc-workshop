import argparse, csv, datetime
import os, urllib.parse
from hydrosdk import sdk

from storage import * 
from orchestrator import *


def main(
    model_version, model_name, application_name_postfix, 
    hydrosphere_address, bucket_name, storage_path="/", **kwargs
):
    
    # Define helper class
    storage = Storage(bucket_name)
    orchestrator = Orchestrator(storage_path=storage_path)

    # Create and deploy endpoint application
    application_name = f"{model_name}{application_name_postfix}"
    model = sdk.Model.from_existing(model_name, model_version)
    
    application = sdk.Application.singular(application_name, model)
    result = application.apply(hydrosphere_address)
    print(result)

    # Export meta to the orchestrator
    application_link = urllib.parse.urljoin(hydrosphere_address, f"applications/{application_name}")
    orchestrator.export_meta("application_name", application_name, "txt")
    orchestrator.export_meta("application_link", application_link, "txt")
    
    if kwargs.get("mlflow_model_link"):

        with open('output.csv', 'w+', newline='') as file:
            fieldnames = ['key', 'value']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({'key': 'mlflow-model-link', 'value': kwargs["mlflow_model_link"]})
            writer.writerow({'key': 'mlflow-drift-detector-link', 'value': kwargs["mlflow_drift_detector_link"]})
            writer.writerow({'key': 'application-link', 'value': application_link})
            writer.writerow({'key': 'data-path', 'value': kwargs["data_path"]})
            writer.writerow({'key': 'model-path', 'value': kwargs["model_path"]})
            writer.writerow({'key': 'model-drift-detector-path', 'value': kwargs["model_drift_detector_path"]})

        namespace = urllib.parse.urlparse(hydrosphere_address).netloc.split(".")[0]
        run_path = os.path.join(namespace, "run", str(round(datetime.datetime.now().timestamp())))
        output_cloud_path = storage.upload_file("output.csv", os.path.join(run_path, "output.csv"))

        metadata = {
            'outputs': [
                {
                    'type': 'table',
                    'storage': storage.prefix,
                    'format': 'csv',
                    'source': output_cloud_path,
                    'header': ['key', 'value'],
                }
            ]
        }
        orchestrator.export_meta("mlpipeline-ui-metadata", metadata, "json")


def aws_lambda(event, context):
    return main(
        model_version=event["model_version"],
        model_name=event["model_name"],
        application_name_postfix=event["application_name_postfix"],
        hydrosphere_address=event["hydrosphere_address"],
        bucket_name=event["bucket_name"],
        storage_path="/tmp/"
    )


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--application-name-postfix', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--bucket-name', required=True)
    parser.add_argument('--mlflow-model-link')
    parser.add_argument('--mlflow-drift-detector-link')
    parser.add_argument('--data-path')
    parser.add_argument('--model-path')
    parser.add_argument('--model-drift-detector-path')
    
    args = parser.parse_args()
    main(
        model_version=args.model_version,
        model_name=args.model_name,
        application_name_postfix=args.application_name_postfix, 
        hydrosphere_address=args.hydrosphere_address,
        bucket_name=args.bucket_name,
        mlflow_model_link=args.mlflow_model_link,
        mlflow_drift_detector_link=args.mlflow_drift_detector_link,
        data_path=args.data_path,
        model_path=args.model_path,
        model_drift_detector_path=args.model_drift_detector_path,
    )
