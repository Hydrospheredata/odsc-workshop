import argparse
import os, urllib.parse
from hydrosdk import sdk

from orchestrator import *


def main(
    model_version, model_name, application_name_postfix, 
    hydrosphere_address, cloud, orchestrator_type, storage_path="/"
):
    
    # Define helper class
    orchestrator = Orchestrator(orchestrator_type, storage_path=storage_path)

    # Create and deploy endpoint application
    application_name = f"{model_name}{application_name_postfix}"
    model = sdk.Model.from_existing(model_name, model_version)
    
    application = sdk.Application.singular(application_name, model)
    result = application.apply(hydrosphere_address)
    print(result)

    # Export meta to the orchestrator
    orchestrator.export_meta("application_name", application_name, "txt")
    orchestrator.export_meta("application_link", urllib.parse.urljoin(hydrosphere_address, f"applications/{application_name}"), "txt")


def aws_lambda(event, context):
    return main(
        model_version=event["model_version"],
        model_name=event["model_name"],
        application_name_postfix=event["application_name_postfix"],
        hydrosphere_address=event["hydrosphere_address"],
        orchestrator_type="step_functions",
        cloud="aws",
        storage_path="/tmp/"
    )


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--application-name-postfix', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--cloud', required=True)
    parser.add_argument('--orchestrator', required=True)
    
    args = parser.parse_args()
    main(
        model_version=args.model_version,
        model_name=args.model_name,
        application_name_postfix=args.application_name_postfix, 
        hydrosphere_address=args.hydrosphere_address,
        cloud=args.cloud,
        orchestrator_type=args.orchestrator,
    )
