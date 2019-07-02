import argparse
import os, urllib.parse
from hydrosdk import sdk


def main(
    model_version, model_name, application_name_postfix, 
    hydrosphere_address, is_dev=False, is_aws=False
):
    
    application_name = f"{model_name}{application_name_postfix}"
    model = sdk.Model.from_existing(model_name, model_version)
    application = sdk.Application.singular(application_name, model)
    result = application.apply(hydrosphere_address)
    print(result, flush=True)

    # Dump application metadata:
    # AWS
    if is_aws: return {
        "application_name": application_name,
        "application_link": urllib.parse.urljoin(hydrosphere_address, f"applications/{application_name}")
    }

    # Kubeflow 
    with open("./application_name.txt" if is_dev else "/application_name.txt", "w+") as file:
        file.write(application_name)

    with open("./application_link.txt" if is_dev else "/application_link.txt", "w+") as file:
        file.write(urllib.parse.urljoin(hydrosphere_address, f"applications/{application_name}"))


def aws_lambda(event, context):
    return main(
        model_version=event["model_version"],
        model_name=event["model_name"],
        application_name_postfix=event["application_name_postfix"],
        hydrosphere_address=event["hydrosphere_address"],
        is_aws=True,        
    )


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--application-name-postfix', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--dev', help='Flag for development purposes', action="store_true")
    
    args = parser.parse_args()
    main(
        model_version=args.model_version,
        model_name=args.model_name,
        application_name_postfix=args.application_name_postfix, 
        hydrosphere_address=args.hydrosphere_address,
        is_dev=args.dev,
    )
