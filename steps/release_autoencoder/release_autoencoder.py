import argparse
import os, boto3, urllib.parse
from hydrosdk import sdk


s3 = boto3.resource('s3')


def main(
    data_path, models_path, hydrosphere_address, model_name, 
    loss, learning_rate, steps, classes, batch_size, is_dev=False, is_aws=False
):

    # Download model 
    os.makedirs("model", exist_ok=True)
    for file in s3.Bucket('odsc-workshop').objects.filter(Prefix=models_path):
        relevant_folder = file.key.split("/")[4:]

        # Create nested folders if necessary
        if len(relevant_folder) > 1:
            os.makedirs(os.path.join('model', *relevant_folder[:-1]), exist_ok=True)
        
        s3.Object(file.bucket_name, file.key).download_file(os.path.join('model', *relevant_folder))

    # Build servable
    payload = [
        os.path.join('model', 'saved_model.pb'),
        os.path.join('model', 'variables')
    ]

    metadata = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'data': data_path,
        'model': models_path,
        'loss': loss,
        'steps': steps
    }

    signature = sdk.Signature('predict') \
        .with_input('imgs', 'float32', [-1, 28, 28, 1], 'image') \
        .with_input('probabilities', 'float32', [-1, classes]) \
        .with_input('class_ids', 'int64', [-1, 1]) \
        .with_input('logits', 'float32', [-1, classes]) \
        .with_input('classes', 'string', [-1, 1]) \
        .with_output('score', 'float32', [-1, 1])

    model = sdk.Model() \
        .with_name(model_name) \
        .with_runtime('hydrosphere/serving-runtime-tensorflow-1.13.1:dev') \
        .with_metadata(metadata) \
        .with_payload(payload) \
        .with_signature(signature)

    result = model.apply(hydrosphere_address)
    print(result)

    # Dump built model metadata:
    # AWS
    if is_aws: return {
        "model_version": result["modelVersion"],
        "model_link": urllib.parse.urljoin(
            hydrosphere_address, f"/models/{result['model']['id']}/{result['id']}/details")
    }

    # Kubeflow 
    with open("./model_version.txt" if is_dev else "/model_version.txt", 'w+') as file:
        file.write(str(result['modelVersion']))
    
    with open("./model_link.txt" if is_dev else "/model_link.txt", "w+") as file:
        model_id = str(result["model"]["id"])
        version_id = str(result["id"])
        link = urllib.parse.urljoin(hydrosphere_address, 
            f"models/{model_id}/{version_id}/details")
        file.write(link)


def aws_lambda(event, context):
    return main(
        data_path=event["data_path"],
        models_path=event["models_path"],
        hydrosphere_address=event["hydrosphere_address"],
        model_name=event["model_name"],
        loss=event.get("loss"),
        learning_rate=event["learning_rate"],
        steps=event.get("steps"),
        classes=event["classes"],
        batch_size=event["batch_size"],
        is_aws=True
    )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--models-path', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--loss')
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--steps'),
    parser.add_argument('--classes', type=int, required=True),
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--dev', help='Flag for development purposes', action="store_true")
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        models_path=args.models_path,
        hydrosphere_address=args.hydrosphere_address,
        model_name=args.model_name,
        loss=args.loss,
        learning_rate=args.learning_rate,
        steps=args.steps,
        classes=args.classes,
        batch_size=args.batch_size,
        is_dev=args.dev,
    )



    
