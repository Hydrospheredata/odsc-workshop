import argparse
import os, urllib.parse
from hydrosdk import sdk

from storage import *
from orchestrator import * 


def main(
    data_path, model_path, hydrosphere_address, model_name, loss, learning_rate, 
    steps, classes, batch_size, cloud, orchestrator_type, bucket_name, storage_path="/"
):

    storage = Storage(cloud, bucket_name)
    orchestrator = Orchestrator(orchestrator_type, storage_path)

    # Download model 
    working_dir = os.path.join(storage_path, "model")
    storage.download_prefix(model_path, working_dir)
    
    # Build servable
    payload = [
        os.path.join(storage_path, 'model', 'saved_model.pb'),
        os.path.join(storage_path, 'model', 'variables')
    ]

    metadata = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'data': data_path,
        'model': model_path,
        'loss': loss,
        'steps': steps,
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

    orchestrator.export_meta("model_version", result["modelVersion"], "txt")
    orchestrator.export_meta("model_link", urllib.parse.urljoin(
        hydrosphere_address, f"/models/{result['model']['id']}/{result['id']}/details"), "txt")


def aws_lambda(event, context):
    return main(
        data_path=event["data_path"],
        model_path=event["model_path"],
        hydrosphere_address=event["hydrosphere_address"],
        model_name=event["model_name"],
        loss=event["loss"],
        learning_rate=event["learning_rate"],
        steps=event["steps"],
        classes=event["classes"],
        batch_size=event["batch_size"],
        cloud="aws",
        orchestrator_type="step_functions",
        storage_path="/tmp/",
        bucket_name=event["bucket_name"],
    )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--steps', required=True),
    parser.add_argument('--loss', required=True)
    parser.add_argument('--classes', type=int, required=True),
    parser.add_argument('--cloud', required=True)
    parser.add_argument('--orchestrator', required=True)
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        model_path=args.model_path,
        hydrosphere_address=args.hydrosphere_address,
        model_name=args.model_name,
        loss=args.loss,
        learning_rate=args.learning_rate,
        steps=args.steps,
        classes=args.classes,
        batch_size=args.batch_size,
        cloud=args.cloud,
        orchestrator_type=args.orchestrator,
        bucket_name=args.bucket_name,
    )



    
