import argparse
import os, urllib.parse
from hydrosdk import sdk
from cloud import CloudHelper


def main(
    data_path, model_path, model_name, loss, learning_rate, 
    steps, classes, batch_size, bucket_name,
):

    cloud = CloudHelper().set_bucket(bucket_name)
    config = cloud.get_kube_config_map()

    # Download model 
    cloud.download_prefix(model_path, "./") 

    metadata = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'data': data_path,
        'model': model_path,
        'loss': loss,
        'steps': steps,
    }

    model = sdk.Model() \
        .with_name(model_name) \
        .with_runtime(config["default.tensorflow_runtime"]) \
        .with_payload(['saved_model.pb', 'variables']) \
        .with_metadata(metadata)

    result = model.apply(config["uri.hydrosphere"])
    cloud.export_metas({
        "model_version": result["modelVersion"],
        "model_uri": urllib.parse.urljoin(
            config["uri.hydrosphere"], f"/models/{result['model']['id']}/{result['id']}/details")
    })


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--steps', required=True),
    parser.add_argument('--loss', required=True)
    parser.add_argument('--classes', type=int, required=True),
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        model_path=args.model_path,
        model_name=args.model_name,
        loss=args.loss,
        learning_rate=args.learning_rate,
        steps=args.steps,
        classes=args.classes,
        batch_size=args.batch_size,
        bucket_name=args.bucket_name,
    )



    
