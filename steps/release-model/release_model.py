import argparse, os, urllib.parse
from hydrosdk import sdk
from cloud import CloudHelper


def main(drift_detector_app, model_name, classes, bucket_name, **kwargs):
    
    cloud = CloudHelper().set_bucket(bucket_name)
    config = cloud.get_kube_config_map()

    # Download model 
    cloud.download_prefix(os.path.join(kwargs["model_path"], "saved_model"), "./")

    # Build servable
    metadata = {
        'learning_rate': kwargs["learning_rate"],
        'batch_size': kwargs["batch_size"],
        'epochs': kwargs["epochs"], 
        'accuracy': kwargs["accuracy"],
        'average_loss': kwargs["average_loss"],
        'loss': kwargs["loss"],
        'global_step': kwargs["global_step"],
        'mlflow_run_uri': kwargs["mlflow_run_uri"], 
        'data': kwargs["data_path"],
        'model_path': kwargs["model_path"],
    }

    monitoring = [
        sdk.Monitoring('Requests').with_spec('CounterMetricSpec', interval=15),
        sdk.Monitoring('Latency').with_spec('LatencyMetricSpec', interval=15),
        sdk.Monitoring('Accuracy').with_spec('AccuracyMetricSpec'),
        sdk.Monitoring('Drift Detector') \
            .with_health(True) \
            .with_spec(
                kind='ImageAEMetricSpec', 
                threshold=0.15, 
                application=drift_detector_app
            )
    ]

    model = sdk.Model() \
        .with_payload(os.listdir()) \
        .with_runtime(config["default.tensorflow_runtime"]) \
        .with_metadata(metadata) \
        .with_monitoring(monitoring) \
        .with_name(model_name)

    result = model.apply(config["uri.hydrosphere"])
    cloud.export_metas({
        "model_version": result["modelVersion"],
        "model_uri": urllib.parse.urljoin(
            config["uri.hydrosphere"], f"/models/{result['model']['id']}/{result['id']}/details")
    })


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--drift-detector-app', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--classes', type=int, required=True)
    parser.add_argument('--bucket-name', required=True)

    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--accuracy', required=True)
    parser.add_argument('--average-loss', required=True)
    parser.add_argument('--loss', required=True)
    parser.add_argument('--learning-rate', required=True)
    parser.add_argument('--batch-size', required=True)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--global-step', required=True)
    parser.add_argument('--mlflow-run-uri', required=True)
    
    args = parser.parse_args()
    main(
        drift_detector_app=args.drift_detector_app,
        model_name=args.model_name,
        classes=args.classes,
        bucket_name=args.bucket_name,
        data_path=args.data_path,
        model_path=args.model_path,
        accuracy=args.accuracy,
        average_loss=args.average_loss,
        loss=args.loss,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        global_step=args.global_step,
        mlflow_run_uri=args.mlflow_run_uri,
    )
