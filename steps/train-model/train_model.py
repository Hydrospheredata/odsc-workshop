import os, json, sys, shutil, tempfile
import tensorflow as tf, numpy as np
import urllib.parse, argparse, mlflow, mlflow.tensorflow
from sklearn.metrics import confusion_matrix
from cloud import CloudHelper


def main(mode, data_path, model_path, learning_rate, batch_size, epochs):
    """ Train tf.Estimator and upload it to the cloud. """
    tf.logging.set_verbosity(tf.logging.INFO)

    # Set up environment and variables
    # mlflow_uri = cloud.get_kube_config_map()["uri.mlflow"]

    # Log params into Mlflow
    # mlflow.set_tracking_uri(mlflow_uri)
    # mlflow.set_experiment(f'default.{model_name}')  # Example usage
    # mlflow.log_params({
    #     "data_path": data_path,
    #     "learning_rate": learning_rate,
    #     "batch_size": batch_size,
    #     "epochs": epochs,
    # })

    # Prepare data inputs
    with np.load(os.path.join(data_path, "train", "imgs.npz")) as np_imgs:
        train_imgs = np_imgs["imgs"]
    with np.load(os.path.join(data_path, "train", "labels.npz")) as np_labels:
        train_labels = np_labels["labels"].astype(int)
    with np.load(os.path.join(data_path, "t10k", "imgs.npz")) as np_imgs:
        test_imgs = np_imgs["imgs"]
    with np.load(os.path.join(data_path, "t10k", "labels.npz")) as np_labels:
        test_labels = np_labels["labels"].astype(int)

    train_fn = input_fn(train_imgs, train_labels, batch_size=batch_size, shuffle=True)
    test_fn = input_fn(test_imgs, test_labels, batch_size=batch_size, shuffle=True)
    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28, 1))
    num_classes = len(np.unique(np.hstack([train_labels, test_labels])))

    # Create the model
    # strategy = tf.distribute.MirroredStrategy()
    # config = tf.estimator.RunConfig(train_distribute=strategy)
    estimator = tf.estimator.DNNClassifier(
        model_dir=model_path,
        n_classes=num_classes,
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        # config=config,
    )
    
    # Train and evaluate the model
    evaluation = estimator.train(train_fn).evaluate(test_fn)
    cm = _calculate_confusion_matrix(test_imgs, test_labels, estimator)
    np.savetxt("cm.csv", cm, fmt='%d', delimiter=',')

    # Export the model 
    serving_input_receiver_fn = tf.estimator.export \
        .build_raw_serving_input_receiver_fn({
            "imgs": tf.placeholder(tf.float32, shape=(None, 28, 28, 1))})
    saved_model_path = estimator.export_saved_model(
        model_path, serving_input_receiver_fn).decode()

    # # Model artifacts can also be exported into MLFlow instance
    # mlflow.tensorflow.log_model(
    #     tf_saved_model_dir=saved_model_path,
    #     tf_meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
    #     tf_signature_def_key="predict",
    #     artifact_path=model_path,
    # )
    
    # Prettify folder structure
    # final_dir = _bring_folder_structure_in_correct_order(model_path, saved_model_path)
    # final_dir_with_prefix = os.path.join(cloud.bucket_name_uri, final_dir)
    
    # Upload files to the cloud
    # cloud.upload_prefix(final_dir, final_dir)
    # cm_cloud_path = cloud.upload_file("cm.csv", os.path.join(final_dir, "cm.csv"))

    # Export metadata to the orchestrator
    # metrics = {
    #     'metrics': [{
    #         'name': 'model-accuracy', 
    #         'numberValue': evaluation["accuracy"].item(), 
    #         'format': "PERCENTAGE",    
    #     }],
    # }

    # metadata = {
    #     'outputs': [
    #         {
    #             'type': 'tensorboard',
    #             'source': final_dir_with_prefix,
    #         },
    #         {
    #             'type': 'table',
    #             'storage': cloud.scheme,
    #             'format': 'csv',
    #             'source': cm_cloud_path,
    #             'header': [
    #                 'one', 'two', 'three', 'four', 'five', 
    #                 'six', 'seven', 'eight', 'nine', 'ten'
    #             ],
    #         }
    #     ]
    # }

    # Collect and export metrics and other parameters
    # for key, value in evaluation.items():
    #     mlflow.log_metric(key, value)
    #     cloud.export_meta(key, value) 
    # mlflow.log_param("model_path", os.path.join(cloud.bucket_name_uri, final_dir))
    
    # run = mlflow.active_run()
    # mlflow_run_uri = f"{mlflow_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

    # cloud.export_meta("mlpipeline-metrics", metrics, "json") 
    # cloud.export_meta("mlpipeline-ui-metadata", metadata, "json") 
    # cloud.export_metas({
    #     "model_path": final_dir_with_prefix,
    #     "classes": num_classes,
    #     "mlflow_run_uri": mlflow_run_uri,
    # })


def input_fn(imgs, labels, batch_size=256, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={"imgs": imgs.reshape((len(imgs), 28, 28, 1))}, 
        y=labels,
        batch_size=batch_size, 
        shuffle=shuffle,
    )


def _calculate_confusion_matrix(imgs, labels, model):
    cm_fn = input_fn(imgs=imgs, labels=labels, shuffle=False)
    result = list(map(lambda x: x["class_ids"][0], model.predict(cm_fn)))
    return confusion_matrix(labels, result)


if __name__ == "__main__":
    hyperparameters_path = "/opt/ml/input/config/hyperparameters.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("mode")

    if os.path.exists(hyperparameters_path): 

        with open(hyperparameters_path, "r") as file:
            parameters = json.load(file)
        parameters["data_path"] = "/opt/ml/input/data"
        parameters["model_path"] = "/opt/ml/model"

    else: 
        parser.add_argument('--data-path', help='Path, where the current run\'s data was stored', default=None)
        parser.add_argument('--learning-rate', default=0.01)
        parser.add_argument('--batch-size', default=256)
        parser.add_argument('--epochs', default=10)
        parser.add_argument('--bucket-name', default="s3://workshop-hydrosphere")
        args = parser.parse_args()

        parameters["data_path"] = "./"
        parameters["model_path"] = os.path.join("model", "mnist")
        parameters["learning_rate"] = args.learning_rate
        parameters["batch_size"] = args.batch_size
        parameters["epochs"] = args.epochs

        cloud = CloudHelper().set_bucket(args.bucket_name)
        cloud.download_prefix(args.data_path, "./")

    args = parser.parse_args()
    main(
        mode=args.mode, 
        data_path=parameters["data_path"],
        model_path=parameters["model_path"],
        learning_rate=float(parameters["learning_rate"]),
        batch_size=int(parameters["batch_size"]),
        epochs=int(parameters["epochs"]),
    )