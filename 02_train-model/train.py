import os, json, sys, shutil, tempfile
import tensorflow as tf
import numpy as np
import argparse, boto3
import urllib.parse
from sklearn.metrics import confusion_matrix


def input_fn(imgs, labels, batch_size=256, epochs=10, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={"imgs": imgs.reshape((len(imgs), 28, 28, 1))}, 
        y=labels, shuffle=shuffle, batch_size=batch_size, num_epochs=epochs)


def relative_move(from_dir, to_dir):
    for root, dirs, files in os.walk(from_dir):
        for file in files:
            relpath = os.path.relpath(root, from_dir)
            os.makedirs(os.path.join(to_dir, relpath), exist_ok=True)
            shutil.move(os.path.join(root, file), os.path.join(to_dir, relpath, file)) 


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', help='Path, where the current run\'s data was stored', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument(
        '--dev', help='Flag for development purposes', action="store_true")
    
    args = parser.parse_args()
    s3 = boto3.resource('s3')

    namespace = urllib.parse.urlparse(args.hydrosphere_address).netloc.split(".")[0]
    models_path = os.path.join(namespace, "models", "mnist")

    # Download training/testing data
    s3.Object('odsc-workshop', os.path.join(args.data_path, "train.npz")).download_file('./train.npz')
    s3.Object('odsc-workshop', os.path.join(args.data_path, "test.npz")).download_file('./test.npz')
    
    # Prepare data inputs
    with np.load("./train.npz") as data:
        train_imgs = data["imgs"]
        train_labels = data["labels"].astype(int)
    
    with np.load("./test.npz") as data:
        test_imgs = data["imgs"]
        test_labels = data["labels"].astype(int)

    img_feature_column = tf.feature_column.numeric_column("imgs", shape=(28,28, 1))
    
    train_fn = input_fn(
        train_imgs, train_labels, 
        batch_size=args.batch_size, 
        epochs=args.epochs)
    
    test_fn = input_fn(
        test_imgs, test_labels,
        batch_size=args.batch_size, 
        epochs=args.epochs)
    
    num_classes = len(np.unique(np.hstack([train_labels, test_labels])))

    # Create the model
    estimator = tf.estimator.DNNClassifier(
        model_dir=models_path,
        n_classes=num_classes,
        hidden_units=[256, 64],
        feature_columns=[img_feature_column],
        optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate))

    # Train and evaluate the model
    estimator.train(train_fn)
    evaluation = estimator.evaluate(test_fn)
    accuracy = float(evaluation["accuracy"])

    cm_fn = input_fn(imgs=test_imgs, labels=test_labels, epochs=1, shuffle=False)
    result = list(map(lambda x: x["class_ids"][0], estimator.predict(cm_fn)))
    cm = confusion_matrix(test_labels, result)

    # Export the model 
    serving_input_receiver_fn = tf.estimator \
        .export.build_raw_serving_input_receiver_fn(
            {"imgs": tf.placeholder(tf.float32, shape=(None, 28, 28, 1))})
    model_save_path = estimator.export_savedmodel(models_path, serving_input_receiver_fn)
    model_save_path = model_save_path.decode()

    # Clean up folder structure
    timestamp = os.path.basename(model_save_path)
    final_dir = os.path.join(models_path, timestamp)
    saved_model_dir = os.path.join(final_dir, "saved_model")
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        relative_move(model_save_path, tmpdir1)
        shutil.rmtree(model_save_path)

        relative_move(models_path, tmpdir2)
        shutil.rmtree(models_path)
        
        os.makedirs(final_dir)
        os.makedirs(saved_model_dir)

        relative_move(tmpdir1, saved_model_dir)
        relative_move(tmpdir2, final_dir)

    # Upload model to S3
    for root, dirs, files in os.walk(final_dir):
        for file in files:
            print(f"Uploading {file} to S3", flush=True)

            location = os.path.join(root, file)
            s3.meta.client.upload_file(location, "odsc-workshop", location, ExtraArgs={'ACL':'public-read'})

    np.savetxt("cm.csv", cm, fmt='%d', delimiter=',')
    cm_path = os.path.join(final_dir, "cm.csv")
    s3.meta.client.upload_file("cm.csv", "odsc-workshop", cm_path, ExtraArgs={'ACL':'public-read'})

    # Perform metrics calculations
    if args.dev: 
        accuracy_file = "./accuracy.txt"
        metrics_file = "./mlpipeline-metrics.json"
        metadata_file = "./mlpipeline-ui-metadata.json"
        model_path = "./model_path.txt"
        classes_path = "./classes.txt"
    else: 
        accuracy_file = "/accuracy.txt"
        metrics_file = "/mlpipeline-metrics.json"
        metadata_file = "/mlpipeline-ui-metadata.json"
        model_path = "/model_path.txt"
        classes_path = "/classes.txt"

    
    metrics = {
        'metrics': [
            {
                'name': 'accuracy-score',   # -- The name of the metric. Visualized as the column 
                                            # name in the runs table.
                'numberValue': accuracy,    # -- The value of the metric. Must be a numeric value.
                'format': "PERCENTAGE",     # -- The optional format of the metric. Supported values are 
                                            # "RAW" (displayed in raw format) and "PERCENTAGE" 
                                            # (displayed in percentage format).
            },
        ],
    }

    metadata = {
        'outputs': [
            {
                'type': 'tensorboard',
                'source': os.path.join("s3://odsc-workshop", final_dir),
            },
            {
                'type': 'confusion_matrix',
                'format': 'csv',
                'schema': [
                    {'name': 'one', 'type': 'CATEGORY'},
                    {'name': 'two', 'type': 'CATEGORY'},
                    {'name': 'three', 'type': 'CATEGORY'},
                    {'name': 'four', 'type': 'CATEGORY'},
                    {'name': 'five', 'type': 'CATEGORY'},
                    {'name': 'six', 'type': 'CATEGORY'},
                    {'name': 'seven', 'type': 'CATEGORY'},
                    {'name': 'eight', 'type': 'CATEGORY'},
                    {'name': 'nine', 'type': 'CATEGORY'},
                    {'name': 'ten', 'type': 'CATEGORY'},
                ],
                'source': os.path.join("s3://odsc-workshop", cm_path),
                'labels': [
                    'one', 'two', 'three', 'four', 'five', 
                    'six', 'seven', 'eight', 'nine', 'ten'
                ],
            }
        ]
    }

    with open(accuracy_file, "w+") as file:
        file.write(str(accuracy))
    
    with open(metrics_file, "w+") as file:
        json.dump(metrics, file)

    with open(metadata_file, "w+") as file:
        json.dump(metadata, file)

    with open(model_path, "w+") as file:
        file.write(model_save_path)
    
    with open(classes_path, "w+") as file:
        file.write(str(num_classes))
