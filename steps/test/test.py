import os, time, requests, sys
import numpy as np
import argparse
from cloud import CloudHelper


def main(data_path, acceptable_accuracy, application_name, bucket_name):
    cloud = CloudHelper().set_bucket(bucket_name)
    config = cloud.get_kube_config_map()

    # Download testing data
    cloud.download_file(os.path.join(data_path, 'test.npz'), './')
    
    # Prepare data inputs
    with np.load("test.npz") as data:
        images = data["imgs"][:100]
        labels = data["labels"].astype(int)[:100]
    
    # Define variables 
    requests_delay = 0.2
    service_link = f"{config['uri.hydrosphere']}/gateway/application/{application_name}"
    print(f"Using URL :: {service_link}", flush=True)

    # Collect responses
    predicted = []
    for index, image in enumerate(images):
        response = requests.post(
            url=service_link, json={'imgs': [image.reshape((1, 28, 28, 1)).tolist()]})
        print(f"{index} | {round(index / len(images) * 100)}% \n{response.text}", flush=True)
        
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(requests_delay)
    
    accuracy = np.sum(labels == np.array(predicted)) / len(labels)
    cloud.export_meta('integration-test-accuracy', accuracy)

    print(f"Achieved accuracy of {accuracy}", flush=True)
    assert accuracy > acceptable_accuracy, \
        f"Accuracy is not acceptable ({accuracy} < {acceptable_accuracy})"
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--acceptable-accuracy', type=float, required=True)
    parser.add_argument('--application-name', required=True)
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        acceptable_accuracy=args.acceptable_accuracy,
        application_name=args.application_name,
        bucket_name=args.bucket_name,
    )

    