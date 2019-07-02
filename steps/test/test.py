import os, time, requests, sys
import numpy as np
import argparse, boto3


s3 = boto3.resource('s3')


def main(
    data_path, hydrosphere_address, acceptable_accuracy, 
    application_name, storage_path="./"
):

    # Download testing data
    s3.Object('odsc-workshop', os.path.join(data_path, "test.npz")) \
        .download_file(os.path.join(storage_path, 'test.npz'))
    
    # Prepare data inputs
    with np.load("./test.npz") as data:
        images = data["imgs"][:100]
        labels = data["labels"].astype(int)[:100]
    
    requests_delay = 0.2
    service_link = f"{hydrosphere_address}/gateway/application/{application_name}"

    print(f"Using URL :: {service_link}", flush=True)

    predicted = []
    for index, image in enumerate(images):
        image = image.reshape((1, 28, 28, 1))
        response = requests.post(url=service_link, json={'imgs': [image.tolist()]})
        print(
            f"{index} | {round(index / len(images) * 100)}% \n{response.text}", 
            flush=True
        )
        predicted.append(response.json()["class_ids"][0][0])
        time.sleep(requests_delay)
    
    accuracy = np.sum(labels == np.array(predicted)) / len(labels)
    print(f"Achieved accuracy of {accuracy}", flush=True)

    assert accuracy > acceptable_accuracy, \
        f"Accuracy is not acceptable ({accuracy} < {acceptable_accuracy})"
    

def aws_lambda(event, context):
    return main(
        data_path=event["data_path"],
        hydrosphere_address=event["hydrosphere_address"],
        acceptable_accuracy=event["acceptable_accuracy"],
        application_name=event["application_name"],
        storage_path="/tmp/"
    )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--acceptable-accuracy', type=float, required=True)
    parser.add_argument('--application-name', required=True)
    
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        hydrosphere_address=args.hydrosphere_address,
        acceptable_accuracy=args.acceptable_accuracy,
        application_name=args.application_name
    )

    