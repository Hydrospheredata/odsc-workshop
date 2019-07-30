import os, gzip, tarfile, shutil, glob, struct
import urllib, urllib.parse, urllib.request
import datetime, argparse, numpy
from PIL import Image
from cloud import CloudHelper


filenames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]


def download_files(base_url, filenames=None):
    """ Download required data """

    if not filenames: 
        # if not any filenames provided, use global instead
        filenames = globals()["filenames"]
    
    for file in filenames:
        print(f"Started downloading {file}", flush=True)
        download_url = urllib.parse.urljoin(base_url, file)
        local_file, _ = urllib.request.urlretrieve(download_url, file)
        unpack_archive(local_file)


def unpack_archive(file):
    """ Unpack compressed file """

    print(f"Unpacking archive {file}", flush=True)
    with gzip.open(file, 'rb') as f_in, open(file[:-3],'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file)


def process_images(dataset):
    """ Preprocess downloaded MNIST datasets """
    
    print(f"Processing images {dataset}", flush=True)
    label_file = dataset + '-labels-idx1-ubyte'
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = numpy.fromfile(file, dtype=numpy.int8) #int8
        new_labels = numpy.zeros((num, 10))
        new_labels[numpy.arange(num), labels] = 1

    img_file = dataset + '-images-idx3-ubyte'
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = numpy.fromfile(file, dtype=numpy.uint8).reshape(num, rows, cols)
        imgs = imgs.astype(numpy.float32) / 255.0

    os.remove(label_file); os.remove(img_file)
    return imgs, labels


def process_and_upload(data_set: str, upload_path: str, cloud: CloudHelper):
    imgs, labels = process_images(data_set)
    os.makedirs(data_set, exist_ok=True)
    numpy.savez_compressed(os.path.join(data_set, "imgs.npz"), imgs=imgs)
    numpy.savez_compressed(os.path.join(data_set, "labels.npz"), labels=labels)
    cloud.upload_prefix(data_set, os.path.join(upload_path, data_set))


def main(bucket_name):
    """ Download MNIST data, process it and upload it to the cloud. """

    cloud = CloudHelper(default_params={"uri.mnist": "http://yann.lecun.com/exdb/mnist/"}) \
        .set_bucket(bucket_name)
    mnist_uri = cloud.get_kube_config_map()["uri.mnist"]
    data_path = os.path.join("data", str(round(datetime.datetime.now().timestamp())))

    download_files(mnist_uri)
    process_and_upload("train", data_path, cloud)
    process_and_upload("t10k", data_path, cloud)

    cloud.export_meta("data_path", os.path.join(cloud.bucket_name_uri, data_path))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', required=True)

    args = parser.parse_args()
    main(bucket_name=args.bucket_name)
