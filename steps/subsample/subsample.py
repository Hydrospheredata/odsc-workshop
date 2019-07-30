import requests, psycopg2
import pickle, os, random, urllib.parse
import numpy as np
import datetime, argparse, hashlib
from hydro_serving_grpc.timemachine.reqstore_client import ReqstoreHttpClient
from cloud import CloudHelper


def get_model_version_id(host_address, application_name):
    addr = urllib.parse.urljoin(host_address, f"api/v2/application/{application_name}")
    resp = requests.get(addr).json()
    assert resp.get("error") is None, resp.get("message")
    return resp["executionGraph"]["stages"][0]["modelVariants"][0]["modelVersion"]["id"]


def main(application_name, bucket_name):

    # Define helper class  
    cloud = CloudHelper().set_bucket(bucket_name)
    config = cloud.get_kube_config_map()

    # Define required variables
    data_path = os.path.join("data", str(round(datetime.datetime.now().timestamp())))
    reqstore_address = urllib.parse.urljoin(config["uri.hydrosphere"], "reqstore")

    client = ReqstoreHttpClient(reqstore_address)
    model_version_id = str(get_model_version_id(config["uri.hydrosphere"], application_name))

    # Initialize connection to Database 
    conn = psycopg2.connect(
        f"postgresql://{config['postgres.user']}:{config['postgres.pass']}@{config['postgres.host']}:{config['postgres.port']}/{config['postgres.dbname']}")
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS 
            requests (hex_uid varchar(256), ground_truth integer);
    ''')
    conn.commit()

    print("Sample data from reqstore", flush=True)
    records = list(client.getRange(0, 1854897851804888100, model_version_id, limit=10000, reverse="false"))
    random.shuffle(records)

    print("Prepare dataset", flush=True)
    imgs, labels = list(), list()

    for timestamp in records:
        for entry in timestamp.entries:
            request_image = np.array(
                entry.request.inputs["imgs"].float_val, dtype=np.float32).reshape((28, 28))
            
            hex_uid = hashlib.sha1(request_image).hexdigest()
            cur.execute("SELECT * FROM requests WHERE hex_uid=%s", (hex_uid,))
            db_record = cur.fetchone()
            if not db_record: continue    
            
            imgs.append(request_image); labels.append(db_record[1])

    if not imgs:
        imgs, labels = np.empty((0, 28, 28)), np.empty((0,))
    else: 
        imgs, labels = np.array(imgs), np.array(labels)
        
    train_imgs, train_labels = imgs[:int(len(imgs) * 0.75)], labels[:int(len(labels) * 0.75)]
    test_imgs, test_labels = imgs[int(len(imgs) * 0.75):], labels[int(len(labels) * 0.75):]

    assert len(train_imgs) > 100, "Not enough training data"
    assert len(test_imgs) > 25, "Not enough testing data"

    print(f"Train subsample size: {len(train_imgs)}", flush=True)
    print(f"Test subsample size: {len(test_imgs)}", flush=True)

    np.savez_compressed("train.npz", imgs=train_imgs, labels=train_labels)
    np.savez_compressed("test.npz", imgs=test_imgs, labels=test_labels)
    cloud.upload_file("train.npz", os.path.join(data_path, "train.npz"))
    cloud.upload_file("test.npz", os.path.join(data_path, "test.npz"))
    
    cloud.export_meta("data_path", data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--application-name', required=True)
    parser.add_argument('--bucket-name', required=True)
    
    args = parser.parse_args()
    main(
        application_name=args.application_name,
        bucket_name=args.bucket_name,
    )
