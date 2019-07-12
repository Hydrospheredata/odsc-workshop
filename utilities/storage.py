import urllib.parse

__all__ = ["Storage"]


class Storage:

    def __init__(self, cloud, bucket_name):
        assert cloud in ("aws", "gcp"), "Only AWS and GCP clouds are supported"

        self.cloud = cloud
        self.bucket_name = bucket_name

        if self.cloud == 'aws': self.prefix = 's3'
        if self.cloud == 'gcp': self.prefix = 'gs'
        self.full_name = f'{self.prefix}://{self.bucket_name}'

        print(f"Initialized {self}")
    
    def __repr__(self):
        return f"Storage(cloud={self.cloud}, bucket_name={self.bucket_name})"

    def upload(self, source_path, destination_path):
        upload_path = None

        if self.cloud == 'aws':
            upload_path = _upload_s3(source_path, destination_path, self.bucket_name)
        if self.cloud == 'gcp':
            upload_path = _upload_gs(source_path, destination_path, self.bucket_name)

        print('File {} has been uploaded to {}'.format(source_path, upload_path), flush=True)
        return upload_path

    def download(self, source_path, destination_path):
        result = urllib.parse.urlparse(source_path)
        if result.scheme: source_path = result.path[1:]

        if self.cloud == 'aws':
            _download_s3(source_path, destination_path, self.bucket_name)
        if self.cloud == 'gcp':
            _download_gs(source_path, destination_path, self.bucket_name)
        
        print('File {} has been downloaded to {}'.format(
            source_path, destination_path), flush=True)
    

def _upload_s3(source_path, destination_path, bucket_name):
    import boto3 
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(Filename=source_path, Bucket=bucket_name, Key=destination_path)
    return f"s3://{bucket_name}/{destination_path}"


def _download_s3(source_path, destination_path, bucket_name):
    import boto3
    s3 = boto3.resource('s3')
    s3.Object(bucket_name, source_path).download_file(destination_path)


def _upload_gs(source_path, destination_path, bucket_name):
    from google.cloud import storage
    storage_client = storage.Client()
    storage_client.get_bucket(bucket_name).blob(destination_path).upload_from_filename(source_path)    
    return f"gs://{bucket_name}/{destination_path}"
    

def _download_gs(source_path, destination_path, bucket_name):
    from google.cloud import storage
    storage_client = storage.Client()
    storage_client.get_bucket(bucket_name).blob(source_path).download_to_filename(destination_path)
