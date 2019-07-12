import os
import json
from collections.abc import Iterable

__all__ = ["Orchestrator"]


class Orchestrator:

    def __init__(self, orchestrator_type, storage_path="./"):
        self.type = orchestrator_type
        self.storage_path = storage_path
        assert orchestrator_type in ("step_functions", "kubeflow")
    
    def _serialize_value(self, value, extension):
        if extension == "json":
            return json.dumps(value)
        return str(value)

    def _write_file(self, filename, content):
        with open(filename, "w+") as file: 
            file.write(content)

    def export(self, parameters):
        if self.type is None:
            return None 
        
        if self.type == "step_functions":
            return parameters

        if self.type == "kubeflow":
            for key, value in parameters.items():
                extension = "txt"
                if not isinstance(value, str) and isinstance(value, Iterable):
                    extension = value[0]; value = value[1]
                value = json.dumps(value) if isinstance(value, dict) else str(value)

                path = os.path.join(self.storage_path, f'{key}.{extension}')
                with open(path, "w+") as file:
                    file.write(value)
                
                print(f"Written new file {path}", flush=True)
    
    def export_meta(self, key, value, extension=None):
        if extension is not None:
            key = f"{key}.{extension}"
            value = self._serialize_value(value, extension)

        self._write_file(
            filename=os.path.join(self.storage_path, key), 
            content=value)