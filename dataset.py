import json
from dataclasses import dataclass
from nnsight.envoy import Envoy

class Dataset:
    def __init__(self, location="data/labeled_sentences.jsonl"):
        self.location = location
        self.examples = []
        self.labels = {}
        self.labels_binary = {}
        self.load_data()

    def load_data(self):
        # Load examples and labels
        with open(self.location, 'r') as lines:
            for line in lines:
                data = json.loads(line)
                self.examples.append(data["sentence"])
                for key in data.keys():
                    if key == "sentence":
                        continue
                    value = data[key]
                    if key not in self.labels:
                        self.labels[key] = []
                    self.labels[key].append(value)
        
        # Construct binarized version of labels for all key/value pairs
        for key in self.labels:
            values = set(self.labels[key])
            for value in values:
                kv = f"{key}-{value}"
                if kv not in self.labels_binary:
                    self.labels_binary[kv] = []
                is_kv = [v == value for v in self.labels[key]]
                self.labels_binary[kv] = is_kv

@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: Envoy
    use_input: bool = False
    is_tuple: bool = False

    def __hash__(self):
        return hash(self.name)

    def get_activation(self):
        if self.use_input:
            out = self.submodule.input # TODO make sure I didn't break for pythia
        else:
            out = self.submodule.output
        if self.is_tuple:
            return out[0]
        else:
            return out

    def set_activation(self, x):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0][:] = x
            else:
                self.submodule.input[:] = x
        else:
            if self.is_tuple:
                self.submodule.output[0][:] = x
            else:
                self.submodule.output[:] = x

    def stop_grad(self):
        if self.use_input:
            if self.is_tuple:
                self.submodule.input[0].grad = t.zeros_like(self.submodule.input[0])
            else:
                self.submodule.input.grad = t.zeros_like(self.submodule.input)
        else:
            if self.is_tuple:
                self.submodule.output[0].grad = t.zeros_like(self.submodule.output[0])
            else:
                self.submodule.output.grad = t.zeros_like(self.submodule.output)
