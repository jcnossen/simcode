import torch

class LabelTensor(torch.Tensor):
    def __new__(cls, data, label_dim, labels, *args, **kwargs):
        if not 0 <= label_dim < len(data.shape):
            raise ValueError("label_dim should be a valid dimension index")
        if len(labels) != data.shape[label_dim]:
            raise ValueError("Number of labels should match the size of the labeled dimension")
        tensor = torch.Tensor._make_subclass(cls, data, *args, **kwargs)
        tensor.label_dim = label_dim
        tensor.labels = labels
        return tensor

    def __getitem__(self, key):
        if isinstance(key, str):
            index = self.labels.index(key)
            return self.select(self.label_dim, index)
        elif isinstance(key, (list, tuple)) and all(isinstance(item, str) for item in key):
            indices = [self.labels.index(item) for item in key]
            result_data = self.index_select(self.label_dim, torch.tensor(indices))
            return LabelTensor(result_data, self.label_dim, [self.labels[i] for i in indices])
        else:
            return super().__getitem__(key)

if __name__ == "__main__":
    # Example usage
    data = torch.randn(2, 3, 4)
    label_dim = 1
    labels = ["label0", "label1", "label2"]
    label_tensor = LabelTensor(data, label_dim, labels)

    print(label_tensor["label0"])  # Regular tensor
    print(label_tensor[:, 0])  # LabelTensor
