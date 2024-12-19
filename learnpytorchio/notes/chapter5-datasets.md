# CH5 - Datasets

## Type of Datasets

- Vision
- Text
- Audio
- Recommendation

## Dataset and DataLoader

- `torch.utils.data.Dataset`: abstract class representing a dataset. It has two methods: `__len__` and `__getitem__`.
- `torch.utils.data.DataLoader`: wraps a dataset and provides a way to iterate over the dataset (batching, shuffling, etc).

Once the dataset is loaded, we can use the DataLoader to iterate over the dataset:

```python	
for X, y in DataLoader(dataset, batch_size=32, shuffle=True):
    # do something with X (data) and y (labels)
    ...
```

## Transforms and Data Augmentation

Reference to `torchvision.transforms` and `torchio.transforms`.