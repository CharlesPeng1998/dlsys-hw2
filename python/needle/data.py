import struct
import gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        h, w = img.shape[0], img.shape[1]
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding + 1, size=2)
        padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        x = self.padding + shift_x
        y = self.padding + shift_y

        return padded_img[x:x + h, y:y + w, :]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
            self,
            dataset: Dataset,
            batch_size: Optional[int] = 1,
            shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, range(self.batch_size, len(self.dataset), self.batch_size))
        return self

    def __next__(self):
        if self.current_idx >= len(self.ordering):
            raise StopIteration
        else:
            indices = self.ordering[self.current_idx]
            self.current_idx += 1
            return [Tensor(x, device=None, requires_grad=False) for x in self.dataset[indices]]


def parse_mnist(image_filename, label_filename):
    with gzip.open(image_filename, 'rb') as image_file:
        image_file_content = image_file.read()
        num_images, = struct.unpack_from(">i", image_file_content, 4)
        x = np.frombuffer(image_file_content, dtype=np.uint8, offset=16)
        x_norm = x.reshape(num_images, 28, 28, 1).astype(np.float32) / 255

    with gzip.open(label_filename, 'rb') as label_file:
        label_file_content = label_file.read()
        y = np.frombuffer(label_file_content, dtype=np.uint8, offset=8)

    return x_norm, y


class MNISTDataset(Dataset):
    def __init__(self, image_filename: str, label_filename: str, transforms: Optional[List] = None):
        super().__init__(transforms)
        self.x, self.y = parse_mnist(image_filename, label_filename)

    def __getitem__(self, index) -> object:
        return self.apply_transforms(self.x[index]), self.y[index]

    def __len__(self) -> int:
        return len(self.x)


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
