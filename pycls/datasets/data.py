# This file is modified from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

# Import necessary modules
import os
import random
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Lambda

from .randaugment import RandAugmentPolicy
from .simclr_augment import get_simclr_ops
import pycls.utils.logging as lu
from pycls.datasets.custom_datasets import CIFAR10, CIFAR100, MNIST, SVHN, CoolRoofDataset
from pycls.datasets.imbalanced_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from pycls.datasets.sampler import IndexedSequentialSampler
from pycls.datasets.tiny_imagenet import TinyImageNet
from pycls.datasets.preprocess_cool_roof import load_and_preprocess

logger = lu.get_logger(__name__)

class Data:
    """
    Contains all data related functions. For working with new dataset, 
    make changes to following functions:
    1. getDataset
    2. getPreprocessOps
    3. getDataLoaders
    """
    def __init__(self, cfg):
        """
        Initializes dataset attributes.
        INPUT:
        cfg: yacs.config, config object
        """
        self.dataset = cfg.DATASET.NAME
        self.data_dir = cfg.DATASET.ROOT_DIR
        self.datasets_accepted = [
            "MNIST", 
            "SVHN", 
            "CIFAR10", 
            "CIFAR100", 
            "TINYIMAGENET", 
            "IMBALANCED_CIFAR10", 
            "IMBALANCED_CIFAR100",
            "cool_roof"  # Add cool_roof here
        ]
        self.eval_mode = False
        self.aug_method = cfg.DATASET.AUG_METHOD
        self.rand_augment_N = 1 if cfg is None else cfg.RANDAUG.N
        self.rand_augment_M = 5 if cfg is None else cfg.RANDAUG.M

    def getPreprocessOps(self):
        """
        Specifies preprocessing steps for the dataset.
        """
        ops = []

        if self.dataset in ["CIFAR10", "CIFAR100", "IMBALANCED_CIFAR10", "IMBALANCED_CIFAR100"]:
            ops = [transforms.RandomCrop(32, padding=4)]
            norm_mean, norm_std = [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
            ops.append(transforms.Normalize(norm_mean, norm_std))

        elif self.dataset == "MNIST":
            ops = [transforms.Resize(32)]
            norm_mean, norm_std = [0.1307], [0.3081]
            ops.append(transforms.Normalize(norm_mean, norm_std))

        elif self.dataset == "TINYIMAGENET":
            ops = [transforms.RandomResizedCrop(64)]
            norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ops.append(transforms.Normalize(norm_mean, norm_std))

        elif self.dataset == "SVHN":
            ops = [transforms.RandomCrop(32, padding=4)]
            norm_mean, norm_std = [0.4376, 0.4437, 0.4728], [0.1980, 0.2010, 0.1970]
            ops.append(transforms.Normalize(norm_mean, norm_std))

        elif self.dataset == "cool_roof":
            # Add normalization suitable for numerical data
            ops = [Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-5))]  # Example normalization for numerical tensors

        else:
            raise NotImplementedError(f"{self.dataset} is not supported in preprocessing operations.")

        return ops



    def getDataset(self, save_dir, isTrain=True, isDownload=False):
        """
        Returns the dataset instance.
        INPUT:
        save_dir: Directory to save or load the dataset.
        isTrain: Bool, if True loads the training set; else loads the validation/test set.
        OUTPUT:
        Returns the dataset instance and its length.
        """
        self.eval_mode = not isTrain
        preprocess_steps = transforms.Compose(self.getPreprocessOps())

        if self.dataset == "MNIST":
            mnist = MNIST(save_dir, train=isTrain, transform=preprocess_steps, download=isDownload)
            return mnist, len(mnist)
        elif self.dataset == "CIFAR10":
            cifar10 = CIFAR10(save_dir, train=isTrain, transform=preprocess_steps, download=isDownload)
            return cifar10, len(cifar10)
        elif self.dataset == "CIFAR100":
            cifar100 = CIFAR100(save_dir, train=isTrain, transform=preprocess_steps, download=isDownload)
            return cifar100, len(cifar100)
        elif self.dataset == "SVHN":
            svhn = SVHN(save_dir, split='train' if isTrain else 'test', transform=preprocess_steps, download=isDownload)
            return svhn, len(svhn)
        elif self.dataset == "TINYIMAGENET":
            tiny = TinyImageNet(save_dir, split='train' if isTrain else 'val', transform=preprocess_steps)
            return tiny, len(tiny)
        elif self.dataset == "IMBALANCED_CIFAR10":
            im_cifar10 = IMBALANCECIFAR10(save_dir, train=isTrain, transform=preprocess_steps)
            return im_cifar10, len(im_cifar10)
        elif self.dataset == "IMBALANCED_CIFAR100":
            im_cifar100 = IMBALANCECIFAR100(save_dir, train=isTrain, transform=preprocess_steps)
            return im_cifar100, len(im_cifar100)
        
        elif self.dataset == "cool_roof":
            if isTrain:
                features, labels, _, _ = load_and_preprocess(self.data_dir)
            else:
                _, _, features, labels = load_and_preprocess(self.data_dir)
            
            # Apply only numerical transformations
            dataset = CoolRoofDataset(features, labels, transform=preprocess_steps)
            return dataset, len(dataset)


        else:
            raise NotImplementedError(f"{self.dataset} is not implemented in getDataset.")



    def makeLUVSets(self, train_split_ratio, val_split_ratio, data, seed_id, save_dir):
        """
        Initialize the labelled and unlabelled set by splitting the data into train
        and validation according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        |<--- Labelled --><---Unlabelled --><--- Validation --->

        INPUT:
        train_split_ratio: Float, Specifies the proportion of data in train set.
        For example: 0.8 means beginning 80% of data is training data.

        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the labelled, unlabelled set along with validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(train_split_ratio, float),"Train split ratio is of {} datatype instead of float".format(type(train_split_ratio))
        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))
        assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)

        lSet = []
        uSet = []
        valSet = []
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)
        train_splitIdx = int(train_split_ratio*n_dataPoints)
        #To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        #Check there should be no overlap with train and val data
        assert train_split_ratio + val_split_ratio < 1.0, "Validation data over laps with train data as last train index is {} and last val index is {}. \
            The program expects val index > train index. Please satisfy the constraint: train_split_ratio + val_split_ratio < 1.0; currently it is {} + {} is not < 1.0 => {} is not < 1.0"\
                .format(train_splitIdx, val_splitIdx, train_split_ratio, val_split_ratio, train_split_ratio + val_split_ratio)
        
        lSet = all_idx[:train_splitIdx]
        uSet = all_idx[train_splitIdx:val_splitIdx]
        valSet = all_idx[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def makeTVSets(self, val_split_ratio, data, seed_id, save_dir):
        """
        Initialize the train and validation sets by splitting the train data according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        INPUT:
        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the train set and the validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))
        assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)

        trainSet = []
        valSet = []
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)

        # To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        
        trainSet = all_idx[:val_splitIdx]
        valSet = all_idx[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        trainSet = np.array(trainSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/trainSet.npy', trainSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/trainSet.npy', f'{save_dir}/valSet.npy'

    def makeUVSets(self, val_split_ratio, data, seed_id, save_dir): 
        """
        Initial labeled pool should already be sampled. We use this function to initialize the train and validation sets by splitting the train data according to split_ratios arguments.

        Visually it does the following:

        |<------------- Unlabeled -------------><--- Validation --->

        INPUT:
        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to uSet instance post initial pool sampling. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the unlabeled set and the validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))
        assert self.dataset in self.datasets_accepted, "Sorry the dataset {} is not supported. Currently we support {}".format(self.dataset, self.datasets_accepted)
        uSet = []
        valSet = []
        
        n_dataPoints = len(data)
        # all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(data)

        # To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        
        uSet = data[:val_splitIdx]
        valSet = data[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def getIndexesDataLoader(self, indexes, batch_size, data):
        """
        Gets reference to the data loader which provides batches of <batch_size> by randomly sampling
        from indexes set. We use SubsetRandomSampler as sampler in returned DataLoader.

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(indexes, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(batch_size, int), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(type(batch_size))
        
        subsetSampler = SubsetRandomSampler(indexes)
        # # print(data)
        # if self.dataset == "IMAGENET":
        #     loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler, pin_memory=True)
        # else:
        loader = DataLoader(dataset=data, batch_size=batch_size, sampler=subsetSampler)
        return loader


    def getSequentialDataLoader(self, indexes, batch_size, data):
        """
        Gets reference to the data loader which provides batches of <batch_size> sequentially 
        from indexes set. We use IndexedSequentialSampler as sampler in returned DataLoader.

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(indexes, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(batch_size, int), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(type(batch_size))
        
        subsetSampler = IndexedSequentialSampler(indexes)
        # if self.dataset == "IMAGENET":
        #     loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler,pin_memory=True)
        # else:

        loader = DataLoader(dataset=data, batch_size=batch_size, sampler=subsetSampler, shuffle=False)
        return loader


    def getTestLoader(self, data, test_batch_size, seed_id=0):
        """
        Implements a random subset sampler for sampling the data from test set.
        
        INPUT:
        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        test_batch_size: int, Denotes the size of test batch

        seed_id: int, Helps in reporoducing results of random operations
        
        OUTPUT:
        (On Success) Returns the testLoader
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        if self.dataset in self.datasets_accepted:
            n_datapts = len(data)
            idx = [i for i in range(n_datapts)]
            #np.random.shuffle(idx)

            test_sampler = SubsetRandomSampler(idx)

            testLoader = DataLoader(data, batch_size=test_batch_size, sampler=test_sampler)
            return testLoader

        else:
            raise NotImplementedError


    def loadPartitions(self, lSetPath, uSetPath, valSetPath):

        assert isinstance(lSetPath, str), "Expected lSetPath to be a string."
        assert isinstance(uSetPath, str), "Expected uSetPath to be a string."
        assert isinstance(valSetPath, str), "Expected valSetPath to be a string."

        lSet = np.load(lSetPath, allow_pickle=True)
        uSet = np.load(uSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        #Checking no overlap
        assert len(set(valSet) & set(uSet)) == 0,"Intersection is not allowed between validationset and uset"
        assert len(set(valSet) & set(lSet)) == 0,"Intersection is not allowed between validationset and lSet"
        assert len(set(uSet) & set(lSet)) == 0,"Intersection is not allowed between uSet and lSet"

        return lSet, uSet, valSet

    def loadTVPartitions(self, trainSetPath, valSetPath):

        assert isinstance(trainSetPath, str), "Expected trainSetPath to be a string."
        assert isinstance(valSetPath, str), "Expected valSetPath to be a string."

        trainSet = np.load(trainSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        #Checking no overlap
        assert len(set(valSet) & set(trainSet)) == 0,"Intersection is not allowed between validationset and trainSet"

        return trainSet, valSet


    def loadPartition(self, setPath):

        assert isinstance(setPath, str), "Expected setPath to be a string."

        setArray = np.load(setPath, allow_pickle=True)
        return setArray


    def saveSets(self, lSet, uSet, activeSet, save_dir):

        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(activeSet, dtype=np.ndarray)

        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/activeSet.npy', activeSet)

        # return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/activeSet.npy'


    def saveSet(self, setArray, setName, save_dir):

        setArray = np.array(setArray, dtype=np.ndarray)
        np.save(f'{save_dir}/{setName}.npy', setArray)
        return f'{save_dir}/{setName}.npy'


    def getClassWeightsFromDataset(self, dataset, index_set, bs):
        temp_loader = self.getIndexesDataLoader(indexes=index_set, batch_size=bs, data=dataset)
        return self.getClassWeights(temp_loader)


    def getClassWeights(self, dataloader):

        """
        INPUT
        dataloader: dataLoader
        
        OUTPUT
        Returns a tensor of size C where each element at index i represents the weight for class i. 
        """

        all_labels = []
        for _,y in dataloader:
            all_labels.append(y)
        print("===Computing Imbalanced Weights===")
        
        
        all_labels = np.concatenate(all_labels, axis=0)
        print(f"all_labels.shape: {all_labels.shape}")
        classes = np.unique(all_labels)
        print(f"classes: {classes.shape}")
        num_classes = len(classes)
        freq_count = np.zeros(num_classes, dtype=int)
        for i in classes:
            freq_count[i] = (all_labels==i).sum()
        
        #Normalize
        freq_count = (1.0*freq_count)/np.sum(freq_count)
        print(f"=== Sum(freq_count): {np.sum(freq_count)} ===")
        class_weights = 1./freq_count
        
        class_weights = torch.Tensor(class_weights)
        return class_weights
    

from pycls.datasets.custom_datasets import CoolRoofDataset
from pycls.datasets.preprocess_cool_roof import load_and_preprocess

def getDataset(name, split, transform=None, test_transform=None):
    if name == "cool_roof":
        if split == "train":
            features, labels, _, _ = load_and_preprocess("/home/yogendra/cool-roofs-active-learning/data/csv_files/chandigarh/chandigarh_roofs_varified.csv")
            return CoolRoofDataset(features, labels, transform=transform, test_transform=test_transform)
        elif split == "val":
            _, _, unseen_cool_features, unseen_noncool_features = load_and_preprocess("/home/yogendra/cool-roofs-active-learning/data/csv_files/chandigarh/chandigarh_roofs_varified.csv")
            features = np.vstack((unseen_cool_features, unseen_noncool_features))
            labels = np.hstack((np.ones(unseen_cool_features.shape[0]), np.zeros(unseen_noncool_features.shape[0])))
            return CoolRoofDataset(features, labels, transform=transform, test_transform=test_transform)
