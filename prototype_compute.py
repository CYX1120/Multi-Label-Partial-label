import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def extract_features(model, dataloader, args):

    # return features, labels
    model.eval()
    features = [torch.zeros(10, model.outputDim).cuda() for _ in range(args.classNum)]
    labels = []

    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(dataloader):
        input, target, groundTruth = input.cuda(), target.cuda(), groundTruth.cuda()

        with torch.no_grad():
            feature = model(input, onlyFeature=True).cuda()
            target_cpu = target

            for i in range(args.classNum):
                features[i] = torch.cat((features[i], feature[target_cpu[:, i] == 1, i]), dim=0)
                
            labels.append(target)

    features = [f[10:] for f in features]  # Remove the initial zeros
    print(f"extract_features shape is : {len(features)}")
    labels = torch.cat(labels, dim=0).cuda()  # Concatenate labels and move to GPU
    print(f"extract_labels shape is : {labels.shape}")

    return features, labels


def clean_features_by_clustering(prototype_nums, features, labels, class_num, prototype_num):
    """
    Clean features by clustering analysis and adjust labels accordingly.

    Parameters:
    - features: extracted feature matrix.
    - labels: label matrix.
    - class_num: number of classes.
    - prototype_num: number of prototypes for each class.

    Returns:
    - cleaned_features: cleaned feature matrix.
    - cleaned_labels: cleaned label matrix.
    """
    # Calculate the number of prototypes for each category
    cleaned_features = []
    cleaned_labels = []
    print("features length is : ", len(features))
    print("class_num is : ", class_num)
    for class_index in range(class_num):
        class_features = features[class_index]
        class_labels = labels[labels[:, class_index] == 1]

        ### prototype_num = prototype_nums[class_index] + 20
        
        # If the number of features in this category is less than the number of prototypes, all features are retained directly.
        if class_features.shape[0] <= prototype_num:
            cleaned_features.append(class_features)
            cleaned_labels.append(class_labels)
            continue
        
        # Transfer features to CPU and convert to numpy array for use with scikit-learn's KMeans
        class_features_np = class_features.cpu().numpy()
        
        #kmeans = KMeans(n_clusters=prototype_num, random_state=0).fit(class_features_np)##下面的为通义千问
        kmeans = KMeans(n_clusters=prototype_num, init='k-means++', random_state=0).fit(class_features_np)
        
        mask = np.ones_like(kmeans.labels_, dtype=bool)
        
        # Keep the filtered features and labels
        cleaned_features.append(torch.from_numpy(class_features_np[mask]).cuda())
        cleaned_labels.append(class_labels[mask].cuda())
    
    cleaned_features = torch.cat(cleaned_features, dim=0)
    print(f"cleaned_features shape is : {cleaned_features.shape}")
    cleaned_labels = torch.cat(cleaned_labels, dim=0)
    print(f"cleaned_labels shape is : {cleaned_labels.shape}")
    
    return cleaned_features, cleaned_labels

    
def compute_prototypes(model, dataloader, args):
    ## print("extract features...")
    features, labels = extract_features(model, dataloader, args)
    print("clean features by clustering...")
    dataset_instance = dataloader.dataset
    class_samples_count = calculate_class_samples_and_total_samples(dataset_instance)

    prototype_nums = compute_prototype_number(class_samples_count, args.prototypeNumber)

    # prototype_nums = [ 1 for i in range(args.classNum) ]

    args.prototype_nums = prototype_nums
    # print(args.prototype_nums)
    cleaned_features, cleaned_labels = clean_features_by_clustering(prototype_nums, features, labels, args.classNum, args.prototypeNumber)
    print("update prototypes...")
    model.update_prototypes(cleaned_features, cleaned_labels, prototype_nums)  #args.prototypeNumber
    return cleaned_features


def dynamic_prototype_number(class_samples, total_samples, max_prototypes=5):
    """Determine the number of prototypes based on the proportion of category samples to total samples"""
    return min(int(class_samples / total_samples * max_prototypes), max_prototypes)


def calculate_class_samples_and_total_samples(dataset):
    """
    Count the number of samples for each class and the total number of samples.

    Parameters:
    - dataset: your dataset instance (e.g. COCO2014, VG, or VOC2007 instance).

    Returns:
    - class_samples_count: a list of the number of samples for each class.
    - total_samples: the total number of all samples.
    """
    class_samples = [0] * dataset.labels.shape[1]  # The first dimension is the number of samples, and the second dimension is the number of categories.
    # total_samples = 0

    for labels in dataset.labels:
        # total_samples += 1
        for idx, label in enumerate(labels):
            if label == 1: 
                class_samples[idx] += 1

    return class_samples# , total_samples

def compute_prototype_number(class_samples_count, max_prototypes=50):
    """
    Dynamically calculate the number of prototypes for each category based on the number of class samples.

    Parameters:
    - class_samples_count: List of the number of samples for each category.
    - max_prototypes: Maximum number of prototypes.

    Returns:
    - prototype_nums: List of the number of prototypes for each category.
    """
    prototype_nums = []
    for class_samples in class_samples_count:
        prototype_num = min(max_prototypes, max(1, int(np.sqrt(class_samples / 2 ))))
        prototype_nums.append(prototype_num)
    # print("prototype_nums: ", prototype_nums)
    
    return prototype_nums


def save_tensors(filename, features, labels):
    torch.save({"features": features, "labels": labels}, filename)