import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def getIntraPseudoLabel(intraCoOccurrence, target, margin=0.50):
    """
    Shape of intraCoOccurrence : (batchSize, classNum ** 2)
    Shape of target : (batchSize, classNum)
    """
    
    batchSize, classNum = target.size(0), target.size(1)
    probCoOccurrence = torch.sigmoid(intraCoOccurrence)

    indexStart, indexEnd, pseudoLabel = 0, 0, torch.zeros((batchSize, classNum, classNum)).cuda()
    for i in range(classNum):
        pseudoLabel[:, i, i] = 1
        indexStart = indexEnd
        indexEnd += classNum-i-1
        pseudoLabel[:, i, i+1:] = probCoOccurrence[:, indexStart:indexEnd]
        pseudoLabel[:, i+1:, i] = probCoOccurrence[:, indexStart:indexEnd]

    target_ = target.detach().clone()
    target_[target_ != 1] = 0
    pseudoLabel = torch.sum(pseudoLabel * target_.view(batchSize, 1, classNum).repeat(1, classNum, 1), dim=2)
    pseudoLabel = pseudoLabel / torch.clamp(torch.sum(target_, dim=1), min=1).view(batchSize, 1).repeat(1, classNum)
    pseudoLabel = torch.clamp(pseudoLabel-margin, min=0, max=1) 

    return pseudoLabel


def getInterPseudoLabel(feature, target, posFeature, classNum, prototype_nums, margin=0.50):

    batch_size = feature.size(0)
    feature_dim = feature.size(2)
    
    pseudoLabel = []

    start_idx = 0
    for i in range(classNum):
        prototype_num = prototype_nums[i]
        end_idx = start_idx + prototype_num

        # Reshape posFeature for the current class
        class_posFeature = posFeature[start_idx:end_idx]  # (prototypeNumber, featureDim)
        class_posFeature = class_posFeature.unsqueeze(0).repeat(batch_size, 1, 1)  # (batchSize, prototypeNumber, featureDim)
        # print(f"{i} -th category class_posFeature shape: {class_posFeature.shape}")
        
        # Extract the feature for the current class
        class_feature = feature[:, i, :]  # (batchSize, featureDim)
        class_feature = class_feature.unsqueeze(1).repeat(1, prototype_num, 1)  # (batchSize, prototypeNumber, featureDim)
        # print(f"{i} -th category class_feature shape: {class_feature.shape}") 
        
        # Use torch.nn.functional.cosine_similarity to compute cosine similarity
        posDistance = torch.nn.functional.cosine_similarity(class_feature, class_posFeature, dim=2)  # (batchSize, prototypeNumber)
        # print(f"{i} -th category posDistance shape: {posDistance.shape}")
        
        # Calculate the pseudo label for the current class
        class_pseudoLabel = torch.clamp(torch.mean(posDistance, dim=1) - margin, min=0, max=1)  # (batchSize)
        pseudoLabel.append(class_pseudoLabel)
        
        start_idx = end_idx
    
    pseudoLabel = torch.stack(pseudoLabel, dim=1)  # Stack all class pseudo labels to shape (batchSize, classNum)
    
    return pseudoLabel


def getInterAdjMatrix(target):
    """
    Shape of target : (BatchSize, ClassNum)
    """

    target_ = target.detach().clone().permute(1, 0)
    target_[target_ != 1] = 0
    adjMatrix = target_.unsqueeze(1).repeat(1, target.size(0), 1) * target_.unsqueeze(2).repeat(1, 1, target.size(0))
    eyeMatrix = torch.eye(target.size(0)).unsqueeze(0).repeat(target.size(1), 1, 1).cuda()
    adjMatrix = torch.clamp(adjMatrix + eyeMatrix, max=1, min=0)

    return adjMatrix


class BCELoss(nn.Module):

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(BCELoss, self).__init__()

        self.margin = margin

        self.reduce = reduce
        self.size_average = size_average

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input, target):

        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)

        loss = positive_mask * positive_loss + negative_mask * negative_loss

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.mean(loss)
            return torch.sum(loss[(positive_mask > 0) | (negative_mask > 0)]) if torch.sum(positive_mask + negative_mask) != 0 else torch.sum(loss)
        return loss


class AsymmetricLoss(nn.Module):

    def __init__(self, margin=0, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduce=None, size_average=None):
        super(AsymmetricLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Get positive and negative mask
        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            input_sigmoid_neg = (input_sigmoid_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = positive_mask * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = negative_mask * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob = input_sigmoid_pos * positive_mask + input_sigmoid_neg * negative_mask
            one_sided_gamma = self.gamma_pos * positive_mask + self.gamma_neg * negative_mask
            one_sided_weight = torch.pow(1 - prob, one_sided_gamma)

            loss *= one_sided_weight

        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss


class intraAsymmetricLoss(nn.Module):

    def __init__(self, classNum, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduce=None, size_average=None):
        super(intraAsymmetricLoss, self).__init__()

        self.classNum = classNum
        self.concatIndex = self.getConcatIndex(classNum)

        self.reduce = reduce
        self.size_average = size_average

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, \sum_{i=1}^{classNum-1}{i})
        Shape of target: (BatchSize, classNum)
        """
        target = target.cpu().data.numpy()
        target1, target2 = target[:, self.concatIndex[0]], target[:, self.concatIndex[1]]
        target1, target2 = (target1 > 0).astype(np.float), (target2 > 0).astype(np.float) # type: ignore
        target = torch.Tensor(target1 * target2).cuda()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            input_sigmoid_neg = (input_sigmoid_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = target * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = (1 - target) * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob = input_sigmoid_pos * target + input_sigmoid_neg * (1 - target)
            one_sided_gamma = self.gamma_pos * target + self.gamma_neg * (1 - target)
            one_sided_weight = torch.pow(1 - prob, one_sided_gamma)

            loss *= one_sided_weight

        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss

    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res




class InstanceContrastiveLoss(nn.Module):
    """
    Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, batchSize, reduce=None, size_average=None):
        super(InstanceContrastiveLoss, self).__init__()

        self.batchSize = batchSize
        self.concatIndex = self.getConcatIndex(batchSize)

        self.reduce = reduce
        self.size_average = size_average

        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, classNum, featureDim)
        Shape of target: (BatchSize, classNum), Value range of target: (-1, 0, 1)  -1 indicates non-existence, 0 indicates uncertainty, and 1 indicates existence
        """

        target_ = target.detach().clone()
        target_[target_ != 1] = 0
        pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        pos2negTarget = 1 - pos2posTarget
        pos2negTarget[(target[self.concatIndex[0]] == 0) | (target[self.concatIndex[1]] == 0)] = 0
        pos2negTarget[(target[self.concatIndex[0]] == -1) & (target[self.concatIndex[1]] == -1)] = 0

        target_ = -1 * target.detach().clone()
        target_[target_ != 1] = 0
        neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

        distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

        if self.reduce:
            pos2pos_loss = (1 - distance)[pos2posTarget == 1]
            pos2neg_loss = (1 + distance)[pos2negTarget == 1]
            neg2neg_loss = (1 + distance)[neg2negTarget == 1]

            if pos2pos_loss.size(0) != 0:
                if neg2neg_loss.size(0) != 0:
                    neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                              torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
                if pos2neg_loss.size(0) != 0:
                    if pos2neg_loss.size(0) != 0:
                        pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
                                                  torch.sort(pos2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)

            loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)

            if self.size_average:
                return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
            return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())

        return distance
  
    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum - 1):
            res[0] += [index for i in range(classNum - index - 1)]
            res[1] += [i for i in range(index + 1, classNum)]
        return res
    
class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, classNum, reduce=True, size_average=True, likelihood_topk=14, loss_mode = 'Default', prior_path='None'):
        super(PrototypeContrastiveLoss, self).__init__()
        self.reduce = reduce
        self.size_average = size_average
        self.classNum = classNum
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)  # 使用 dim=2 进行余弦相似度计算
        self.likelihood_topk = likelihood_topk
        self.loss_mode = loss_mode

        self.prior_classes = None
        if prior_path != 'None':
            df = pd.read_csv(prior_path)
            self.prior_classes = dict(zip(df.values[:, 0], df.values[:, 1]))
            class_to_index = {class_name: i for i, class_name in enumerate(df['Classes'])}
            # Using class indexes as dictionary keys
            self.prior_classes = {class_to_index[class_name]: avg_pred for class_name, avg_pred in zip(df['Classes'], df['avg_pred'])}
            print("Prior information loaded successfully.")
        
        log_file_path = '/export/home/cyx/Project/MLRL1/record_coco6_8.txt'
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                pass
        self.log_file = open(log_file_path, 'w')

    def forward(self, input, target, prototype, prototype_nums, epoch):
        """
        Input shape: (BatchSize, classNum, featureDim)
        Target shape: (BatchSize, classNum), target value range: (0, 1, -1)
        Prototype shape: (total_prototype_count, featureDim)
        """
        batchSize, featureDim = input.size(0), input.size(2)

        start_idx = 0
        total_loss = 0
        probabilities = torch.zeros((batchSize, self.classNum), device=input.device)
        # Calculate the similarity and probability of all categories
        for i in range(self.classNum):
            prototype_num = prototype_nums[i]
            end_idx = start_idx + prototype_num

            # Extract the prototype of the current category
            class_prototype = prototype[start_idx:end_idx]  # (prototypeNum, featureDim)
            class_prototype = class_prototype.unsqueeze(0).repeat(batchSize, 1, 1)  # (BatchSize, prototypeNum, featureDim)

            # Extract features of the current category
            class_input = input[:, i, :]  # (BatchSize, featureDim)
            class_input = class_input.unsqueeze(1).repeat(1, prototype_num, 1)  # (BatchSize, prototypeNum, featureDim)

            # Calculate cosine similarity
            distance = torch.mean(self.cos(class_input, class_prototype), dim=1)  # (BatchSize)

            # Store the probability of belonging to the current category
            probabilities[:, i] = 1 - distance  # The probability of belonging to the current category

            # Calculate the loss for the current category
            ## class_loss = torch.where(target[:, i] == -1, 1 + distance, 1 - distance)
            class_loss = torch.where(target[:, i] == 1, 1 - distance, 1 + distance)

            if self.loss_mode == 'Ignore':
                class_loss = torch.where(      # ignore mode
                    target[:, i] == 1,  
                    1 - distance,  
                    torch.where(
                        target[:, i] == -1, 
                        1 + distance, 
                        0    
                    )
                )

            # Eliminate uncertain samples based on prior knowledge and probability
            if self.loss_mode == 'Default':
                if self.likelihood_topk > 0 and self.prior_classes is not None:
                    avg_pred_i = self.prior_classes.get(i, 0)  # Get the average prediction value of the current category
                    unknown_mask = target[:, i] == 0  # Labeling samples with uncertain labels
                    threshold = avg_pred_i * self.likelihood_topk
                    uncertain_mask = unknown_mask & (probabilities[:, i] < threshold)
                    class_loss[uncertain_mask] *= 12  # Multiply this loss by 12

                    uncertain_mask_high = unknown_mask & (probabilities[:, i] > threshold)
                    class_loss[uncertain_mask_high] = 0  # Set this part of the loss to 0

            total_loss += class_loss.sum()
            start_idx = end_idx

        if self.reduce:
            if self.size_average:
                return total_loss / (batchSize * self.classNum)
            return total_loss
        return total_loss


def computePrototype(model, train_loader, args):

    from sklearn.cluster import KMeans

    model.eval()
    prototypes, features = [], [torch.zeros(10, model.outputDim) for i in range(args.classNum)]

    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):

        input, target, groundTruth = input.cuda(), target.cuda(), groundTruth.cuda()

        with torch.no_grad():
            feature = model(input, onlyFeature=True).cpu()
            for i in range(args.classNum):
                if features[i].device != 'cpu':
                    features[i] = features[i].cpu()

            target_cpu = target.cpu()

            for i in range(args.classNum):
                features[i] = torch.cat((features[i], feature[target_cpu[:, i] == 1, i]), dim=0)

    for i in range(args.classNum):
        kmeans = KMeans(n_clusters=args.prototypeNumber).fit(features[i][10:].numpy())
        prototypes.append(torch.tensor(kmeans.cluster_centers_).cuda())
    model.prototype = torch.stack(prototypes, dim=0)

def count_elements_numpy(matrix): # type: ignore
    matrix_array = matrix.cpu().numpy()
    unique, counts = np.unique(matrix_array, return_counts=True)
    result = dict(zip(unique, counts))
    
    # Make sure all required elements are present in the result dictionary
    for value in [-1, 0, 1]:
        if value not in result:
            result[value] = 0
    
    return result
