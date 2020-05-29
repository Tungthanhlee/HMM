import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
from scipy.special import softmax
from dataset import get_mfcc, get_class_data, clustering, get_dataset
from config import *
from sklearn.metrics import accuracy_score
import math

def hmm_model(n_com, start_prob, trans_prior):
    """
    Get model hmm learn
    """
    model = hmmlearn.hmm.MultinomialHMM(
            n_components=n_com*3, random_state=0, n_iter=1000, verbose=True,
            startprob_prior = start_prob,
            transmat_prior = trans_prior
    )
    return model

def train(dataset):
    # Get all vectors in the datasets
    all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in dataset.items()], axis=0)
    print("vectors", all_vectors.shape)
    # Run K-Means algorithm to get clusters
    kmeans = clustering(all_vectors)
    print("centers", kmeans.cluster_centers_.shape)

    models = {}
    for cname in CLASS_NAMES:
    #     print(cname[:4])
        # class_vectors = dataset[cname]
        # convert all vectors to the cluster index
        # dataset['one'] = [O^1, ... O^R]
        # O^r = (c1, c2, ... ct, ... cT)
        # O^r size T x 1
        dataset[cname] = list([kmeans.predict(v).reshape(-1,1) for v in dataset[cname]])

        if cname == "benh_nhan":
            hmm = hmm_model(N_COMPONENT_BN, START_PROB_BN, TRANSMAT_PRIOR_BN)
        elif cname == "cua":
            hmm = hmm_model(N_COMPONENT_CUA, START_PROB_CUA, TRANSMAT_PRIOR_CUA)
        elif cname == "khong":
            hmm = hmm_model(N_COMPONENT_KHONG, START_PROB_KHONG, TRANSMAT_PRIOR_KHONG)
        elif cname == "nguoi":
            hmm = hmm_model(N_COMPONENT_NGUOI, START_PROB_NGUOI, TRANSMAT_PRIOR_NGUOI)
        #define model
        # hmm = hmm_model()
        if 'test' not in cname:
            X = np.concatenate(dataset[cname])
            lengths = list([len(x) for x in dataset[cname]])
            print("training class", cname)
            print(X.shape, lengths, len(lengths))
            hmm.fit(X, lengths=lengths)
            models[cname] = hmm
    
    print("Training done")
    return models

def valid(models, dataset):
    print("Testing")
    # preds = []
    # ground_truths = []
    ACC = []
    for true_cname in CLASS_NAMES:
        preds = []
        ground_truths = []
        if 'test' in true_cname:
            for O in dataset[true_cname]:
                score_dict = {cname : model.score(O, [len(O)]) for cname, model in models.items()}
                score = [model.score(O, [len(O)]) for _, model in models.items()]
                label_pred = np.argmax(score, axis=0)
                preds.append(label_pred)
                
                gt = get_gt(true_cname)
                ground_truths.append(gt)
                print(true_cname, score_dict)
        acc = accuracy_score(ground_truths, preds)
        if not math.isnan(acc):
            ACC.append(acc)
        print(f"{true_cname} accuracy: ",acc)
    ACC = np.asarray(ACC)
    print(ACC)
    print(f"Mean acc: {np.mean(ACC)}")
    return preds
if __name__ == '__main__':

    dataset = get_dataset()

    trained_model = train(dataset)
    preds = valid(trained_model, dataset)
