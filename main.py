import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
from scipy.special import softmax
from dataset import get_mfcc, get_class_data, clustering, get_dataset
from config import CLASS_NAMES,TRANSMAT_PRIOR, START_PROB
from config import get_gt
from sklearn.metrics import accuracy_score

def hmm_model():
    """
    Get model hmm learn
    """
    model = hmmlearn.hmm.MultinomialHMM(
            n_components=6, random_state=0, n_iter=1000, verbose=True,
            startprob_prior = START_PROB,
            transmat_prior = TRANSMAT_PRIOR
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

        #define model
        hmm = hmm_model()
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
        print(f"{true_cname} accuracy: ",acc)
    return preds
if __name__ == '__main__':

    dataset = get_dataset()

    trained_model = train(dataset)
    preds = valid(trained_model, dataset)
