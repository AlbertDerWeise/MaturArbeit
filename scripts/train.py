import numpy as np
import scipy
import pickle as pkl
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
from tqdm import tqdm
import itertools
from multiprocessing import Pool

LIKE_CAP = 10_000
WEIGHT = 0.75
AMP = 500
tags=['school', 'games', 'work', 'family',
     'everyday_life/health', 'money', 'relationships',
     'animals', 'sports', 'science_technology',
     'clothing', 'mental_state', 'literature_television',
     'nostalgia', 'drugs', 'celebrities',
     'society_and_social_procedures', 'gender_specific',
     'common_relatability', 'politics', 'sex', 'music_art',
     'ethnicity_culture_languages', 'political_incorrectness', 'misc']




def mutual_information(X, Y, bins=10):
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Compute joint histogram and marginal histograms
    joint_hist, x_edges, y_edges = np.histogram2d(X, Y, bins=bins, density=True)
    x_hist = np.histogram(X, bins=x_edges, density=True)[0]
    y_hist = np.histogram(Y, bins=y_edges, density=True)[0]

    # Flatten histograms to 1D arrays and remove zeros to avoid log(0) issues
    joint_hist = joint_hist.flatten()
    joint_hist = joint_hist[joint_hist > 0]
    x_hist = x_hist[x_hist > 0]
    y_hist = y_hist[y_hist > 0]

    # Calculate scipy.stat.entropy of each distribution
    H_X = scipy.stats.entropy(x_hist)              # scipy.stat.entropy of X
    H_Y = scipy.stats.entropy(y_hist)              # scipy.stat.entropy of Y
    H_XY = scipy.stats.entropy(joint_hist)         # Joint scipy.stat.entropy of (X, Y)

    # Mutual Information I(X; Y) = H(X) + H(Y) - H(X,Y)
    MI = H_X + H_Y - H_XY

    return MI
def dynamic_time_warping(X, Y):
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Lengths of the two sequences
    n, m = len(X), len(Y)

    # Initialize the cost matrix with infinity
    cost = np.full((n + 1, m + 1), float('inf'))
    cost[0, 0] = 0

    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Calculate the squared distance between points
            dist = (X[i - 1] - Y[j - 1]) ** 2
            # Update the cost matrix
            cost[i, j] = dist + min(cost[i - 1, j],    # Insertion
                                    cost[i, j - 1],    # Deletion
                                    cost[i - 1, j - 1])  # Match

    # The DTW distance is the square root of the final value in the cost matrix
    return np.sqrt(cost[n, m])



def distance_correlation(X, Y):
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Verify that the arrays have the same length
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Arrays must have the same length")

    # Number of samples
    n = X.shape[0]

    # Calculate pairwise distances
    A = np.abs(X[:, None] - X[None, :])
    B = np.abs(Y[:, None] - Y[None, :])

    # Centering the distance matrices
    A_centered = A - A.mean(axis=0) - A.mean(axis=1)[:, None] + A.mean()
    B_centered = B - B.mean(axis=0) - B.mean(axis=1)[:, None] + B.mean()

    # Calculate distance covariance
    dcov_XY = np.sum(A_centered * B_centered) / (n * n)
    dcov_X = np.sum(A_centered * A_centered) / (n * n)
    dcov_Y = np.sum(B_centered * B_centered) / (n * n)

    # Calculate distance correlation
    if dcov_X == 0 or dcov_Y == 0:
        return 0.0
    else:
        return np.sqrt(dcov_XY / np.sqrt(dcov_X * dcov_Y))
def corrrel(xarr, yarr):
    corrarr = np.correlate(xarr, yarr, mode='full')
    pi = scipy.signal.find_peaks(corrarr)[0]
    ph = [corrarr[p] for p in pi]
    ph = sorted(ph, reverse=True)
    mp = ph[0]
    ph[1:len(ph)//4]
    c = 1 - (np.sum(ph)/len(ph))/mp
    corrrel = c**2
    return corrrel
def smooth(arr, sigma=1):
    return scipy.ndimage.gaussian_filter1d(arr, sigma=sigma)
def normalize(arr) -> np.array:
    return arr / np.sum(arr)

def cos_sim(a, b):
    return cosine_similarity([a], [b])[0][0]

def pearson_corrcoeff(a, b):
    return np.corrcoef(a, b)[0][1]

def spearman_corrcoeff(a,b):
    return scipy.stats.spearmanr(a, b)[0]


def create_dist(it) -> np.ndarray:
    arr = np.zeros(len(tags))
    for _ in range(it):
        randarr = np.random.randn(5)
        arr[random.sample(range(0, len(arr)), 5)] += randarr
    return normalize(np.abs(arr))

def save_eval_hist(int):
    with open(os.path.join('../', 'synth_test', 'eval_hist.db'), 'rb') as f:
        hist = pkl.load(f)
    hist.append(int)
    with open(os.path.join('../', 'synth_test', 'eval_hist.db'), 'wb') as f:
        pkl.dump(hist, f)
        print('saving eval hist...')

def get_likes(dist, favs):
    zd = zip(tags,dist)
    ev = 0
    for tag, num in zd:
        if tag in list(favs):
            ev += num
    return int(round(ev*LIKE_CAP, 0))

def pdb(ind) -> None:
    with open(os.path.join('../', 'synth_test', 'memes.db'), 'rb') as f:
        db = pkl.load(f)
    del db[ind]
    with open(os.path.join('../', 'synth_test', 'memes.db'), 'wb') as f:
        pkl.dump(db, f)

def fit(arr) -> int:
    with open(os.path.join('../', 'synth_test', 'memes.db'), 'rb') as f:
        db = pkl.load(f)
    yarr = arr
    simarr = [cos_sim(xarr, yarr) for xarr in db]
    return db[np.argmax(simarr)], np.argmax(simarr)

def save_hist(arr) -> None:
    with open(os.path.join('../', 'synth_test', 'hist.db'), 'rb') as f:
        hist = pkl.load(f)
    hist.append(arr)
    with open(os.path.join('../', 'synth_test', 'hist.db'), 'wb') as f:
        pkl.dump(hist, f)
        print('saving hist..')

def create_white_noise(sum, len):
    return sum*normalize(np.abs(np.random.randn(len)))


class Prep:
    def __init__(self):
        if not os.path.exists(os.path.join(os.getcwd(),'../', 'synth_test')):
            os.mkdir(os.path.join(os.getcwd(),'../', 'synth_test'))
            print('Creating')
        return str(os.path.join(os.getcwd(),'../', 'synth_test'))

    def create_db(self):
        path = os.path.join('../', 'synth_test')
        with open(os.path.join(os.getcwd(),'../', 'synth_test', 'memes.db'), 'wb') as f:
            pkl.dump([create_dist(6) for _ in range(1000)], f)
            print('dumping...')
        return path

    def create_hist(self):
        path = os.path.join('../', 'synth_test')
        with open(os.path.join(path, 'hist.db'), 'wb') as f:
            pkl.dump([], f)
            print('dumping...')

    def create_eval_hist(self):
        path = os.path.join('../', 'synth_test')
        with open(os.path.join(path, 'eval_hist.db'), 'wb') as f:
            pkl.dump([0], f)
            print('dumping...')

    def main_vector_init(self):
        path = os.path.join('../', 'synth_test')
        with open(os.path.join(path,'main_vector.db'), 'wb') as f:
            pkl.dump(normalize(np.abs(np.random.randn(len(tags)))), f)
            print('dumping...')

    def prep(self):
        self.create_db(self)
        self.create_hist(self)
        self.main_vector_init(self)
        self.create_eval_hist(self)
        print('Prepped')






def comb_eval():
    with open(os.path.join('../', 'synth_test', 'hist.db'), 'rb') as f:
        hist = pkl.load(f)
        vc = [[] for i in range(len(tags))]
        for v in hist:
            for i, val in enumerate(v):
                vc[i].append(val)
        vc = np.array(vc)

        combs = list(itertools.combinations(range(25),3))
        print(len(combs))

    with open(os.path.join('../', 'synth_test', 'eval_hist.db'), 'rb') as f:
        eval_hist = list(pkl.load(f))
        del eval_hist[0]

    combarr = [vc[a] + vc[b] + vc[c] for a, b, c in combs]
    combarr = [normalize(comb) for comb in combarr]
    combarr = [smooth(np.diff(c)) for c in combarr]
    eval_hist = smooth(np.diff(eval_hist))
    print(len(combarr[0]), len(eval_hist))
    return combs, combarr, eval_hist

def get_most_similar(arr: np.ndarray, combarr) -> int:
    simarr = [corrrel(xarr, arr) for xarr in combarr]
    return np.argmax(simarr)

def get_highest_cossim(arr: np.ndarray, combarr) -> int:
    simarr = [cos_sim(xarr, arr) for xarr in combarr]
    return np.argmax(simarr)

def get_highest_pearson(arr: np.ndarray, combarr) -> int:
    simarr = [pearson_corrcoeff(xarr, arr) for xarr in combarr]
    return np.argmax(simarr)

def get_highest_spearman(arr: np.ndarray, combarr) -> int:
    simarr = [spearman_corrcoeff(xarr, arr) for xarr in combarr]
    return np.argmax(simarr)

def get_highest_distance_correlation(arr: np.ndarray, combarr) -> int:
    simarr = [distance_correlation(xarr, arr) for xarr in combarr]
    return np.argmax(simarr)
def get_highest_combined(arr: np.ndarray, combarr) -> int:
    simarr = [(cos_sim(xarr, arr) + pearson_corrcoeff(xarr, arr) + spearman_corrcoeff(xarr, arr) + corrrel(xarr, arr) + distance_correlation(xarr, arr))/5 for xarr in combarr]
    return np.argmax(simarr)

def get_highest_timewarping(arr: np.ndarray, combarr) -> int:
    simarr = [dynamic_time_warping(xarr, arr) for xarr in combarr]
    return np.argmax(simarr)

def get_highest_mi(arr: np.ndarray, combarr) -> int:
    simarr = [mutual_information(xarr, arr) for xarr in combarr]
    return np.argmax(simarr)

'''favs = ['money', 'everyday_life/health', 'sports']
most_sim = get_most_similar(eval_hist)
highest_cos = get_highest_cossim(eval_hist)
highest_pearson = get_highest_pearson(eval_hist)
highest_spearman = get_highest_spearman(eval_hist)
highest_combined = get_highest_combined(eval_hist)
highest_distance_correlation = get_highest_distance_correlation(eval_hist)
highest_timewarping = get_highest_timewarping(eval_hist)
highest_mi = get_highest_mi(eval_hist)
print(favs)
print([tags.index(f) for f in favs])
print(combs[most_sim])
print('corrrel ',[tags[f] for f in combs[most_sim]])
print('cos_sim ',[tags[f] for f in combs[highest_cos]])
print('pearson ',[tags[f] for f in combs[highest_pearson]])
print('spearman ',[tags[f] for f in combs[highest_spearman]])
print('distance_corr ',[tags[f] for f in combs[highest_distance_correlation]])
#print('timewarping ',[tags[f] for f in combs[highest_timewarping]])
print('mi ',[tags[f] for f in combs[highest_mi]])
print('comb', [tags[f] for f in combs[highest_combined]])'''


def update_vector(c, favorites):
    #print(favorites)
    path = os.path.join('../', 'synth_test')
    with open(os.path.join(path,'main_vector.db'), 'rb') as f:
        main_vector = pkl.load(f)
    save_hist(main_vector)
    dbarr, ind = fit(main_vector)
    pdb(ind)
    eval = get_likes(dbarr, favorites)
    save_eval_hist(eval)
    new_vector = normalize(smooth(WEIGHT*main_vector + (eval**2)*dbarr*(1-WEIGHT) + create_white_noise(len=(len(main_vector)), sum=AMP)))
    if c > 5:
        combs, combarr, evalhist = comb_eval()
        hc = get_highest_cossim(evalhist, combarr)
        for f in combs[hc]: new_vector[f] *= 5
        new_vector = normalize(new_vector)

        print('cos_sim ', [tags[f] for f in combs[hc]])
    with open(os.path.join(path,'main_vector.db'), 'wb') as f:
        pkl.dump(new_vector, f)
        print('dumping vector...')
        if c > 5 and set([tags[f] for f in combs[hc]]) == set(np.array(favorites)): print('Found after', c); return 1
        else: return 0

def save_inc(index, sum):
    if not os.path.exists(os.path.join('../','synth_test', 'increment.db')):
        with open(os.path.join('../', 'synth_test', 'increment.db'), 'wb') as f:
            pkl.dump([(index, sum)], f)
    else:
        with open(os.path.join('../', 'synth_test', 'increment.db'), 'rb') as f:
            indlist = pkl.load(f)
            indlist.append((index, sum))
        with open(os.path.join('../', 'synth_test', 'increment.db'), 'wb') as f:
            pkl.dump(indlist, f)
        print(index, sum, 'dumped')

def run(it, favs):
    Prep.prep(Prep)
    increment = []
    first_point = 0
    for i, c in tqdm(enumerate(range(it))):
        addition = update_vector(c, favs)
        if set(increment) == {0} and addition == 1:
            first_point = i
        increment.append(addition)
    save_inc(first_point, increment)
    with open(os.path.join('../', 'synth_test','eval_hist.db'), 'rb') as f:
        eval_hist = pkl.load(f)
        eval_hist = smooth(eval_hist, sigma=5)
        return eval_hist


'''favs = ['games', 'ethnicity_culture_languages', 'misc']'''

def stat(n, runit):
    with open(os.path.join('../', 'synth_test', 'favs.db'), 'wb') as f:
        pkl.dump([], f)
    with open(os.path.join('../', 'synth_test', 'results.db'), 'wb') as f:
        pkl.dump([], f)
    with open(os.path.join('../', 'synth_test', 'increment.db'), 'wb') as f:
        pkl.dump([], f)
    for i in tqdm(range(n)):
        if os.path.exists(os.path.join('../', 'synth_test','favs.db')):
            with open(os.path.join('../', 'synth_test', 'favs.db'), 'rb') as f:
                favs = pkl.load(f)
        else:
            with open(os.path.join('../', 'synth_test', 'favs.db'), 'wb') as f:
                pkl.dump([], f)
                favs = []
        favtags = random.sample(tags, 3)
        print(favtags)
        favs.append(favtags)
        with open(os.path.join('../', 'synth_test', 'favs.db'), 'wb') as f:
            pkl.dump(favs, f)
        ehist = run(runit, favtags)
        print(ehist, favtags)
        if not os.path.exists(os.path.join('../', 'synth_test', 'results.db')):
            with open(os.path.join('../', 'synth_test', 'results.db'), 'wb') as f:
                pkl.dump([ehist], f)
        else:
            with open(os.path.join('../', 'synth_test', 'results.db'), 'rb') as f:
                results = pkl.load(f)
                results.append(ehist)
            with open(os.path.join('../', 'synth_test', 'results.db'), 'wb') as f:
                pkl.dump(results, f)


stat(60, 60)