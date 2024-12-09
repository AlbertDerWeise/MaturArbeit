import numpy as np
import pickle as pkl
import os
from matplotlib import pyplot as plt
import scipy
import itertools

def corrrel(xarr, yarr):
    corrarr = np.correlate(xarr, yarr, mode='full')
    pi = scipy.signal.find_peaks(corrarr)[0]
    ph = [corrarr[p] for p in pi]
    ph = sorted(ph, reverse=True)
    mp = ph[0]
    ph = ph[1:len(ph)//4]
    if ph == []: ph=[0]
    c = 1 - (np.sum(ph)/len(ph))/mp
    corrrel = c**2
    return corrrel
def find_best_fit(y, max_degree=5):
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    x = np.arange(len(y))
    x_scaled = (x - np.mean(x)) / np.std(x)
    y_scaled = (y - np.mean(y)) / np.std(y)

    best_degree = None
    best_mse = float('inf')
    best_coefficients = None

    for degree in range(1, max_degree + 1):
        coefficients = np.polyfit(x_scaled, y_scaled, degree)
        polynomial = np.poly1d(coefficients)
        y_pred = polynomial(x_scaled)
        mse = mean_squared_error(y_scaled, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_degree = degree
            best_coefficients = coefficients

    best_polynomial = np.poly1d(best_coefficients)
    fit_curve_scaled = best_polynomial(x_scaled)
    fit_curve = fit_curve_scaled * np.std(y) + np.mean(y)

    return fit_curve, best_degree
PATH = os.path.join('../', 'synth_test')

def calculate_cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def ema():
    with open(os.path.join(PATH, 'results.db'), 'rb') as f:
        results = pkl.load(f)

    print(len(results))

    plt.figure(figsize=(8, 6))

    emaarr = results[0]
    for arr in results[1:]:
        emaarr = 0.1 * arr + 0.9 * emaarr  # Exponential Moving Average



    coms= list(itertools.combinations(results, 2))
    cossimarr = [corrrel(x,  y) for x,y in coms]
    print(np.sum(cossimarr)/len(cossimarr))


    # Add legend


    # Plot each array
    for array in results:
        plt.plot(array, alpha=0.05, color='purple')  # Alpha sets transparency

    plt.plot(emaarr, color="#006400", label="Exponential Moving Average")

    plt.legend()

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Evaluation')
    plt.title('Individual development curves & exponential moving average')
    plt.show()

def normalize(array):
    return array / np.sum(array)

def inc():
    with open(os.path.join(PATH, 'increment.db'), 'rb') as f:
        results = pkl.load(f)

    first_index = [x for x,y in results]
    print(f'{first_index = }')
    ave_first = np.sum(first_index)/len(first_index)
    print('median' ,np.mean(first_index))
    print(f'{ave_first = }')
    functions = [y for x, y in results]
    sums = [sum(f) for f in functions]
    print(f'average sum = {np.sum(sums) / len(sums)}')
    ema_arr = functions[0]
    for f in functions[1:]:
        plt.plot(f, color='purple', alpha=0.03)
        ema_arr = 0.1 * np.array(f) + 0.9 * np.array(ema_arr)
    standard_deviation = np.std(ema_arr)
    p_dist = normalize(np.cumsum(ema_arr))
    #plt.plot(p_dist)
    chance = 0
    for e, i in enumerate(p_dist):
        chance += i
        if chance >= 0.5:
            print(f'50% chance at index {e}')
            break


    print(standard_deviation)
    plt.plot(scipy.ndimage.gaussian_filter1d(ema_arr, 1.5), color="green", label='EMA of correct hits: probability curve')
    plt.legend()
    plt.title('increment gradient and probability')
    plt.show()

inc()