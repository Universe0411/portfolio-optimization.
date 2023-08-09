import numpy as np
import csv
import os

def load_csv(path):
    with open(path,encoding = 'utf-8') as f:
        data = np.loadtxt(f, dtype=str, delimiter=",")
    X = data[0:, 0].astype(float)
    Y = data[0:, 1:].astype(float)
    return X, Y

def to_csv(sample_list, num_list, result_path, alg, postfix):
    num_np = np.array(num_list).T
    num_mean = np.mean(num_np, axis=1)
    num_var = np.var(num_np, axis=1)
    log = result_path + alg + postfix
    with open(log, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(num_mean)):
            writer.writerow([sample_list[i], num_mean[i], num_var[i]])

def write_log(batch_size, trials, iters, lr, interval,  alg, lamda, beta, loss, gradient, postfix=''):
    result_folder = "./results" + postfix + "/"
    logname = result_folder + 'log_batch.csv'
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            [batch_size, trials, iters, lr, interval,  alg, lamda, beta, np.mean(np.array(loss)), np.var(np.array(loss)),
             np.mean(np.array(gradient)), np.var(np.array(gradient))])

    logname = result_folder + alg + '_log_batch.csv'
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            [batch_size, trials, iters, lr, interval,  alg, lamda, beta, np.mean(np.array(loss)), np.var(np.array(loss)),
             np.mean(np.array(gradient)), np.var(np.array(gradient))])

def make_log_file(algorithm, postfix=''):
    result_folder = "./results" + postfix + "/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    logname = result_folder + 'log_batch.csv'
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['batch_size', 'trials', 'iters', 'lr', 'eval_interval', 'alg', 'lamda', 'beta', 'avg_loss', 'var_loss', 'avg_gradient', 'var_gradient'])

    logname = result_folder + algorithm + '_log_batch.csv'
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['batch_size', 'trials', 'iters', 'lr', 'eval_interval', 'alg', 'lamda', 'beta', 'avg_loss', 'var_loss', 'avg_gradient', 'var_gradient'])

