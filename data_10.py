from utils import *
from algorithm import *
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--alg', help='SMVR, scsc, ogd, npag, Tscgd, NLASG, NASA, easy', type=str, default='SMVR')
    parser.add_argument('--lamda', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--M', type=int, default=5)
    parsed = vars(parser.parse_args())

    trials = parsed['trials']
    iters = parsed['iters']
    interval = parsed['eval_interval']
    lr = parsed['lr']
    alg = parsed['alg']
    lamda = parsed['lamda']
    batch_size = parsed['batch_size']
    alpha = parsed['alpha']
    M = parsed['M']
    beta = parsed['beta']

    print('Arguments:')
    print(f'Trials : {trials} \n Iters : {iters} \n Eval_interval : {interval} \n Learning Rate : {lr} \nAlgorithm : {alg} \nLambda : {lamda} \nBatch_size : {batch_size} \n Alpha :{alpha} \n Beta :{beta} \n M :{M}')

    make_log_file(alg, postfix='_10')

    path = "./data/10/10_Industry_Portfolios_Daily_1.csv"
    year, train_X = load_csv(path)

    loss_all_list, gradient_all_list, sample_list = [], [], []
    avg_loss_list, avg_gradient_list = [], []
    fw_all_list, gm_all_list = [], []
    avg_fw_list, avg_gm_list = [], []

    for i in range(trials):
        loss_list, gradient_list, trained_count = [], [], 0
        fw_list, gm_list = [], []
        time_list = []
        time_list.append(0)
        initial_time = time.time()
        if alg == 'Tscgd':
            time4Tscgd = 1
            lr = 0.3
        elif alg == 'ogd':
            theta = (np.random.rand(len(train_X[0])) * 2 - 1) / 1000

        elif alg == "scsc":
            T = iters * int(len(train_X) / batch_size)
            lr = 1 / np.sqrt(T)
            alpha = min(0.5, alpha)

        elif alg == "npag":
            alpha = 0

        elif alg == "NLASG":
            lr = 0.1
            T = iters * int(len(train_X) / batch_size)
            alpha = 1 / np.sqrt(T)

        np.random.seed(i)
        theta_before = (np.random.rand(len(train_X[0])) * 2 - 1) / 1000
        theta_before = theta_before / sum(theta_before)
        alpha_lst=[]
        layer1_before_lst=[]
        grad1_before_lst = []
        grad2_before_lst =[]
        grad_lst = []
        for j in range(iters):
            np.random.seed(123456+j)
            perm = np.random.permutation(len(train_X))
            train_X = train_X[perm]
            for ind in range(int(len(train_X) / batch_size)):

                trained_count += batch_size
                train_X_batch = train_X[ind * batch_size: (ind + 1) * batch_size]

                if j == 0 and ind == 0 and alg != 'ogd':
                    layer1_before = first_layer(theta_before, train_X_batch)
                    grad1_before = first_layer_grad(train_X_batch)
                    grad2_before = second_layer_grad(train_X_batch, layer1_before,lamda)
                    grad = all_grad(grad1_before, grad2_before)
                    theta = theta_before - lr * grad
                    theta = project_simplex(theta)
                    layer1_before_lst.append(layer1_before)
                    grad_lst.append(grad)
                    grad1_before_lst.append(grad1_before)
                    grad2_before_lst.append(grad2_before)
                    alpha_lst.append(0.3)

                if alg == 'ogd':
                    grad = OGD_Grad(theta, train_X_batch, lamda)

                elif alg == 'Tscgd':
                    time4Tscgd = time4Tscgd + 1
                    grad, layer1_before = T_SCGD(theta, theta_before, lamda, train_X_batch, time4Tscgd,
                                                                layer1_before)

                elif alg == 'SMVR':
                    grad, layer1_before, grad1_before, grad2_before = SMVR(theta,theta_before,train_X_batch,alpha, lamda,
                                                                                                              layer1_before,
                                                                                                              grad1_before,
                                                                                                              grad2_before)

                elif alg == 'scsc':
                    grad, layer1_before, = SCSC(theta, theta_before, lamda, train_X_batch,
                                                                    alpha, layer1_before)

                elif alg == 'npag':
                    grad, layer1_before,  grad1_before, grad2_before = SMVR(theta, theta_before,
                                                                                                train_X_batch,
                                                                                                0, lamda,
                                                                                                layer1_before,
                                                                                                grad1_before,
                                                                                                grad2_before)
                elif alg == 'NLASG':
                    theta, grad, layer1_before= NLASG(theta, grad, layer1_before, train_X_batch, lamda, alpha)
                elif alg == 'NASA':
                    theta, grad, layer1_before = NASA(theta, grad, layer1_before, train_X_batch, lamda, alpha, beta=beta, M=M)
                elif alg == 'easy':
                    theta,theta_before, grad, layer1_before = our_easy_method(theta, theta_before, train_X_batch,alpha, lamda, layer1_before, grad,lr)
                elif alg == 'improve':
                    theta, theta_before, grad, layer1_before = our_improve_method(theta, theta_before, grad, layer1_before, train_X_batch, lamda, alpha, beta, lr,M=M)

                if alg != 'NLASG' and alg != 'NASA' and alg !='easy' and alg != 'improve':
                    theta_before = theta
                    theta = theta - lr * grad
                    theta = project_simplex(theta)
                loss = Loss(theta, train_X, lamda)
                gradient = Grad(theta, train_X, lamda)
                fw_gap = FW_Gap(theta, train_X, lamda)
                grad_mapping = Gradient_Mapping(theta, train_X, lamda,1)
                time_gap = time.time() - initial_time

                loss_list.append(loss)
                gradient_list.append(gradient)
                time_list.append(time_gap)
                fw_list.append(fw_gap)
                gm_list.append(grad_mapping)

                # add sample
                sample_list.append(trained_count)

            if (j + 1) % interval == 0:
                print(f'[{j + 1} / {iters}] avg loss: {np.mean(np.array(loss_list))}')
                print(f'[{j + 1} / {iters}] avg gradient: {np.mean(np.array(gradient_list))}')
                print(f'[{j + 1} / {iters}] avg fw: {np.mean(np.array(fw_list))}')
                print(f'[{j + 1} / {iters}] avg gm: {np.mean(np.array(gm_list))}')

        avg_loss_list.append(np.mean(np.array(loss_list)))
        avg_gradient_list.append(np.mean(np.array(gradient_list)))
        avg_fw_list.append(np.mean(np.array(fw_list)))
        avg_gm_list.append(np.mean(np.array(gm_list)))
        print(f'[trials: {i}] avg loss: {np.mean(np.array(loss_list))}')
        print(f'[trials: {i}] avg gradient: {np.mean(np.array(gradient_list))}')
        print(f'[trials: {i}] avg fw: {np.mean(np.array(fw_list))}')
        print(f'[trials: {i}] avg gm: {np.mean(np.array(gm_list))}')

        loss_all_list.append(loss_list)
        gradient_all_list.append(gradient_list)
        fw_all_list.append(fw_list)
        gm_all_list.append(gm_list)

    print(f'avg loss: {np.mean(np.array(avg_loss_list))}')
    print(f'var loss: {np.var(np.array(avg_loss_list))}')

    print(f'avg gradient: {np.mean(np.array(avg_gradient_list))}')
    print(f'var gradient: {np.var(np.array(avg_gradient_list))}')

    print(f'avg fw: {np.mean(np.array(avg_fw_list))}')
    print(f'avg gm: {np.mean(np.array(avg_gm_list))}')

    write_log(batch_size, trials, iters, lr, interval, alg, lamda, alpha, avg_loss_list, avg_gradient_list,
              postfix='_10')
    if not os.path.exists('./figure_10/Loss'):
        os.makedirs('./figure_10/Loss')
    if not os.path.exists('./figure_10/Gradient'):
        os.makedirs('./figure_10/Gradient')
    if not os.path.exists('./figure_10/FW'):
        os.makedirs('./figure_10/FW')
    if not os.path.exists('./figure_10/GradientMapping'):
        os.makedirs('./figure_10/GradientMapping')

    to_csv(sample_list, loss_all_list, './figure_10/Loss/', alg, '_loss.csv')
    to_csv(sample_list, gradient_all_list, './figure_10/Gradient/', alg, '_gradient.csv')
    to_csv(sample_list, fw_all_list, './figure_10/FW/', alg, '_fw.csv')
    to_csv(sample_list, gm_all_list, './figure_10/GradientMapping/', alg, '_gm.csv')


if __name__ == '__main__':
    main()