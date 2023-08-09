import numpy as np

#from github
def project_simplex(v, z=1.0, axis=-1):
    def project_2d(v, z):
        shape = v.shape
        if shape[1] == 1:
            w = np.array(v)
            w[:] = z
            return w

        mu = np.sort(v, axis=1)
        mu = np.flip(mu, axis=1)
        cum_sum = np.cumsum(mu, axis=1)
        j = np.expand_dims(np.arange(1, shape[1] + 1), 0)
        rho = np.sum(mu * j - cum_sum + z > 0.0, axis=1, keepdims=True) - 1
        max_nn = cum_sum[np.arange(shape[0]), rho[:, 0]]
        theta = (np.expand_dims(max_nn, -1) - z) / (rho + 1)
        w = (v - theta).clip(min=0.0)
        return w

    shape = v.shape

    if len(shape) == 0:
        return np.array(1.0, dtype=v.dtype)
    elif len(shape) == 1:
        return  project_2d(np.expand_dims(v, 0), z)[0, :]
    else:
        axis = axis % len(shape)
        t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
        tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
        v_t = np.transpose(v, t_shape)
        v_t_shape = v_t.shape
        v_t_unroll = np.reshape(v_t, (-1, v_t_shape[-1]))

        w_t = project_2d(v_t_unroll, z)

        w_t_reroll = np.reshape(w_t, v_t_shape)
        return np.transpose(w_t_reroll, tt_shape)

def first_layer(theta,r):#(d+1,1)
    return np.concatenate((-1*np.array([np.mean(np.dot(r, theta))]), theta), axis=0)

def second_layer(r, layer1, lamda):
    theta = layer1[1:]
    layer1 = layer1[0]
    # 1
    return layer1 + lamda * np.mean((np.dot(r, theta) + layer1) ** 2)

def first_layer_grad(r):
    d =r.shape[1]
    I = np.eye(r.shape[1]) #dxd
    return np.concatenate((np.array([-np.mean(r, axis=0)]).reshape(d,1), I), axis=1) # d x d+1

def second_layer_grad(r, layer1,lamda):
    y_1 = layer1[0]
    y_2 = layer1[1:]
    first = np.array([1 - 2 * lamda * np.mean((np.dot(r, y_2) + y_1))])
    second = 2 * lamda * np.dot(r.T, (np.dot(r, y_2) + y_1)) / r.shape[0]
    return np.concatenate((first, second), axis=0)

def all_grad(grad1, grad2,):
    # f_2/y_1 y_1/theta
    part1 = grad2[0] * grad1[:,0]
    # f_2/y_2 y_2/theta
    part2 = np.dot(grad1[:,1:],grad2[1:])
    return part1 + part2 # d

def first_v_estimate(value_before, theta, theta_before, r, alpha):
    return (1 - alpha) * value_before + first_layer(theta, r) - (1 - alpha) * first_layer(theta_before, r)

def project(v):
    return v

def first_grad_estimate(value_before, r, alpha):
    return project((1 - alpha) * value_before + first_layer_grad(r) - (1 - alpha) * first_layer_grad(r))

def second_grad_estimate(value_before, layer1, layer1_before, r, alpha, lamda):
    temp1= second_layer_grad(r, layer1, lamda)
    temp2= second_layer_grad(r, layer1_before,lamda)
    return project((1 - alpha) * value_before + temp1 - (1 - alpha) * temp2)

def SMVR(theta, theta_before, r, alpha, lamda, layer1_before,  grad1_before, grad2_before):
    layer1 = first_v_estimate(layer1_before, theta, theta_before, r, alpha)
    grad1 = first_grad_estimate(grad1_before, r, alpha)
    grad2 = second_grad_estimate(grad2_before,layer1, layer1_before, r, alpha, lamda)
    grad = all_grad(grad1, grad2)
    return grad, layer1, grad1, grad2

def SCSC(theta, theta_before, lamda, r, alpha, layer1_before):
    layer1 = first_v_estimate(layer1_before, theta, theta_before, r, alpha)
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1,lamda)
    grad = all_grad(grad1, grad2)
    return grad, layer1

def OGD_Grad(theta, r, lamda):
    layer1 = first_layer(theta, r)
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)
    return grad

def NLASG(theta_before, grad_before, layer1_before, r, lamda, beta):
    u = theta_before - grad_before / (2 * np.sqrt(3)) # beta?? 可能固定为了2根号3
    u = project_simplex(u)
    theta = (1 - beta) * theta_before + beta * u # x

    layer1 = first_layer(theta_before, r)
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)
    Grad = (1 - beta) * grad_before + beta * grad # z
    #w
    ELayer1 = (1 - beta) * layer1_before + beta * layer1 + np.dot(theta - theta_before, grad1)

    return theta, Grad, ELayer1

def NASA(theta_before, grad_before, layer1_before, r, lamda,alpha, beta, M=10):
    def ICG(x, z, beta, num_iter):
        y = x  # y -> w_0
        for _ in range(num_iter):
            index = np.argmin((z + beta * (y - x)))
            v = np.zeros(x.shape)
            v[index]=1
            if np.linalg.norm(v - y, 2) < 1e-4:
                return y
            else:
                mu = min(1, np.dot((beta * (x - y) - z).T, (v - y)) / (beta * np.linalg.norm(v - y, 2) ** 2))
            y = (1 - mu) * y + mu * v
        return y

    y = ICG(theta_before, grad_before, beta, M)
    theta = theta_before + alpha*(y - theta_before)
    layer1 = first_layer(theta_before, r)
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)
    Grad = (1 - alpha) * grad_before + alpha * grad # z
    Elayer1 = (1-alpha) * layer1_before + alpha * layer1 + np.dot(theta - theta_before, grad1)

    return theta, Grad, Elayer1

def T_SCGD(theta, theta_before, lamda, r, time, Layer1_before):
    def T_first_fv_estimate(layer1_before, theta, theta_before, r, time):
        beta1 = min(1, 2 * (time ** (-5 / 10)))
        tmp = (1 - 1 / beta1) * theta_before + theta / beta1
        return (1 - beta1) * layer1_before + beta1 * first_layer(tmp, r)

    layer1 = T_first_fv_estimate(Layer1_before, theta, theta_before, r, time)
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)
    return time ** (-6 / 10) * grad, layer1

def our_easy_method(theta, theta_before, r, alpha, lamda, layer1_before,  grad_before, lr):
    #u
    layer1 = first_v_estimate(layer1_before, theta, theta_before, r, alpha)
    #V
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)

    old_grad2 = second_layer_grad(r, layer1_before, lamda)
    old_grad = all_grad(grad1,old_grad2)

    grad = grad + (1 - alpha) * (grad_before - old_grad)
    index = np.argmin(grad)
    z = np.zeros(grad.shape)
    z[index]=1
    theta_before = theta
    theta = theta + lr * (z - theta)
    return theta, theta_before, grad, layer1

def our_improve_method(theta, theta_before, grad_before, layer1_before, r, lamda, alpha, beta, lr, M=10):

    layer1 = first_v_estimate(layer1_before, theta, theta_before, r, alpha)
    # V
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)

    old_grad2 = second_layer_grad(r, layer1_before, lamda)
    old_grad = all_grad(grad1, old_grad2)

    grad = grad + (1 - alpha) * (grad_before - old_grad)
    def ICG(x, v, beta, num_iter):
        w = x
        for _ in range(num_iter):
            index = np.argmin(v + beta * (w - x))
            z = np.zeros(x.shape)
            z[index] = 1
            if np.linalg.norm(w - z, 2) < 1e-4:
                return w
            else:
                gamma = min(1, np.dot((beta * (x - w) - v).T,(z - w))/(beta * np.linalg.norm(z - w, 2) ** 2))
            w = (1 - gamma) * w + gamma * z
        return w

    z = ICG(theta, grad, beta, M)
    theta_before = theta
    theta = theta + lr * (z - theta)
    return theta , theta_before, grad, layer1

def Loss(theta, r, lamda):
    outputs = np.dot(r, theta)
    return -np.mean(outputs) + lamda * (np.std(outputs)) ** 2


def Grad(theta, r, lamda):
    layer1 = first_layer(theta, r)
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)
    return np.linalg.norm(grad, ord=2)

def FW_Gap(theta, r, lamda):
    layer1 = first_layer(theta, r)
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)
    index = np.argmin(grad)
    z = np.zeros(theta.shape)
    z[index] = 1.0
    return np.dot((z - theta).T , -grad)

def Gradient_Mapping(theta, r, lamda, beta):
    layer1 = first_layer(theta, r)
    grad1 = first_layer_grad(r)
    grad2 = second_layer_grad(r, layer1, lamda)
    grad = all_grad(grad1, grad2)
    theta_2 = project_simplex(theta - grad / beta)
    return np.linalg.norm(beta*(theta - theta_2), ord=2)
