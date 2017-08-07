import numpy as np
import matplotlib.pyplot as plt
import math

def calculate_elu(valArray):
    temp_list = list()
    for val in valArray:
        if val < 0:
            temp_list.append(math.exp(val) - 1)
        else:
            temp_list.append(val)
    return temp_list

step = 10
x_axis = np.linspace(step*-1, step, num=step*20+1)
zeros = np.zeros(x_axis.shape)
relu = np.maximum(x_axis, zeros)
softplus = np.log(np.exp(x_axis) + 1)
softsign = x_axis / (np.absolute(x_axis) + 1)
sigmoid = 1 / (1+np.exp(x_axis*-1))
tanh = np.tanh(x_axis)
elu = calculate_elu(x_axis)


plt.plot(x_axis, relu, label='relu')
plt.plot(x_axis, softplus, label='softplus')
plt.plot(x_axis, softsign, label='softsign')
plt.plot(x_axis, sigmoid, label='sigmoid')
plt.plot(x_axis, tanh, label='tanh')
plt.plot(x_axis, elu, label='elu')
plt.legend(loc=2)
plt.grid(True)
plt.show()
