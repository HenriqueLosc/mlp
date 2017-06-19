import sys
import numpy as np
import math as m


def f(net):
    return 1 / (1 + m.exp(-net))


def df_dnet(net):
    return f(net) * (1 - f(net))


class Arch(object):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 f=f,
                 df=df_dnet
                 ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.output_weights = np.random.rand(output_size, hidden_size + 1) - 0.5
        self.hidden_weights = np.random.rand(hidden_size, input_size + 1) - 0.5

        self.f = f
        self.df = df

    def forward(self, x_i):
        hidden_out_f = np.zeros(self.hidden_size)
        hidden_out_df = np.zeros(self.hidden_size)

        # hidden layer
        for i in range(self.hidden_size):
            hidden_net_i = np.dot(np.append(x_i, [1]), np.transpose(self.hidden_weights[i, :]))
            hidden_out_f[i] = self.f(hidden_net_i)
            hidden_out_df[i] = self.df(hidden_net_i)

        output_out_f = np.zeros(self.output_size)
        output_out_df = np.zeros(self.output_size)
        # output layer
        for j in range(self.output_size):
            output_net_i = np.dot(np.append(hidden_out_f, [1]), np.transpose(self.output_weights[j, :]))
            output_out_f[j] = self.f(output_net_i)
            output_out_df[j] = self.df(output_net_i)

        return (hidden_out_f, hidden_out_df, output_out_f, output_out_df)

    def backpropagation(self, x, y, model, eta, threshold):
        sqerror = 2 * threshold

        while sqerror > threshold:
            sqerror = 0

            for i in range(0, len(x)):
                x_i = x[i, :] #x are the inputs
                y_i = y[i] #y are the expected results

                (hidden_out_f, hidden_out_df, output_out_f, output_out_df) = self.forward(x_i)
                delta_i = y_i - output_out_f #output is the iteration result
                #delta_i is the difference between expected and actual output of one neuron
                sqerror += np.dot(delta_i, np.transpose(delta_i))

                w_length = self.hidden_size
                output_out_delta = np.multiply(delta_i, output_out_df)
                hidden_out_delta = np.multiply(hidden_out_df, np.dot(output_out_delta,
                                                                     self.output_weights[:, w_length - 1]))


                self.output_weights = self.output_weights + eta * np.transpose(np.dot(np.transpose([np.append(hidden_out_f, [1])]),
                                                                         output_out_delta))
                self.hidden_weights = self.hidden_weights + eta * np.transpose(np.multiply(hidden_out_delta,
                                                                               np.transpose([np.append(x_i,[1])])))

            sqerror = sqerror/len(x)
            print("Avg. Squared error : " + str(sqerror))

def xor_test():
    dataset = np.loadtxt("xor.dat", skiprows=0)
    X = dataset[:, 0:len(dataset[1]) - 1]
    Y = dataset[:, len(dataset[1]) - 1:len(dataset[1])]

    print("Inputs: \n",X)
    print("Expected: \n",Y)

    model = Arch(2, 2, 1)
    model.backpropagation(X, Y, model, 0.1, 0.01)

    for p in range(len(X)):
        x_p = X[p, :]
        y_p = Y[p, :]

        (f_h_net_h_pj, df_h_dnet_h_pj, f_o_net_o_pk, df_o_dnet_o_pk) = model.forward(x_p)
        print("Input: ",x_p)
        print("Expected: ",y_p)
        print("Output: ",f_o_net_o_pk, "\n")


def pca(x):
    print("DEBUG A")
    x = np.transpose(x)
    mean_i = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        mean_i[i] = np.sum(x[i]) / x[i].shape[0]
        #    mean_i = np.array([np.sum(col) / col.shape[0] for col in np.transpose(x)])
        #    mean_i = np.reshape(mean_i, (mean_i.shape[0], 1))
    x = np.transpose(x)
    print(mean_i)
    h = np.ones((x.shape[0], 1))
    b = np.multiply(np.asmatrix(h), np.transpose(mean_i))
    b = x - b
    print(b.shape[0])
    print(b.shape[1])
    print("DEBUG B")
    c = np.asmatrix(b) * np.transpose(np.asmatrix(b))
    d, v = np.linalg.eig(c)
    np.real(d)
    np.real(v)
    print("DEBUG C")
    sorted_eig = list(zip(d, np.transpose(v)))
    sorted_eig.sort(key = lambda e:e[0])
    sorted_eig.reverse()
    np.real(sorted_eig)
    print("DEBUG D")
    print(sorted_eig)
    g = np.zeros(len(sorted_eig))
    for i in range(len(sorted_eig)):
        g[i] = sorted_eig[i][1]
    print(g)


def conversion(num):
    vec = np.zeros(10)
    if num == 0:
        vec[0] = 1
    elif num == 1:
        vec[1] = 1
    elif num == 2:
        vec[2] = 1
    elif num == 3:
        vec[3] = 1
    elif num == 4:
        vec[4] = 1
    elif num == 5:
        vec[5] = 1
    elif num == 6:
        vec[6] = 1
    elif num == 7:
        vec[7] = 1
    elif num == 8:
        vec[8] = 1
    elif num == 9:
        vec[9] = 1
    #print("vec ", vec)
    return vec



def deconversion(vec):
    num = 0
    if vec[0] == 1:
        num = 0
    elif vec[1] == 1:
        num = 1
    elif vec[2] == 1:
        num = 2
    elif vec[3] == 1:
        num = 3
    elif vec[4] == 1:
        num = 4
    elif vec[5] == 1:
        num = 5
    elif vec[6] == 1:
        num = 6
    elif vec[7] == 1:
        num = 7
    elif vec[8] == 1:
        num = 8
    elif vec[9] == 1:
        num = 9
    #print("num ", num)
    return num


def digit_test():
    dataset = np.loadtxt("treino_full.csv", delimiter=',')
    X = np.round(dataset[:, 1:len(dataset[1])]/ 255)
    #pca(X)
    aux = []
    for i in range(len(dataset[:, 0])):
        aux2 = list(conversion(int(dataset[i, 0])))
        aux2 = [int(j) for j in aux2]
        aux += [aux2]
    Y = np.matrix(aux)

    print("X : ", X)
    print("Y : ", Y)

    model = Arch(784, 252, 10)
    model.backpropagation(X, Y, model, 0.1, 0.1)
    #0.3 = 71%
    #0.25 = 77%
    #0.2 = 79%
    # 100 hl

    #0.3 = 77%
    #0.2 = 81%
    #0.1 = 87.7%
    # 252 hl, calculado numero otimizado de neuronios

    #todo: fazer uma funcao pro eta mudar de acordo com a convergencia

    b = 0
    test = np.round(np.loadtxt("teste_full.csv", delimiter=',') / 255)
    test_result = np.round(np.loadtxt("teste_full.csv", delimiter=','))
    test_result = test_result[:, 0]
    test_fw = test[:, 1:test.shape[0]]
    for i in range(test.shape[0]):
        (_, _, net_i, _) = model.forward(test_fw[i])
        #print("net_i ", net_i)
        num = deconversion(np.round(net_i))
        #print(num, " -> ", test_result[i])
        if np.round(num) != test_result[i]:
            b +=1

    #a = np.sum(test_result)
    print("Casos de teste : ", test.shape[0])
    print("Numero de erros : ", b)
    print("Accuracy : ", 100 - ((b/test.shape[0]) * 100))


def gender_num(gender):
    if gender == "male":
        return 1.0
    else:
        return 0.0


def name_num(name):
    name = str(name)
    if ("Miss" in name or
        "Mlle" in name or
        "Ms" in name or
        "Mrs" in name or
        "Dona" in name or
        "Lady" in name or
        "the Countess" in name or
        "Mme" in name):
            return 0.0
    elif "Mr" in name:
        return 0.35
    elif ("Dr" in name or
        "Master" in name):
        return 0.7
    elif (
	 "Rev" in name or
	 "Capt" in name or
	 "Col" in name or
	 "Don" in name or
	 "Major" in name or
	 "Sir" in name or
	 "Jonkheer" in name):
        return  1.0


def ticket_num(ticket):
    return 1.0


def cabin_num(cabin):
    return 1.0


def port_num(port):
    if port == 'C':
        return 1.0
    if port == 'Q':
        return 0.5
    if port == 'S':
        return 0.0

def titanic():
    dataset = np.loadtxt("t_treino.csv", delimiter=',',
                         converters={3 : name_num,
                                     4 : gender_num,
                                     8 : ticket_num,
                                     10 : cabin_num,
                                     11 : port_num})
    Y = dataset[:, 1] #survived ? 1 = yes : 0 = no

    dataset[:,2] = (dataset[:, 2] / 3) #normalization of pclass

    age = [(x / 100 if x != 0 else np.round(np.mean(dataset[:,5])) / 100) for x in dataset[:, 5]]
    dataset[:, 5] = age #filling missing info and normalizing

    fare = [(x / np.max(dataset[:,9]) if x != 0 else np.round(np.mean(dataset[:, 9])) / np.max(dataset[:,9])) for x in dataset[:, 5]]
    dataset[:,9] = fare

    print(fare)

    # print(Y)
    # print(dataset[:,2])
    # print(dataset[:,5])
    # print(dataset[:,3])
    X = dataset[:, 2:11]

    model = Arch(X.shape[1], 8, 1)
    model.backpropagation(X, Y, model, 0.01, 0.2)



    b = 0
    test = np.loadtxt("t_teste.csv", skiprows=1, delimiter=',',
                               converters={3 : name_num,
                                     4 : gender_num,
                                     8 : ticket_num,
                                     10 : cabin_num,
                                     11 : port_num})


    age = [(x / 100 if x != 0 else np.round(np.mean(test[:, 5])) / 100) for x in test[:, 5]]
    test[:, 5] = age  # filling missing info and normalizing
    test[:, 2] = (test[:, 2] / 3)

    fare = [x / np.max(test[:, 9]) for x in test[:, 9]]
    test[:, 9] = fare

    test_result = test[:, 1]
    test_fw = test[:, 2:11]
    for i in range(test.shape[0]):
        (_, _, net_i, _) = model.forward(test_fw[i])
        print("net_i ", net_i)
        num = np.round(net_i)
        print(num, " -> ", test_result[i])
        if np.round(num) != test_result[i]:
            b += 1

    # a = np.sum(test_result)
    print("Casos de teste : ", test.shape[0])
    print("Numero de erros : ", b)
    print("Accuracy : ", 100 - ((b / test.shape[0]) * 100))



def main():
    titanic()

if __name__ == "__main__":
    main()