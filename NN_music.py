import numpy as np
import scipy as sp
import sys, random, math, csv

# run two different implementations

# one through incremental training (online learning) (stochastic GD)
# where we constantly update the weights and train the neuralnet
# over the updated weights / network as we go through the training set

# one through simple gradient descent using batch learning
# where we run through the entire training set and save the outputs
# with the initial weights, and then update the weights at the final
# run with a saved vector of the outputs

def flatten(l):
    return (flatten(l[0]) +
    (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l])

def normalize_train_set(train_set):
    normalized_train_s = []
    for instance in train_set:
        normalized_inst = []
        for count, val in enumerate(instance):
            if val == "yes": normalized_inst.append(1.0)
            if val == "no": normalized_inst.append(0.0)
            elif count == 0: normalized_inst.append((float(val) - 1900) / 100.0)
            elif count == 1: normalized_inst.append(float(val) / 7)
        normalized_train_s.append(normalized_inst)
    return normalized_train_s

def get_gd_input_weights():
    input_weights = []
    for input_neuron_i in xrange(5):
        # for each input (+ bias = 5 input neurons)
        input_neuron = []
        for neuron_i in xrange(3):
            # pointing to each neuron in hidden layer
            if input_neuron_i == 0:
                input_neuron.append(1.0)
            else:
                init_weight = random.uniform(0.0, 0.0001)
                input_neuron.append(init_weight)
        input_weights.append(input_neuron)
        input_neuron = []
    return input_weights

def get_gd_hidden_weights():
    hidden_weights = []
    for hidden_neuron_i in xrange(4):
        hidden_weights_perneuron = []
        for weight_i in xrange(1):
            init_weight = random.uniform(0.0, 0.0001)
            hidden_weights_perneuron.append(init_weight)
        hidden_weights.append(hidden_weights_perneuron)
        hidden_weights_perneuron = []
    return hidden_weights

def apply_sigmoid(val):
    # val type float
    return 1.0 / (1.0 + math.exp(-val))

def sigmoid(array):
    # array type numpy array
    return 1.0 / (1.0 + np.exp(-array))

def compute_loss(output_matrix, label_matrix):
    # output matrix type numpy matrix
    # label matrix just a 2d array
    # compute the loss function E = 1\2 \sum(t-o)^2
    loss_vector = []
    for count, label in enumerate(label_matrix):
        label_val = -1
        if label == "yes":
            label_val = 1.0
        elif label == "no":
            label_val = 0.0
        loss_val = (label_val - output_matrix[count,0])**2
        loss_vector.append(loss_val)
    return sum(loss_vector) / 2.0

def convert_to_t(train_set_labels):
    # convert yes values to 1.0 and no values to 0.0
    output_t = []
    n_train_set_labels = flatten(train_set_labels)
    for inst in n_train_set_labels:
        if inst == "yes":
            output_t.append([1.0])
        elif inst == "no":
            output_t.append([0.0])
    return output_t

def gd_update_weights(output_matrix, train_set_labels, input_weight_m,
                    hidden_layer_m, x_0_matrix, x_1_matrix, z_0_matrix):
    # now we will update the weight matrices used in gradient descent to
    # using the mathematical formulas
    #print "input_weight_m :", input_weight_m
    #print "hidden_layer_m :", hidden_layer_m
    x_1_t = np.transpose(x_1_matrix)
    learn_r = 0.05
    label_matrix = np.asmatrix(convert_to_t(train_set_labels))
    ones = np.ones(output_matrix.shape)
    #print "ones.shape :", ones.shape
    deriv_sigmoid = np.multiply(output_matrix, np.subtract(ones, output_matrix))
    #print "deriv_sigmoid.shape :", deriv_sigmoid.shape
    t_minus_0 = np.subtract(label_matrix, output_matrix)
    #print "t_minus_0.shape :", t_minus_0.shape
    deriv_w = np.multiply(t_minus_0, deriv_sigmoid)
    delta_w1 = np.dot(x_1_t, deriv_w)
    #print "delta_w1.shape :", delta_w1.shape
    #print "delta_w1 :", delta_w1

    #print "updated hidden_layer_m :", hidden_layer_m
    x_0_t = np.transpose(x_0_matrix)
    #print "x_0_t.shape :", x_0_t.shape
    #deriv_E_z1 = np.dot(x_0_t, deriv_w)
    w_1_t = np.transpose(hidden_layer_m)[0,1:]
    #print "w_1_t.shape :", w_1_t.shape
    #print "mean_z_0_matrix :", mean_z_0_matrix
    sig = np.vectorize(apply_sigmoid)
    sig_z_0 = sig(z_0_matrix)
    #print "sig_z_0.shape :", sig_z_0.shape
    ones_v = np.ones((len(output_matrix),3))
    #print "ones_v.shape :", ones_v.shape
    deriv_from_w = np.multiply(sig_z_0,np.subtract(ones_v,sig_z_0))
    #print "deriv_from_w.shape :", deriv_from_w.shape
    temp = np.dot(deriv_w, w_1_t)
    #print "temp.shape :", temp.shape
    temp2 = np.multiply(temp, deriv_from_w)
    #print "temp2.shape :", temp2.shape
    delta_w0 = np.dot(x_0_t,temp2)
    #print "delta_w0.shape :", delta_w0.shape
    # now we update the weights

    hidden_layer_m = np.add(hidden_layer_m,np.multiply(learn_r,delta_w1))
    input_weight_m = np.add(input_weight_m, np.multiply(learn_r,delta_w0))
    return input_weight_m, hidden_layer_m

def gradient_descent(train_set, train_set_labels):
    # in simple GD, we initialize our weights to be very small random numbers
    input_weight_m = np.asmatrix(get_gd_input_weights())
    hidden_layer_m = np.asmatrix(get_gd_hidden_weights())
    label_array = flatten(train_set_labels)
    input_layer_neurons = [1.0]
    for epoch in xrange(5000):
        ones_col = np.ones((len(train_set),1))
        train_s = np.append(ones_col,
                            np.asmatrix(normalize_train_set(train_set)), axis=1)
        z0 = np.dot(train_s, input_weight_m)
        #print "z0 :", z0
        A0 = np.append(ones_col, sigmoid(z0), axis=1)
        #print "A0 :", A0
        z1 = np.dot(A0,hidden_layer_m)
        A1 = sigmoid(z1)
        # compute the loss value
        total_loss = compute_loss(A1, label_array)
        print total_loss
        #then update the weights accordingly through batch GD
        input_weight_m, hidden_layer_m = gd_update_weights(A1,
                                                        train_set_labels,
                                                        input_weight_m,
                                                        hidden_layer_m,
                                                        train_s,
                                                        A0,
                                                        z0)
    print total_loss
    return (input_weight_m, hidden_layer_m, total_loss)


def get_sgd_input_weights():
    file = sys.argv[-2]
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        weights_as_list = list(reader)
    return weights_as_list

def get_sgd_hidden_weights():
    file = sys.argv[-1]
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        weights_as_list = list(reader)
    return weights_as_list

def stochastic_gd(train_set, train_set_labels):
    # we are given the architecture and the inital weights for stochastic GD
    learn_r = 0.4 # our learning rate
    input_weight_m = np.asmatrix(get_sgd_input_weights()).astype(np.float)
    hidden_layer_m = np.asmatrix(get_sgd_hidden_weights()).astype(np.float)
    # fetch the weights that we are initially given from the file
    train_s = normalize_train_set(train_set) # normalize instances in train set
    input_layer_neurons = [1] # init input layer neurons with 1 for bias neuron
    # for stochastic gd, we update the weights everytime
    label_array = flatten(train_set_labels)
    for epoch in xrange(15):
        final_output_matrix = []
        for count, instance in enumerate(train_s):
            #print "instance :", instance
            hidden_layer_inputs = [1]
            z_0_vec = []
            label_val = -1
            n_instance = input_layer_neurons + instance
            input_layer_neurons_m = np.asmatrix(n_instance) #x_0
            hidden_neuron_inputs = np.dot(input_layer_neurons_m, input_weight_m)
            #print "hidden_neuron_inputs :", hidden_neuron_inputs
            for neuron_i in xrange(3):
                # for neuron in hidden layer (3 neurons + 1 bias input)
                valbefore_sigmoid = hidden_neuron_inputs[0, neuron_i]
                #print "valbefore_sigmoid :", valbefore_sigmoid
                hidden_layer_inputs.append(apply_sigmoid(valbefore_sigmoid))
                z_0_vec.append(apply_sigmoid(valbefore_sigmoid))
            x_1_t = np.transpose(np.asmatrix(hidden_layer_inputs))
            z_0_vec = np.asmatrix(z_0_vec)
            if train_set_labels[count][0] == "yes":
                label_val = 1.0
            elif train_set_labels[count][0] == "no":
                label_val = 0.0
            output = apply_sigmoid(np.dot(hidden_layer_inputs,
                                                    hidden_layer_m))
            final_output_matrix.append([output])

            # now update the weights
            deriv_o_to_x = (label_val-output)*output*(1-output)
            delta_w1 = np.multiply(x_1_t, deriv_o_to_x)
            x_0_t = np.transpose(input_layer_neurons_m)
            w_1_t = np.transpose(hidden_layer_m)[0,1:]
            three_ones = np.ones((1,3))
            inv_sig_z = np.subtract(three_ones,z_0_vec)
            deriv_from_z = np.multiply(z_0_vec,inv_sig_z)

            second_p = np.multiply(np.dot(deriv_o_to_x,w_1_t),deriv_from_z)
            delta_w0 = np.dot(x_0_t, second_p)
            # apply changes in weights

            #print "hidden_layer_m before change:", hidden_layer_m
            hidden_layer_m = np.add(hidden_layer_m,np.multiply(learn_r,delta_w1))
            #print "hidden_layer_m after change :", hidden_layer_m

            #print "input_weight_m before change :", input_weight_m
            input_weight_m = np.add(input_weight_m,np.multiply(learn_r,delta_w0))
            #print "input_weight_m after change :", input_weight_m

        output_matrix = np.asmatrix(final_output_matrix)
        total_loss = compute_loss(output_matrix, label_array)
        print total_loss
    return (input_weight_m, hidden_layer_m, total_loss)


def print_predictions(nn_input_weight_m, nn_hidden_layer_m, dev_set):
    # take the weights of the neural net and calculate the final value
    # of each instance in the development set (test set)
    norm_dev_set = normalize_train_set(dev_set)
    bias_term = [1]
    for instance in norm_dev_set:
        hidden_layer_inputs = [1]
        instance_input_m = np.asmatrix(bias_term + instance)
        hidden_neuron_inputs = np.dot(instance_input_m, nn_input_weight_m)
        for i in xrange(3):
            # 3 hidden layer neurons
            hidden_layer_inputs.append(apply_sigmoid(hidden_neuron_inputs[0,i]))
        output = apply_sigmoid(np.dot(hidden_layer_inputs,nn_hidden_layer_m))
        if output > 0.5: print "yes"
        else: print "no"
    return None

def predict(gd_neuralnet, st_gd_neuralnet, dev_set):
    # make predictions with our new weights on a tree
    if gd_neuralnet[2] < st_gd_neuralnet[2]:
        # training error for gradient descent neural net lower than stochastic
        # gradient descent neural net
        input_weight_m, hidden_layer_m, _ = gd_neuralnet
        print_predictions(input_weight_m, hidden_layer_m, dev_set)
        return None
    else:
        # training error for stochastic gradient descent neural net lower than
        # gradient descent neural net
        input_weight_m, hidden_layer_m, _ = st_gd_neuralnet
        print_predictions(input_weight_m, hidden_layer_m, dev_set)
        return None


def clean_data_file(file):
    final_file = []
    for instance in open(file, 'r').readlines():
        cleaned_inst = (instance.strip("\r\n")).split(",")
        final_file.append(cleaned_inst)
    return final_file[1:]

def clean_label_file(file):
    final_file = []
    for instance in open(file, 'r').readlines():
        clean_inst = (instance.strip("\r\n")).split(" ")
        final_file.append(clean_inst)
    return final_file

def main():
    seed = random.seed(343495)
    train_file = sys.argv[-5]
    train_file_labels = sys.argv[-4]
    dev_file = sys.argv[-3]

    train_set_arr = clean_data_file(train_file)
    train_set_labels = clean_label_file(train_file_labels)
    dev_set = clean_data_file(dev_file)
    gd_neuralnet = gradient_descent(train_set_arr, train_set_labels)
    print "GRADIENT DESCENT TRAINING COMPLETED!"
    st_gd_neuralnet = stochastic_gd(train_set_arr, train_set_labels)
    print "STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED! NOW PREDICTING."
    predictions = predict(gd_neuralnet, st_gd_neuralnet, dev_set)
    return None

if __name__ == "__main__":
    main()
