import numpy as np
import h5py
import pylab
import matplotlib.pyplot as plt
import sys
import time
import pypet as pypet
import matplotlib
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.mlab import griddata
import tqdm
import os.path
from collections import deque
from collections import namedtuple
import copy
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from fanova import fANOVA
import fanova.visualizer
import pickle

sigmoid_table = np.array([241, 257, 273, 291, 310, 330, 351, 374, 398, 
424, 452, 481, 512, 545, 580, 618, 658, 
700, 746, 794, 845, 900, 958, 1020, 1086, 
1156, 1231, 1310, 1395, 1485, 1581, 1683, 1792, 
1908, 2031, 2163, 2302, 2451, 2610, 2778, 2958, 
3149, 3353, 3569, 3800, 4046, 4307, 4586, 4882, 
5198, 5534, 5891, 6272, 6677, 7109, 7568, 8057, 
8578, 9133, 9723, 10351, 11020, 11732, 12490, 13298, 
14157, 15072, 16046, 17083, 18187, 19362, 20614, 21946, 
23364, 24874, 26482, 28193, 30015, 31955, 34020, 36218, 
38559, 41051, 43703, 46528, 49535, 52736, 56144, 59772, 
63634, 67747, 72125, 76786, 81748, 87030, 92655, 98642, 
105017, 111803, 119028, 126719, 134908, 143626, 152907, 162788, 
173307, 184506, 196429, 209122, 222635, 237021, 252337, 268642, 
286000, 304481, 324155, 345100, 367398, 391136, 416409, 443313, 
471956, 502449, 534911, 569471, 606262, 645430, 687127, 731518, 
778775, 829083, 882641, 939656, 1000353, 1064968, 1133755, 1206983, 
1284937, 1367922, 1456264, 1550307, 1650418, 1756988, 1870434, 1991198, 
2119752, 2256596, 2402265, 2557326, 2722382, 2898078, 3085096, 3284165, 
3496057, 3721598, 3961661, 4217179, 4489143, 4778607, 5086691, 5414588, 
5763565, 6134970, 6530234, 6950883, 7398534, 7874908, 8381835, 8921258, 
9495242, 10105982, 10755806, 11447191, 12182766, 12965319, 13797815, 14683398, 
15625403, 16627371, 17693056, 18826440, 20031744, 21313440, 22676270, 24125253, 
25665706, 27303256, 29043856, 30893803, 32859755, 34948747, 37168207, 39525980, 
42030340, 44690011, 47514188, 50512551, 53695286, 57073104, 60657254, 64459547, 
68492362, 72768671, 77302042, 82106659, 87197322, 92589461, 98299132, 104343019, 
110738432, 117503289, 124656109, 132215986, 140202562, 148635989, 157536888, 166926294, 
176825593, 187256449, 198240719, 209800353, 221957289, 234733325, 248149986, 262228372, 
276988992, 292451586, 308634936, 325556655, 343232976, 361678522, 380906069, 400926304, 
421747581, 443375670, 465813513, 489060985, 513114667, 537967630, 563609241, 590024999, 
617196386, 645100769, 673711332, 702997045, 732922695, 763448954, 794532500, 826126195, 
858179312, 890637815, 923444686, 956540298, 989862833, 1023348727, 1056933148, 1090550498, 
1124134919, 1157620813, 1190943348, 1224038960, 1256845831, 1289304334, 1321357451, 1352951146, 
1384034692, 1414560951, 1444486601, 1473772314, 1502382877, 1530287260, 1557458647, 1583874405, 
1609516016, 1634368979, 1658422661, 1681670133, 1704107976, 1725736065, 1746557342, 1766577577, 
1785805124, 1804250670, 1821926991, 1838848710, 1855032060, 1870494654, 1885255274, 1899333660, 
1912750321, 1925526357, 1937683293, 1949242927, 1960227197, 1970658053, 1980557352, 1989946758, 
1998847657, 2007281084, 2015267660, 2022827537, 2029980357, 2036745214, 2043140627, 2049184514, 
2054894185, 2060286324, 2065376987, 2070181604, 2074714975, 2078991284, 2083024099, 2086826392, 
2090410542, 2093788360, 2096971095, 2099969458, 2102793635, 2105453306, 2107957666, 2110315439, 
2112534899, 2114623891, 2116589843, 2118439790, 2120180390, 2121817940, 2123358393, 2124807376, 
2126170206, 2127451902, 2128657206, 2129790590, 2130856275, 2131858243, 2132800248, 2133685831, 
2134518327, 2135300880, 2136036455, 2136727840, 2137377664, 2137988404, 2138562388, 2139101811, 
2139608738, 2140085112, 2140532763, 2140953412, 2141348676, 2141720081, 2142069058, 2142396955, 
2142705039, 2142994503, 2143266467, 2143521985, 2143762048, 2143987589, 2144199481, 2144398550, 
2144585568, 2144761264, 2144926320, 2145081381, 2145227050, 2145363894, 2145492448, 2145613212, 
2145726658, 2145833228, 2145933339, 2146027382, 2146115724, 2146198709, 2146276663, 2146349891, 
2146418678, 2146483293, 2146543990, 2146601005, 2146654563, 2146704871, 2146752128, 2146796519, 
2146838216, 2146877384, 2146914175, 2146948735, 2146981197, 2147011690, 2147040333, 2147067237, 
2147092510, 2147116248, 2147138546, 2147159491, 2147179165, 2147197646, 2147215004, 2147231309, 
2147246625, 2147261011, 2147274524, 2147287217, 2147299140, 2147310339, 2147320858, 2147330739, 
2147340020, 2147348738, 2147356927, 2147364618, 2147371843, 2147378629, 2147385004, 2147390991, 
2147396616, 2147401898, 2147406860, 2147411521, 2147415899, 2147420012, 2147423874, 2147427502, 
2147430910, 2147434111, 2147437118, 2147439943, 2147442595, 2147445087, 2147447428, 2147449626, 
2147451691, 2147453631, 2147455453, 2147457164, 2147458772, 2147460282, 2147461700, 2147463032, 
2147464284, 2147465459, 2147466563, 2147467600, 2147468574, 2147469489, 2147470348, 2147471156, 
2147471914, 2147472626, 2147473295, 2147473923, 2147474513, 2147475068, 2147475589, 2147476078, 
2147476537, 2147476969, 2147477374, 2147477755, 2147478112, 2147478448, 2147478764, 2147479060, 
2147479339, 2147479600, 2147479846, 2147480077, 2147480293, 2147480497, 2147480688, 2147480868, 
2147481036, 2147481195, 2147481344, 2147481483, 2147481615, 2147481738, 2147481854, 2147481963, 
2147482065, 2147482161, 2147482251, 2147482336, 2147482415, 2147482490, 2147482560, 2147482626, 
2147482688, 2147482746, 2147482801, 2147482852, 2147482900, 2147482946, 2147482988, 2147483028, 
2147483066, 2147483101, 2147483134, 2147483165, 2147483194, 2147483222, 2147483248, 2147483272, 
2147483295, 2147483316, 2147483336, 2147483355, 2147483373, 2147483389, 2147483405, 0])

E_SIGMOID = 9
INT_MAX = 0x7fffffff
INT_MIN = -INT_MAX

def scale_parameters(param):

    new_param = param - 0.5
    new_param = new_param * 2**30
    new_param = np.clip(new_param, - (2**31), 2**31 - 1)
    
    return new_param

def sigmoid_unscaled(x):
    x = int(x) >> 1;
    x += 1 << 30;
    return sigmoid_table[int(x) >> (31 - E_SIGMOID)] >> 5

def ANN_forward_unscaled(w1, b1, w2, input_values, learning_rate):

    hidden_activations = np.zeros(w1.shape[1])
    output_activations = 0
    
    for hidden_num in range(w1.shape[1]):
        input_sum = 0
        for input_num, input_val in enumerate(input_values):
            input_sum += int(w1[input_num, hidden_num] * input_val) >> 26
        input_sum += b1[hidden_num]
        
        if (input_sum < INT_MIN):
            input_sum = INT_MIN
        elif (input_sum >= INT_MAX):
            input_sum = INT_MAX
                
        hidden_activations[hidden_num] = sigmoid_unscaled(input_sum)
        output_activations += int(w2[hidden_num][0] * hidden_activations[hidden_num]) >> 26
              
    if (output_activations < INT_MIN):
        output_activations = INT_MIN
    elif (output_activations >= INT_MAX):
        output_activations = INT_MAX
            
    output_activations_wo_lr = output_activations
    output_activations = int(output_activations * learning_rate) >> 32;
    
    output_activation_scaled = output_activations >> 27
    output_activation_scaled += 32
    
    return hidden_activations, output_activations, output_activations_wo_lr #output_activation_scaled
    
def ANN_forward_scaled(w1, b1, w2, input_values, learning_rate):
    
    hidden_activations = np.zeros(len(w1[0]))
    output_activations = 0
    
    for hidden_num in range(len(w1[0])):
        input_sum = 0
        for input_num, input_val in enumerate(input_values):
            input_sum += int(w1[input_num][hidden_num](input_val)) >> 26
        input_sum += b1[hidden_num]
        
        if (input_sum < INT_MIN):
            input_sum = INT_MIN
        elif (input_sum >= INT_MAX):
            input_sum = INT_MAX
                
        hidden_activations[hidden_num] = sigmoid_unscaled(input_sum)
        output_activations += int(w2[hidden_num][0] * hidden_activations[hidden_num]) >> 26
              
    if (output_activations < INT_MIN):
        output_activations = INT_MIN
    elif (output_activations >= INT_MAX):
        output_activations = INT_MAX
            
    output_activations_wo_lr = output_activations
    output_activations = int(output_activations * learning_rate) >> 32;
    
    output_activation_scaled = output_activations >> 27
    output_activation_scaled += 32
    
    return hidden_activations, output_activations, output_activations_wo_lr #output_activation_scaled

def GetPopSize(traj):

    #Check if this is an ES run and mirror sampling is turned on
    factor = 1
    
    try:        
        if traj.parameters.mirrored_sampling_enabled:
            factor = 2
    except:
        pass
    
    try:
        return factor * traj.parameters.pop_size
    except:
        try:
            return factor * traj.parameters.max_pop_size
        except:
            try:
                return factor * traj.parameters.n_random_steps
            except:
                return factor * traj.parameters.n_parallel_runs

def GetIterations(traj):

    return traj.parameters.n_iteration

def PrepareWeightFunction(a, b, w, inp_value):

    return lambda inp_value: (inp_value * (copy.deepcopy(b) - copy.deepcopy(a)) + copy.deepcopy(a)) *copy.deepcopy(w)


if __name__ == '__main__':

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 28}

    matplotlib.rc('font', **font)

    #Number of best performing runs to investigate
    nBest = 10
    n_samples = 2000
    LRP_rule = 'beta1'
    currentlyBest = -np.inf
    bestPerforming = deque(nBest * (0, 0), nBest)
    n_input_neurons = 5
    n_hidden_neurons = 7
    n_output_neurons = 1
    ann_input_tuple = namedtuple('ann_input', ('step', 'rewards', 'action', 'weight', 'oweight'))
    ann_input = ann_input_tuple(step = [0, 100 << 18],                                  
                                rewards = [-(1 << 26), 1 << 26],
                                action = [-(1 << 26), 1 << 26],
                                weight = [0, 1 << 26],
                                oweight = [0, 1 << 26])
    #ann_weights = [-(1<<31), (1<<31) - 1]
    ann_weights = scale_parameters(np.array([0., 1.]))

    f = h5py.File(sys.argv[1])
    
    for member in f[sys.argv[2]]['results']:
        if 'run' in member:
            for run in f[sys.argv[2]]['results'][member]:
                fitness = f[sys.argv[2]]['results'][member][run]['fitness']['fitness'][0]
                annWeights = f[sys.argv[2]]['results'][member][run]['individual']['individual'][0][0]
                learningRate = f[sys.argv[2]]['results'][member][run]['individual']['individual'][0][1]
                banditProbs = f[sys.argv[2]]['results'][member][run]['bandit_probabilities']['bandit_probabilities']

                if fitness > currentlyBest:
                    currentlyBest = fitness
                    bestPerforming.append((fitness, annWeights, learningRate, run, banditProbs))

    #Best performing one bestPerforming[-1]

    #Analyze the weights, while also considering the different input strength
    #Parameters are representing: 1.) Weights input to hidden
    #                             2.) Bias for hidden neurons
    #                             3.) Weights hidden to output
    #Currently only consider the best performing one
    size_w1 = n_input_neurons * n_hidden_neurons
    size_w2 = n_hidden_neurons * n_output_neurons
    #w1 = np.array(bestPerforming[-1][1][:size_w1]).reshape((n_input_neurons, n_hidden_neurons))
    #b1 = np.array(bestPerforming[-1][1][size_w1:(size_w1 + n_hidden_neurons)])
    #w2 = np.array(bestPerforming[-1][1][(size_w1 + n_hidden_neurons):(size_w1 + n_hidden_neurons + size_w2)]).reshape((n_hidden_neurons, n_output_neurons))
    #lr = np.array(bestPerforming[-1][2])

    '''THIS CAN BE REMOVED'''
    '''Overwrite the parameters from the values Franz provided'''
    ann_parameters = np.array([0.51502079, 0.49740018, 0.20023772, 0.28295012, 0.20839904, 0.20295959,
 0.58402285, 0.1,        0.18168052, 0.13090999, 0.78121004, 0.72488774,
 0.24002499, 0.68832076, 0.10375894, 0.21000279, 0.2803778,  0.83113176,
 0.18101726, 0.36358029, 0.60059928, 0.9,        0.37448422, 0.40058301,
 0.27326555, 0.43463887, 0.66596137, 0.13436804, 0.72311331, 0.81158696,
 0.5622522,  0.1461575,  0.48496516, 0.72801292, 0.32241905, 0.82438525,
 0.78395973, 0.78074635, 0.41544295, 0.86644921, 0.35569846, 0.47646374,
 0.11145061, 0.41872347, 0.31664511, 0.87257462, 0.88002756, 0.23176495,
 0.79220276])
    w1 = ann_parameters[:size_w1].reshape((n_input_neurons, n_hidden_neurons))
    b1 = ann_parameters[size_w1:(size_w1 + n_hidden_neurons)]
    w2 = ann_parameters[(size_w1 + n_hidden_neurons):(size_w1 + n_hidden_neurons + size_w2)].reshape((n_hidden_neurons, n_output_neurons))
    lr = np.array([0.8141041966914108])

    import ipdb
    #ipdb.set_trace()
    
    #Scale the weights and the learning rate as Franz did it in his script
    w1 = scale_parameters(w1)
    b1 = scale_parameters(b1)
    w2 = scale_parameters(w2)
    lr = lr * (2**32 - 1)

    #Compute the normalized weights from the input layer to the hidden layer
    #Normalization is done such that the product of input and weight is between [0, 1]
    normalized_w1 = []
    for inp in range(n_input_neurons):
        temp = []
        for hidd in range(n_hidden_neurons):
            temp.append(PrepareWeightFunction(copy.deepcopy(ann_input[copy.deepcopy(inp)][0]),
                                              copy.deepcopy(ann_input[copy.deepcopy(inp)][1]),
                                              copy.deepcopy(w1[copy.deepcopy(inp)][copy.deepcopy(hidd)]),
                                              1))
        normalized_w1.append(temp)
                                        
    #Initial importance measure: I_i = Sum_j ((w_oj + b_j) * w_ji)
    '''
    importance_vector = np.zeros(n_input_neurons)

    for i in range(n_input_neurons):
        importance = 0

        for j in range(n_hidden_neurons):
            importance += (w2[j, 0] + b1[j]) * w1[i, j]

        importance_vector[i] = importance

    print(importance_vector)
    '''
    
    #Compute the relevance scores according to LRP
    '''
    inputs = np.zeros((n_input_neurons, n_samples))
    inputs[0, :] = np.random.randint(ann_input[0][0], ann_input[0][1], n_samples)
    inputs[1, :] = (np.random.randint(0, 2, n_samples) * (ann_input[1][1] - ann_input[1][0])) + ann_input[1][0]
    inputs[2, :] = (np.random.randint(0, 2, n_samples) * (ann_input[2][1] - ann_input[2][0])) + ann_input[2][0]
    inputs[3, :] = np.random.randint(ann_input[3][0], ann_input[3][1] + 1, n_samples)
    inputs[4, :] = np.random.randint(ann_input[4][0], ann_input[4][1] + 1, n_samples)
    '''

    inputs = np.random.random((n_input_neurons, n_samples))
    inputs[1, :] = np.random.randint(0, 2, n_samples)
    inputs[2, :] = np.random.randint(0, 2, n_samples)

    import ipdb
    #ipdb.set_trace()
    '''Read the inputs from a pre-generated npz file'''
    #temp = np.load('ann_input.npz')
    #outputs = np.copy(temp['arr_1'])    
    #temp = temp['arr_0']
    #inputs = np.zeros((np.shape(temp)))

    '''THIS CAN BE REMOVED'''
    '''Read the input files from a pickle file, provided by Franz'''
    f = pickle.load(open('/home/thomas/Downloads/data.pkl', 'rb'))
    inputs = f['inputs']
    #inputs = inputs[:2000, :]
    #outputs = np.zeros(inputs.shape[0])
    outputs = f['outputs']

    '''Scale the inputs'''
    inputs[:, 0] = inputs[:, 0] / float(100. * 2**18)
    inputs[:, 1] = (inputs[:, 1] + (1 << 26)) / (2 * (1 << 26)) 
    inputs[:, 2] = (inputs[:, 2] + (1 << 26)) / (2 * (1 << 26))
    inputs[:, 3] = inputs[:, 3] / (1 << 26)#np.max(inputs[:, 3])#float(63. * 2**20)
    inputs[:, 4] = inputs[:, 4] / (1 << 26)#np.max(inputs[:, 4])#float(63. * 2**20)
    inputs = np.transpose(inputs)
    
    import ipdb
    #ipdb.set_trace()

    #it = tqdm.tqdm(range(inputs.shape[1]))
    #for sample in it:
    #    '''THIS CAN BE REMOVED'''
    #    '''For the inputs from Franz, consider the unscaled version of the ANN'''
    #    _, _, outp_unscaled = ANN_forward_scaled(normalized_w1, b1, w2, inputs[:, sample], lr)            
    #    outputs[sample] = outp_unscaled
        
    if LRP_rule == 'beta':
        
        beta = 1.

        '''Compute the forward path. This takes the unscaled output to be either negative or positive, indicating the change of the synaptic weight'''
        relevances = np.zeros((n_samples, n_input_neurons))
        outputs = np.zeros(n_samples)
        it = tqdm.tqdm(range(n_samples))
        for sample in it:
            hidden_act, _, output = ANN_forward_scaled(normalized_w1, b1, w2, inputs[:, sample], lr)
            outputs[sample] = output
            
            z = []
            for hidden_neuron in range(n_hidden_neurons):
                z.append(hidden_act[hidden_neuron] * w2[hidden_neuron][0])

            z = np.array(z)
            zn = np.sum(np.clip(z, -np.inf, 0.))
            zp = np.sum(np.clip(z, 0., np.inf))

            '''Compute the hidden relevances'''
            relevances_hidden = np.zeros(n_hidden_neurons)
            for hidden_neuron in range(n_hidden_neurons):
                relevances_hidden[hidden_neuron] = ((1 + beta) * ((np.clip(z[hidden_neuron], 0., np.inf)) / (zp)) - beta * ((np.clip(z[hidden_neuron], -np.inf, 0.)) / (zn))) * output

            z = np.zeros((n_input_neurons, n_hidden_neurons))
            for input_neuron in range(n_input_neurons):
                for hidden_neuron in range(n_hidden_neurons):
                    z[input_neuron][hidden_neuron] = normalized_w1[input_neuron][hidden_neuron](inputs[input_neuron, sample])

            '''Compute the input relevances'''
            #ipdb.set_trace()
            relevances_input = []
            for input_neuron in range(n_input_neurons):
                current_relevance = 0                
                for hidden_neuron in range(n_hidden_neurons):
                    current_relevance += ((1 + beta) * ((np.clip(z[input_neuron, hidden_neuron], 0., np.inf)) / (np.finfo(np.float32).eps + np.sum(np.clip(z[:, hidden_neuron], 0., np.inf)))) - beta * ((np.clip(z[input_neuron, hidden_neuron], -np.inf, 0.)) / (-np.finfo(np.float32).eps + np.sum(np.clip(z[:, hidden_neuron], -np.inf, 0.))))) * relevances_hidden[hidden_neuron]
                    
                    if np.isnan(current_relevance):
                        #ipdb.set_trace()
                        pass

                if np.sum(np.isnan(current_relevance)) > 0.:
                    import ipdb
                    #ipdb.set_trace()
                relevances_input.append(current_relevance)
            
            relevances[sample, :] = np.array(relevances_input)

    '''Try the thing with fANOVA'''
    import ipdb
    #ipdb.set_trace()
    X = np.transpose(inputs)
    Y = np.transpose(np.array(outputs))
    
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter("0", 0., 1., default_value=0.5))
    cs.add_hyperparameter(CategoricalHyperparameter("1", [0., 1.], default_value=1.))
    cs.add_hyperparameter(CategoricalHyperparameter("2", [0., 1.], default_value=1.))
    cs.add_hyperparameter(UniformFloatHyperparameter("3", 0., 1., default_value=0.5))
    cs.add_hyperparameter(UniformFloatHyperparameter("4", 0., 1., default_value=0.5))
    param = cs.get_hyperparameters()

    f = fANOVA(X, Y, config_space=cs, n_trees=50)

    print('Total importances')
    res = f.quantify_importance((0,))
    print(res[(0,)]['total importance'])
    res = f.quantify_importance((1,))
    print(res[(1,)]['total importance'])
    res = f.quantify_importance((2,))
    print(res[(2,)]['total importance'])
    res = f.quantify_importance((3,))
    print(res[(3,)]['total importance'])
    res = f.quantify_importance((4,))
    print(res[(4,)]['total importance'])
    
    print('Pair-wise importance')
    print(f.get_most_important_pairwise_marginals((0, 1, 2, 3, 4)))

    import fanova.visualizer
    vis = fanova.visualizer.Visualizer(f, cs, "/home/thomas/Dropbox/Uni/Telematik/Master/Masterarbeit/git/masterigi/Code/src/Hardware/plots/")
    #vis.create_most_important_pairwise_marginal_plots("./plots/", 2)
    #vis.plot_pairwise_marginal([0,2])
    vis.create_all_plots()

    exit()
        
    '''
    #import ipdb
    #ipdb.set_trace()
    bins = np.arange(-5, 6, 1)
    indices = np.digitize(outputs - 32, bins)
    
    clustered_relevances = []
    for b in range(len(bins)):
        clustered_relevances.append((relevances[indices == b, :], np.mean(relevances[indices == b, :], axis=0), np.std(relevances[indices == b, :], axis=0)))
    
    print((clustered_relevances[5][1], clustered_relevances[5][2]))    
    '''        

