import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

############################################ Exercice 0 ############################################
def generate_balanced_random_patterns(N, M):
    return np.array(np.random.choice([-1, 1], (M, N)),dtype=float)


def update_state(S, W, beta=4):
    h = np.dot(W, S)
    return (np.tanh(beta * h))

#Ex 0.2
def flip_bits_V1(pattern, c):
    flip_indices = np.random.choice(len(pattern), size=int(len(pattern) * c), replace=False)
    pattern_flipped = pattern.copy()
    pattern_flipped[flip_indices] *= -1
    
    return pattern_flipped


def compute_overlap(state, patterns):
    return np.dot(patterns,state) / len(state)

def run_standard_hopfield_network(N, M, T):
    patterns = generate_balanced_random_patterns(N, M)
    W = 1/N * np.dot(patterns.T, patterns)
 
    # Set initial state close to the first pattern
    initial_state = flip_bits_V1(patterns[0], c=0.05)
    # Let the network evolve
    state = initial_state
    for t in range(T):  # Simulate for 20 time steps
            state = update_state(state, W)
            overlaps = compute_overlap(state, patterns)
            #print(f"Time step {t}, Overlaps: {overlaps}")
    
    return state, patterns 

def plot_standard_hopfield_network_results(M, state, patterns):
    fig, ax = plt.subplots(nrows=M, ncols=2, figsize=(5, 5))
    # Display the original pattern
    for i in range(M):
        ax[i, 0].imshow(patterns[i].reshape(10, 10), cmap='binary', vmin=-1, vmax=1)
        ax[i, 0].set_title(f'Original Pattern {i+1}')
        ax[i, 0].axis('off')  # Hide grid lines and ticks for clarity
        # Display the retrieved pattern
        ax[i, 1].imshow(state.reshape(10, 10), cmap='binary', vmin=-1, vmax=1)
        ax[i, 1].set_title(f'Retrieved Pattern {i+1}')
        ax[i, 1].axis('off')  # Hide grid lines and ticks for clarity
    plt.suptitle("Comparison of Original and Retrieved Patterns")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Ex0.2_pattern_retrieval.png", )
    plt.show()


#################################### Exercice 1 #######################################
##############################################Ex1.1

def update_state_with_overlaps(state,patterns,N, beta=4):
    #h = np.zeros_like(S, dtype=float)
    #for i in range(N):
        #h[i] = np.sum(m * patterns[:, i])
    m= np.dot(patterns, state) 
    h=np.dot(m,patterns)
    return np.tanh(beta * h)

def run_standard_hopfield_network_with_overlaps(N, M, T):
    patterns = generate_balanced_random_patterns(N, M)
    

    #W = 1/N * np.dot(patterns.T, patterns)
    # Set initial state close to the first pattern
    initial_state = flip_bits_V1(patterns[0], c=0.05)
    # Let the network evolve
    state = initial_state
    for t in range(T):  # Simulate for 20 time steps
        state = update_state_with_overlaps(state,patterns,N,beta=4)


    return state, patterns


#################################################Ex1.2
def hamming_distance(P1, P2):
    return (len(P1) - np.dot(P1, P2)) / (2 * len(P1))

################################################Ex1.3

def run_standard_hopfield_network_with_hamming_distance(N,M,T):
    patterns = generate_balanced_random_patterns(N, M)
    patterns =np.array(patterns,dtype=float)
    initial_state = flip_bits_V1(patterns[0], c=0.15)
    state = initial_state
    distances = []


  
    # Simulate the network
    for t in range(T):

        distances.append([hamming_distance(state, p) for p in patterns])
        state = update_state_with_overlaps(state,patterns,N)  # Example overlap
        overlaps = compute_overlap(state, patterns)
        print(f"Time step {t}, Overlaps: {overlaps}")
    
    return np.array(distances,dtype=float), state, patterns

def plot_patterns_state_comparison_hamming_distances(distances,M, state,patterns):

    fig, ax = plt.subplots(nrows=M, ncols=2, figsize=(12, 12))
    
    for i in range(M):
        # Display the original pattern
        ax[i, 0].imshow(patterns[i].reshape(15, 20), cmap='binary', vmin=-1, vmax=1)
        ax[i, 0].set_title(f'Original Pattern {i+1}')
        ax[i, 0].axis('off')  # Hide grid lines and ticks for clarity
    
        # Display the retrieved pattern (assuming final state resembles the first pattern)
        ax[i, 1].imshow(state.reshape(15, 20), cmap='binary', vmin=-1, vmax=1)
        ax[i, 1].set_title(f'Retrieved Pattern {i+1}')
        ax[i, 1].axis('off')  # Hide grid lines and ticks for clarity
    
    # Add a super title and show the plot for patterns
    plt.suptitle('Comparison of Original and Retrieved Patterns')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Plot Hamming distances
    plt.figure(figsize=(10, 5))
    
    for i in range(M):
        plt.plot(distances[:, i], label=f'Pattern {i+1}')
    
    plt.axhline(y=0.05,xmin=0,color="black",linestyle='--')
    plt.xlim(0,8)
    plt.xlabel('Time step')
    plt.ylabel('Hamming distance')
    plt.title('Evolution of Hamming Distances Over Time')
    plt.legend()
    plt.savefig("./Figures/Ex1.3-Hamming distances.png")
    plt.show()

##############################################Ex1.4
def pattern_retrieval_error_and_count(patterns, N, T=50, beta=4):
    retrieval_errors = []
    retrieval_counts = []
    for pattern in patterns:
        initial_state = flip_bits_V1(pattern, c=0.05)
        state = initial_state.copy()
        for t in range(T):
            state = update_state_with_overlaps(state,patterns, beta)
            retrieval_errors.append(hamming_distance(state,pattern))
        retrieval_counts.append(hamming_distance(state,pattern) <= 0.05)
    return np.mean(retrieval_errors), np.std(retrieval_errors), np.sum(retrieval_counts)

# Run simulations for different dictionary initializations
def run_simulation_dictionary(M, N=300, iterations=5,beta=4):
    mean_errors = []
    std_errors = []
    pattern_counts = []
    for _ in range(iterations):
        patterns = generate_balanced_random_patterns(N, M)
        mean_error, std_error, count = pattern_retrieval_error_and_count(patterns, N, beta=beta)
        mean_errors.append(mean_error)
        std_errors.append(std_error)
        pattern_counts.append(count)
    return np.mean(mean_errors), np.mean(std_errors), np.mean(pattern_counts)

#####################################Ex1.7
def capacity_study(N_values, loading_values, trials=10):
    """Study the capacity of Hopfield networks across different sizes and loadings."""
    results = {N: [] for N in N_values}

    for N in N_values:
        print(N)
        for L in loading_values:
            M = int(L * N)
            retrieval_rates = [run_simulation_dictionary(M, N) for _ in range(trials)]
            mean_retrieval_rate = np.mean(retrieval_rates)
            std_retrieval_rate = np.std(retrieval_rates)
            results[N].append((mean_retrieval_rate, std_retrieval_rate))

    # Plotting the results
    plt.figure(figsize=(8, 6))
    for N in N_values:
        means, stds = zip(*results[N])
        plt.errorbar(loading_values, means, yerr=stds, label=f'N={N}')

    plt.xlabel('Loading L = M/N')
    plt.ylabel('Average Retrieved Patterns / N')
    plt.title('Network Capacity vs Network Size')
    plt.legend()
    plt.grid(True)
    plt.savefig("./Figures/Network Capacity vs Network size")
    plt.show()

##########################################Ex1.8

def plot_beta_impact(N, M, beta_values):
    """ Plot the impact of beta on network retrieval capacity. """
    retrieval_rates=[]
    for beta in beta_values:
        _,_,retrieval_rate=run_simulation_dictionary(M, N, beta=beta)
        retrieval_rates.append(retrieval_rate)
    plt.figure(figsize=(8, 5))
    plt.plot(beta_values, retrieval_rates, marker='o')
    plt.xlabel('Inverse Temperature Beta')
    plt.ylabel('Retrieval Rate')
    plt.title('Impact of Beta on Memory Retrieval')
    plt.savefig("./Figures/impact_of_beta.png")
    plt.grid(True)
    plt.show()


############################################# Exercice 2 ##############################################
#########################################Ex2.2
def generate_low_activity_patterns(N, M, activity):
    """
    Generate M low-activity patterns with N neurons each,
    where each neuron has a probability 'activity' of being 1.
    """
    return np.random.choice([0, 1], (M, N), p=[1-activity, activity]).astype(float)
def compute_weight_matrix(pattern,N,a,b):
    pattern_a=pattern-np.full((pattern.shape),a)
    pattern_b=pattern-np.full((pattern.shape),b)
    
    c= 2/(a*(1-a))
    
    return c/N* np.dot(pattern_b.T, pattern_a)

def hamming_distance_V2(P, Q):
    '''
    Compute the Hamming distance between two patterns.
    '''

    return np.sum(P != Q,dtype=float) /(2*len(P))

def flip_bits_V2(pattern, c):
    flip_indices = np.random.choice(len(pattern), size=int(len(pattern) * c), replace=False)
    pattern_flipped = pattern.copy()
    for indices in flip_indices:
        if pattern_flipped[indices] == 1:
            pattern_flipped[indices] == 0
        elif pattern_flipped[indices] == 0:
            pattern_flipped[indices] == 1

    
    return pattern_flipped

def stochastic_spike_variable(S):
    """
    Generate a stochastic spike variable for each neuron based on its state S.
    Probability is derived from the neuron's continuous value.
    """
    return np.random.binomial(1, 0.5 * (S + 1))

def compute_overlaps(patterns, S,a):
    """
    Compute the overlaps m_mu for each pattern.
    """
    overlaps= (2/(a*(1-a)))*np.dot((patterns-a),S)
   
    return overlaps

def update_states_with_overlaps(patterns, overlaps,theta, beta,b):
    """
    Update the states of the network based on overlaps and pattern influence.
    """
    H= np.dot(overlaps,(patterns-b))
    H-=theta
    return np.tanh(beta * H)


def run_simulation_low_activity(N, M, a, b, theta_values, beta, T, c=2):
    """
    Run the simulation for multiple theta values and plot the retrieval accuracy.
    """
    patterns = generate_low_activity_patterns(N, M, a)
    initial_state = flip_bits_V2(patterns[0], c=0.05)  # Initialize the state close to the first pattern
    hamming_distances = []

    for theta in theta_values:
            S = initial_state.copy()
            for t in range(T):
                overlaps = compute_overlaps(patterns, S,a)
                S = update_states_with_overlaps(patterns, overlaps,theta, beta,b)
                S = np.array([stochastic_spike_variable(si) for si in S])
            # Evaluate performance after the last update
            distances = [hamming_distance_V2(S, p) for p in patterns]
            hamming_distances.append(distances[0])

    return theta_values, hamming_distances


##########################################Ex2.3
def pattern_retrieval_error_and_count_low(patterns, N,a,theta, T=50, beta=4):
    retrieval_errors = []
    retrieval_counts = []


    for pattern in patterns:
        initial_state = flip_bits_V1(pattern, c=0.05)
        state = initial_state.copy()
        for t in range(T):
            overlaps = compute_overlaps(patterns, state,a)
            state = update_states_with_overlaps(patterns,overlaps,theta, beta,a)
            state = np.array([stochastic_spike_variable(si) for si in state])
        retrieval_errors.append(hamming_distance_V2(state,pattern))

        retrieval_counts.append(hamming_distance_V2(state,pattern) <= 0.05)

    return np.mean(retrieval_errors), np.std(retrieval_errors), np.sum(retrieval_counts)

def run_low_activity_simulation_dictionary(M, a,theta,N=300, iterations=5,beta=4):
    mean_errors = []
    std_errors = []
    pattern_counts = []
    for _ in range(iterations):
        patterns = generate_low_activity_patterns(N, M,a)
        mean_error, std_error, count = pattern_retrieval_error_and_count_low(patterns, N,a,theta,beta=beta)
        mean_errors.append(mean_error)
        std_errors.append(std_error)
        pattern_counts.append(count)
    return np.mean(mean_errors), np.mean(std_errors), np.mean(pattern_counts)

def capacity_vs_theta(patterns,N,a,b,theta_values,T=20,beta=4):

    capacities=[]
    for theta in theta_values:
        retrieval_counts=0
        for pattern in patterns:
            initial_state= flip_bits_V1(pattern,c=0.05)
            state=initial_state.copy()
            for t in range(T):
                overlaps = compute_overlaps(patterns,state,a)
                state = update_states_with_overlaps(patterns,overlaps,theta, beta,b)
                state = np.array([stochastic_spike_variable(si) for si in state])
            if hamming_distance_V2(state,pattern) <= 0.05:
                retrieval_counts+=1
        capacities.append(retrieval_counts/N)
    return capacities,

def capacity_study_theta(N,M_values,a,b, theta_values, trials=10):
    """Study the capacity of Hopfield networks across different sizes of dictionary and theta values."""
      # Range of theta values to test
    M_capacities=np.zeros((len(M_values), len(theta_values)))
    for i, M in enumerate(M_values):
        print(M)
        patterns = generate_low_activity_patterns(N, M,a)
        capacities = capacity_vs_theta(patterns,N,a,b,theta_values)
        
        capacities=np.reshape(capacities,(len(capacities[0]),))
        M_capacities[i]+=capacities


    # Plotting the results
    plt.figure(figsize=(8, 6))
    
    for i, M in enumerate(M_values):
    
        plt.plot(theta_values,M_capacities[i],label=f'M={M}')

    plt.xlabel('Theta values')
    plt.ylabel('Capacity')
    plt.title('Network Capacity vs Theta values')
    plt.legend()
    plt.grid(True)
    plt.savefig("./Figures/Network Capacity vs Network size")
    plt.show()
    
#Ex2.6


def simulate_capacity(N, M, activity, theta, beta=4, iterations=100):
    patterns = generate_low_activity_patterns(N, M, activity)
    retrieved_patterns = 0

    for _ in range(iterations):
        initial_state = np.random.choice([0, 1], N, p=[1-activity, activity])
        state = initial_state

        for t in range(20):  # Run for a certain number of time steps
            overlaps= compute_overlaps(patterns, state, activity)
            state = update_states_with_overlaps(patterns,overlaps,[theta], beta,activity)

        # Check if the first pattern is retrieved
        if hamming_distance(state, patterns[0]) <= 0.05:
            retrieved_patterns += 1

    return retrieved_patterns / iterations


###################################################### Exercice 3 ################################################################
############# Retrival with weight matrix  ###############

def initialize_neuron_types(N, p_exc=0.8):
    """
    Randomly assign each neuron as excitatory or inhibitory.
    Args:
    N: Total number of neurons
    p_exc: Probability that a neuron is excitatory

    Returns:
    neuron_types: Array of neuron types (1 for excitatory, 0 for inhibitory)
    """
    neuron_types = np.random.choice([1, 0], size=N, p=[p_exc, 1-p_exc])
    
    return neuron_types

def flip_bits(pattern, c, neuron_types):
    """
    Flip a portion 'c' of bits in the pattern, but only for excitatory neurons.
    
    Args:
    pattern (np.array): The array representing the neural pattern.
    c (float): The fraction of the excitatory neurons to flip.
    neuron_types (np.array): An array indicating whether each neuron is excitatory (1) or inhibitory (0).
    
    Returns:
    np.array: The new pattern with flipped bits for a subset of excitatory neurons.
    """
    pattern = pattern.reshape(-1, 1)
    # Get indices of excitatory neurons
    excitatory_indices = np.where(neuron_types == 1)[0]

    # Choose a subset of excitatory neurons to flip
    num_to_flip = int(len(excitatory_indices) * c)
    #print("Number of indices to flip", num_to_flip)
    flip_indices = np.random.choice(excitatory_indices, size=num_to_flip, replace=False)

    pattern_flipped = pattern.copy()
    #print("Flipped indices:", flip_indices)
    for i in flip_indices:
        pattern_flipped[i] = 1 - pattern[i]  # This flips 0 to 1 and 1 to 0

    pattern_flipped = pattern_flipped.reshape(1, -1)
    return pattern_flipped


def create_weight_matrix(N, neuron_types, c, a, N_I, K, patterns):
    """
    Create a weight matrix for the neural network.
    Args:
    N: Total number of neurons
    neuron_types: Array of neuron types (1 for excitatory, 0 for inhibitory)
    c: Scaling factor for excitatory connections
    a: Activity level of the network
    N_I: Number of inhibitory neurons
    K: Scaling factor for inhibitory connections
    patterns: Array of stored patterns

    Returns:
    W: Weight matrix (N x N)
    """
    W = np.zeros((N, N), dtype=float)

    # Get indices of excitatory and inhibitory neurons
    exc_neurons = np.where(neuron_types == 1)[0]
    inh_neurons = np.where(neuron_types == 0)[0]
    
    # Compute contributions from patterns for E-E connections
    for pattern in patterns:
        pattern_exc = pattern[exc_neurons]
        # Outer product gives a matrix of all pairwise multiplications
        W[np.ix_(exc_neurons, exc_neurons)] += c / N * np.outer(pattern_exc, pattern_exc)
        W[neuron_types] += (-c * a / N_I)*pattern
    # Set I-E connections (note: I-E weights are independent of patterns)
    

    # Set E-I connections (note: E-I weights are independent of patterns)
    W[neuron_types] = 1 / K

    # I-I connections are set to zero, which is already done by np.zeros

    return W



def update_states_with_weight(N, W, state, neuron_types, theta, beta, update_mode):
    # Ensure state is a column vector
    if update_mode == "sequential":
        state = state.reshape(-1,1)
        h = W.dot(state)
        new_state = np.zeros_like(state,dtype=float)
        #print("Updating")
        for i in range(N):
            #print(i)
            if neuron_types[i] == 1:
                new_state[i] = np.tanh(beta*(h[i] - theta))
                
                #print("new_state: Excitatory", new_state[i])
            else:
                new_state[i] = h[i]
                #print("new_state: Inhibitory", new_state[i])
    
    #if update_mode == "synchronous" :
    return new_state
    




def plot_exc_inh(patterns, initial_state, w, neuron_types, N):
    # Filter to include only excitatory neurons for pattern and state
    excitatory_indices = np.where(neuron_types == 1)[0]
    patterns_excitatory = patterns[:,excitatory_indices]
    initial_state_excitatory = initial_state[:, excitatory_indices]

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # Adjusted size to accommodate four plots

    sns.heatmap(patterns_excitatory, ax=axes[0], cmap='viridis', cbar=False, annot=False)
    axes[0].set_title('Excitatory Neurons - Memory Pattern')

    sns.heatmap(initial_state_excitatory, ax=axes[1], cmap='viridis', cbar=False, annot=False)
    axes[1].set_title('Excitatory Neurons - Initial State')

    sns.heatmap(w, ax=axes[2], cmap='viridis', cbar=False, annot=False)
    axes[2].set_title('Weight Matrix')

    # Plotting the neuron types
    neuron_type_map = neuron_types.reshape(N,1 )  # Reshape for heatmap compatibility
    sns.heatmap(neuron_type_map, ax=axes[3], cmap='coolwarm', cbar=False, annot=False)
    axes[3].set_title('Neuron Types (Red = Excitatory, Blue = Inhibitory)')

    plt.tight_layout()
    plt.show()

def run_network_exc_inh(N, M, N_i, K, a, c, theta,inh_prob, T, beta, update_mode = "sequetial", plot = True):
    # Generate patterns
    patterns = generate_low_activity_patterns(N, M, a)
    #print("Patterns:\n", patterns)

    
    #plot_exc_inh(patterns, initial_state, w, neuron_types, N)
    # Initialize retrieval accuracy list
    retrieval_accuracies = []
    # Iterate over each pattern
    for idx in range(M):
        #print(f"Pattern {patterns[idx]}")

        # Initialize neuron types
        neuron_types = initialize_neuron_types(N, inh_prob)
        #print("Neuron types:", neuron_types)
    
        # Flip bits in the initial state
        initial_state = flip_bits(patterns[idx], 0.1, neuron_types)
        #print("Initial state:\n", initial_state)
        
        # Create the weight matrix
        w = create_weight_matrix(N, neuron_types, c, a, N_i, K, patterns[idx].reshape(1, -1))
        #print("Weights:\n", np.unique(w))
        #state =initial_state.copy()
        for i in range(T):
            #print(f"Updating step {i}")
           
            initial_state = update_states_with_weight(N, w, initial_state, neuron_types, theta, beta, update_mode='sequential')
            #print("State after update:", np.unique(initial_state))
            # Call stochastic_spike_variable with the entire array
            #initial_state = stochastic_spike_variable(initial_state)
            #print("State after stochastic update:", np.unique(initial_state))
            #plot_exc_inh(patterns, initial_state.reshape(1,-1), w, neuron_types, N)
    #initial_state = stochastic_spike_variable(initial_state)
    for pattern in patterns:
        accuracy = 1 - hamming_distance(initial_state, pattern.reshape(1,-1)) / N
        retrieval_accuracies.append(accuracy)
        
    #print(np.unique(initial_state[np.where(neuron_types == 1)]))
    #print(np.unique(initial_state))
    if plot:
        plot_exc_inh(patterns, initial_state.reshape(1,-1), w, neuron_types, N)
    
    return np.mean(retrieval_accuracies)

def real_test():
    N = 300
    M = 1
    N_i = 80
    K = 60
    a = 0.1
    c = 2 / (a * (1-a))
    theta = 0.
    inh_prob = (N-N_i)/N
    T = 10
    beta = 1.
    
    run_network_exc_inh(N, M, N_i, K, a, c, theta, inh_prob, T, beta)


################# same calculations with overlaps ###########
'''
def compute_overlaps_exc_inh(patterns, S, neuron_types, a):
    """
    Compute the overlaps m_mu for each pattern.
    
    Args:
    patterns (numpy.ndarray): Array of stored patterns in the network.
    S (numpy.ndarray): Current state of the network.
    neuron_types (numpy.ndarray): Array indicating neuron type (1 for excitatory, 0 for inhibitory).
    a (float): Activity level of the network.
    
    Returns:
    tuple: Tuple containing overlaps for excitatory neurons (m_mu_exc) and inhibitory neurons (m_mu_inh).
    """
    #S = S.flatten()
    # Separate excitatory and inhibitory neurons

    S = S.flatten()
    # Separate excitatory and inhibitory neurons
    S_exc = S[neuron_types == 1]  # States of excitatory neurons
    S_inh = S[neuron_types == 0]  # States of inhibitory neurons

    patterns_exc = patterns[:, neuron_types == 1]  # Patterns for excitatory neurons
    patterns_inh = patterns[:, neuron_types == 0]  # Patterns for inhibitory neurons

    # Compute overlaps for excitatory neurons
    
    overlaps_exc = np.dot((patterns_exc - a),S_exc)

    # Compute overlaps for inhibitory neurons
    #overlaps_inh = np.dot((patterns_inh - np.full(patterns.shape,a)), S_inh)
    overlaps_inh = (1/S_inh.size)*patterns_inh
    
    return overlaps_exc,overlaps_inh

    
    overlaps_inh = np.zeros(S_inh.shape, dtype=float)
    for pattern_inh in patterns_inh:
        overlaps_inh += 1/(S_inh.size)*pattern_inh
    

def update_state_with_overlaps(patterns, S, neuron_types, a,beta,theta):
    S.flatten()
    # Separate excitatory and inhibitory neurons
    S_exc = S[:,neuron_types == 1]  # States of excitatory neurons
    S_inh = S[:,neuron_types == 0]  # States of inhibitory neurons
    
    overlaps_exc,overlaps_inh = compute_overlaps_exc_inh(patterns, S, neuron_types, a)
    
    # Update states using the overlaps: apply activation function to excitatory and direct assignment to inhibitory
    new_S_exc = np.dot(overlaps_exc, S_exc.reshape(1,-1))
    
    new_S_inh = np.dot(overlaps_inh.T, S_inh.reshape(1,-1))
    
    print("New state exc",new_S_exc)
    print("New state inh",new_S_inh)
    # Combine updated states into a single array matching the original state array
    new_S = np.zeros_like(S,dtype=float)
    #print(new_S)
    new_S[neuron_types == 1] = new_S_exc

    new_S[neuron_types == 0] = new_S_inh.ravel()

    new_S= np.tanh(beta*(new_S - theta))
    
    print("New state before phi function:", new_S)
    return new_S
    '''
def compute_overlaps_exc_inh(patterns, S, neuron_types, a,K, N_i,c,N):
    """
    Compute the overlaps m_mu for each pattern.
    
    Args:
    patterns (numpy.ndarray): Array of stored patterns in the network.
    S (numpy.ndarray): Current state of the network.
    neuron_types (numpy.ndarray): Array indicating neuron type (1 for excitatory, 0 for inhibitory).
    a (float): Activity level of the network.
    
    Returns:
    tuple: Tuple containing overlaps for excitatory neurons (m_mu_exc) and inhibitory neurons (m_mu_inh).   
    """
    S = S.flatten()
    # Separate excitatory and inhibitory neurons
    S_exc = S[neuron_types == 1]  # States of excitatory neurons
    S_inh = S[neuron_types == 0]  # States of inhibitory neurons

    patterns_exc = patterns[:, neuron_types == 1]  # Patterns for excitatory neurons
    patterns_inh = patterns[:, neuron_types == 0]  # Patterns for inhibitory neurons
    #print("patterns-a",patterns_exc.shape)
    #print(S_exc.shape)
    #print(patterns_inh.shape)
    #print("inh",np.dot(np.array([1/N_i]) , patterns_exc).shape)
    #print(np.dot(patterns_exc - a, S_exc).shape)
    
    
    #S_exc = np.array(stochastic_spike_variable(S_exc),dtype=float)
    #S_inh = np.array(stochastic_spike_variable(S_inh),dtype=float)
    
    # Compute overlaps for excitatory neurons
    overlaps_exc = c/N*np.dot(patterns_exc - a, S_exc) - np.sum(np.dot(np.array([1/N_i]) , patterns_inh))
    
    print("patterns_exc", patterns_exc)
    print("S_exc", S_exc)
    
    print("patterns_inh", patterns_inh)
    print("sum patterns_inh",np.sum(patterns_inh))

    print('S_inh', S_inh)
    print('Positive part:',np.dot(patterns_exc - a, S_exc))
    print("Negative part:",np.sum(np.dot(np.array([1/N_i]) , patterns_inh)))
    
    # Compute overlaps for inhibitory neurons
    overlaps_inh = 1/K
    print("overlap inh",overlaps_inh)
    print("overlap exc",overlaps_exc)

    overlaps = np.zeros_like(S,dtype=float)

    overlaps[neuron_types==1] = overlaps_exc
    overlaps[neuron_types==0] = overlaps_inh

    return overlaps

def update_states_v4(N, W, c, patterns, a, K, N_i, state, neuron_types, theta, beta, update_mode):
    # Ensure state is a column vector
    state = state.reshape(-1, 1)
    state = stochastic_spike_variable(state)
    # Compute the input field h for all neurons
    h = compute_overlaps_exc_inh(patterns, state, neuron_types,a,K,N_i,c,N)

    # Update the states for excitatory neurons using tanh activation function
    excitatory_indices = (neuron_types == 1).reshape(-1)
    new_state_exc = np.tanh(beta * (h[excitatory_indices] - theta))

    # Update the states for inhibitory neurons (assuming no activation function is applied)
    inhibitory_indices = (neuron_types == 0).reshape(-1)
    new_state_inh = h[inhibitory_indices]

    new_state_exc = new_state_exc.reshape(-1, 1)
    new_state_inh = new_state_inh.reshape(-1, 1)
    print("new_state_exc",np.unique(new_state_exc))
    print(new_state_inh.shape)
    # Combine the updated states into a single array
    new_state = np.zeros_like(state, dtype=float)
    print(new_state.shape)
    new_state[excitatory_indices] = new_state_exc
    new_state[inhibitory_indices] = new_state_inh

    return new_state

def stochastic_spike_variable_v2(S,neuron_types):
    """
    Generate a stochastic spike variable for each neuron based on its state S.
    Probability is derived from the neuron's continuous value.
    """
    spikes = np.zeros(S.shape,dtype=float)
    spikes[neuron_types==1] = np.random.binomial(1, 0.5 * (S[neuron_types==1] + 1))
    spikes[neuron_types==0] = S[neuron_types==0]
    return spikes

def run_network_exc_inh_overlaps(N, M, N_i, K, a, c, theta,inh_prob, T, beta, update_mode = "sequetial", plot = True):
    # Generate patterns
    patterns = generate_low_activity_patterns(N, M, a)
    #print("Patterns:\n", patterns)

    
    #plot_exc_inh(patterns, initial_state, w, neuron_types, N)
    # Initialize retrieval accuracy list
    retrieval_accuracies = []
    # Iterate over each pattern
    for idx in range(M):
        #print(f"Pattern {patterns[idx]}")

        # Initialize neuron types
        neuron_types = initialize_neuron_types(N, inh_prob)
        #print("Neuron types:", neuron_types)
    
        # Flip bits in the initial state
        initial_state = flip_bits(patterns[idx], 0.1, neuron_types)
        #print("Initial state:\n", initial_state)
        
        # Create the weight matrix
        w = create_weight_matrix(N, neuron_types, c, a, N_i, K, patterns[idx].reshape(1, -1))
        #print("Weights:\n", np.unique(w))
        #initial_state =initial_state.copy()
        for i in range(T):
            #print(f"Updating step {i}")
            
            initial_state = update_states_v4(N,w,c,patterns,a,K,N_i,initial_state,neuron_types,theta,beta,"sequetial")
            initial_state = stochastic_spike_variable_v2(initial_state,neuron_types)
            #print("State after update:", np.unique(initial_state))
            # Call stochastic_spike_variable with the entire array
            #initial_state = stochastic_spike_variable(initial_state)
            #print("State after stochastic update:", np.unique(initial_state))
            #plot_exc_inh(patterns, initial_state.reshape(1,-1), w, neuron_types, N)
    #initial_state = stochastic_spike_variable(initial_state)
    for pattern in patterns:
        accuracy = 1 - hamming_distance(initial_state.T, pattern.reshape(1,-1).T) / N
        retrieval_accuracies.append(accuracy)
        
    #print(np.unique(initial_state[np.where(neuron_types == 1)]))
    #print(np.unique(initial_state))
    if plot:
        plot_exc_inh(patterns, initial_state.reshape(1,-1), w, neuron_types, N)
    
    return np.mean(retrieval_accuracies)

def overlap_test():
    N = 300
    M = 1
    N_i = 80
    K = 60
    a = 0.1
    c = 2 / (a * (1-a))
    theta = 0.
    inh_prob = (N-N_i)/N
    T = 10
    beta = 1.
    
    run_network_exc_inh_overlaps(N, M, N_i, K, a, c, theta,inh_prob, T, beta, update_mode = "sequetial", plot = True)
