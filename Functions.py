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
            i=0
            mean_retrieve=[]
            std_retrieve=[]
            M = int(L * N)
            while(i<5):
                i+=1
                retrieval_rates = [run_simulation_dictionary(M, N) for _ in range(trials)]
                mean_retrieve.append(retrieval_rates)
                std_retrieve.append(np.std(retrieval_rates))

            mean_retrieval_rate = np.mean(mean_retrieve)
            std_retrieval_rate = np.mean(std_retrieve)
                
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
    # Determine the number of bits to flip
    num_flips = int(len(pattern) * c)

    # Choose indices to flip
    flip_indices = np.random.choice(len(pattern), size=num_flips, replace=False)

    # Create a copy of the pattern to avoid modifying the original
    pattern_flipped = pattern.copy()

    # Flip the bits at the chosen indices
    pattern_flipped[flip_indices] = 1 - pattern_flipped[flip_indices]

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
    H= np.dot(overlaps,(patterns-b))/(len(patterns[0]))
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

def capacity_study_theta(N,M_values,a,b, theta_values,trials=10):
    """Study the capacity of Hopfield networks across different sizes of dictionary and theta values."""
      # Range of theta values to test
    M_capacities=np.zeros((len(M_values), len(theta_values)))
    for i, M in enumerate(M_values):
        #print(M)
        patterns = generate_low_activity_patterns(N, M,a)
        
        capacities = capacity_vs_theta(patterns,N,a,b,theta_values)
        M_capacities[i]+=capacities

    capacities=np.reshape(capacities,(len(capacities[0]),))


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


def capacity_study_V2(N_values,a,theta, loading_values, trials=10):
    """Study the capacity of Hopfield networks across different sizes and loadings."""
    results = {N: [] for N in N_values}

    for N in N_values:
        print(N)
        for L in loading_values:
            M = int(L * N)
            mean_retrieval_rate,std_retrieval_rate,_=run_low_activity_simulation_dictionary(M,a, theta) 
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


###################################################### Exercice 3 ################################################################
############# Retrival with weight matrix  ###############

def initialize_neuron_types(N, M, p_exc=0.8, include_second_inhibitory=False):
    """
    Randomly assign each neuron as excitatory, first inhibitory population, or second inhibitory population based on a boolean flag.

    Args:
    N: Total number of neurons
    M: Number of pattern sets
    p_exc: Probability that a neuron is excitatory
    include_second_inhibitory: Boolean flag to include the second inhibitory population

    Returns:
    neuron_types_list: List of arrays of neuron types for each pattern set
    """
    neuron_types_list = []
    for i in range(M):
        if include_second_inhibitory:
            # Calculate probabilities for inhibitory populations
            p_inh_total = 1 - p_exc
            p_inh1 = p_inh_total / 2
            p_inh2 = p_inh_total / 2

            # Ensure the probabilities sum to 1
            assert np.isclose(p_exc + p_inh1 + p_inh2, 1.0), "Probabilities must sum to 1"
            neuron_types = np.random.choice([1, 0, 2], size=N, p=[p_exc, p_inh1, p_inh2])
        else:
            # Only excitatory and first inhibitory population
            assert np.isclose(p_exc + (1 - p_exc), 1.0), "Probabilities must sum to 1"
            neuron_types = np.random.choice([1, 0], size=N, p=[p_exc, 1 - p_exc])
        neuron_types_list.append(neuron_types)
    return neuron_types_list


import numpy as np


def flip_bits_V3(patterns, c, neuron_types, include_second_inhibitory=False):
    """
    Flip a portion 'c' of bits in the pattern, but only for excitatory neurons.

    Args:
    patterns (np.array): The array representing the neural patterns.
    c (float): The fraction of the excitatory neurons to flip.
    neuron_types (np.array): An array indicating whether each neuron is excitatory (1), first inhibitory (0), or second inhibitory (2).
    include_second_inhibitory (bool): Whether to include the second inhibitory population.

    Returns:
    np.array: The new patterns with flipped bits for a subset of excitatory neurons.
    """
    # Get indices of excitatory neurons
    excitatory_indices = np.where(neuron_types == 1)[0]

    # Number of excitatory neurons to flip
    num_to_flip = int(len(excitatory_indices) * c)

    # Prepare a list to store the flipped patterns
    initial_state_list = []

    for pattern in patterns:
        pattern = pattern.reshape(-1)

        # Choose a subset of excitatory neurons to flip
        flip_indices = np.random.choice(excitatory_indices, size=num_to_flip, replace=False)

        # Create a copy of the pattern to avoid modifying the original
        pattern_flipped = pattern.copy()

        # Flip the bits at the chosen indices
        pattern_flipped[flip_indices] = 1 - pattern_flipped[flip_indices]

        # Append the flipped pattern to the list
        initial_state_list.append(pattern_flipped.reshape(1, -1))

    return initial_state_list


def hamming_distance_V3(P1, P2):
    """Compute the Hamming distance between two binary vectors."""
    """
    P1 = P1.ravel()
    P2 = P2.ravel()
    print(P1.shape)
    return (P1.shape[0] - np.dot(P1.reshape(1,-1), P2.reshape(-1,1))) / (2 * P1.shape[0])
    """
    #print(P1.ravel(),P2.ravel())
    return np.sum(P1.ravel() != P2.ravel()) / (2 * P1.shape[0])


def plot_exc_inh(patterns, initial_state, neuron_types, N, h):
    # Filter to include only excitatory neurons for pattern and state
    excitatory_indices = np.where(neuron_types == 1)[0]
    inhibitory_indices = np.where(neuron_types == 0)[0]
    patterns = patterns.reshape(1, -1)
    patterns_excitatory = patterns[:, excitatory_indices]
    initial_state_excitatory = initial_state[:, excitatory_indices]

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # Adjusted size to accommodate four plots

    sns.heatmap(patterns_excitatory, ax=axes[0], cmap='viridis', cbar=False, annot=False)
    axes[0].set_title('Excitatory Neurons - Memory Pattern')

    sns.heatmap(initial_state_excitatory, ax=axes[1], cmap='viridis', cbar=False, annot=False)
    axes[1].set_title('Excitatory Neurons - Initial State')

    # Plotting the neuron types
    neuron_type_map = neuron_types.reshape(N, 1)  # Reshape for heatmap compatibility
    sns.heatmap(neuron_type_map, ax=axes[2], cmap='coolwarm', cbar=False, annot=False)
    axes[2].set_title('Neuron Types (Red = Excitatory, Blue = Inhibitory)')

    sns.heatmap(h.reshape(-1, 1), ax=axes[3], cmap='viridis', cbar=False, annot=False)
    axes[3].set_title('Overlap function h')

    plt.tight_layout()
    plt.show()


def plot_exc_inh_external(patterns, states_over_time, neuron_types_list, N, overlaps):
    # Plot overlap values
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    num_positions = len(next(iter(overlaps.values())))
    positions = list(range(num_positions))

    for iteration, (key, values) in enumerate(overlaps.items()):
        ax2.plot(positions, values, label=f'Pattern {key + 1}', marker='o')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Overlap Value')
    ax2.set_title('Overlap Values as a Function of Iteration')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Create combined raster plots for excitatory and inhibitory neurons
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # Colors for each pattern
    colors = sns.color_palette("husl", len(patterns))

    # Combined raster plot for excitatory neurons
    offset = 0
    for mu in range(len(patterns)):
        neuron_types = neuron_types_list[mu]
        states = states_over_time[mu]

        excitatory_indices = np.where(neuron_types == 1)[0]
        for neuron_idx, neuron_id in enumerate(excitatory_indices):
            spike_times = [t for t, state in enumerate(states) if state[neuron_id] == 1]
            ax3.eventplot(spike_times, lineoffsets=neuron_idx + offset, linelengths=0.9, colors=[colors[mu]])

        ax3.text(-5, offset + len(excitatory_indices) / 2, f'Pattern {mu + 1}', va='center', ha='right', fontsize=10, color='white')
        offset += len(excitatory_indices)
        ax3.axhline(y=offset - 0.5, color='gray', linestyle='--')  # Add horizontal line to separate patterns

    ax3.set_title('Combined Raster Plot - Excitatory Neurons')
    ax3.set_ylabel('Neuron Index')

    # Combined raster plot for inhibitory neurons
    offset = 0
    for mu in range(len(patterns)):
        neuron_types = neuron_types_list[mu]
        states = states_over_time[mu]

        inhibitory_indices = np.where(neuron_types == 0)[0]
        for neuron_idx, neuron_id in enumerate(inhibitory_indices):
            spike_times = [t for t, state in enumerate(states) if state[neuron_id] == 1]
            ax4.eventplot(spike_times, lineoffsets=neuron_idx + offset, linelengths=0.9, colors=[colors[mu]])

        ax4.text(-5, offset + len(inhibitory_indices) / 2, f'Pattern {mu + 1}', va='center', ha='right', fontsize=10, color='white')
        offset += len(inhibitory_indices)
        ax4.axhline(y=offset - 0.5, color='gray', linestyle='--')  # Add horizontal line to separate patterns

    ax4.set_title('Combined Raster Plot - Inhibitory Neurons')
    ax4.set_ylabel('Neuron Index')
    ax4.set_xlabel('Time Step')

    plt.tight_layout()
    plt.show()


def compute_h_in_terms_of_overlaps(pattern, S, neuron_types, a, c, K, patterns, use_second_inhibitory,
                                   use_external_input, J):
    """
    Compute the total input h for excitatory and inhibitory neurons in terms of the overlap variables.

    Args:
    patterns (numpy.ndarray): Array of stored patterns in the network.
    S (numpy.ndarray): Current state of the network.
    neuron_types (numpy.ndarray): Array indicating neuron type (1 for excitatory, 0 for first inhibitory population, 2 for second inhibitory population).
    a (float): Activity level of the network.
    c (float): Scaling factor.
    K (int): Number of excitatory neurons each inhibitory neuron connects to.
    use_second_inhibitory (bool): Whether to include the second inhibitory population in the calculations.

    Returns:
    numpy.ndarray: Total input h for all neurons.
    """
    N = len(S)
    num_patterns = patterns.shape[0]

    # Separate excitatory and inhibitory neurons
    S_exc = S[neuron_types == 1]  # States of excitatory neurons
    S_inh_1 = S[neuron_types == 0]  # States of first inhibitory population

    N_exc = S_exc.shape[0]
    N_I1 = S_inh_1.shape[0]
    J = 2.0

    # Initialize overlaps
    m_exc = np.zeros(num_patterns)
    m_inh_1 = np.zeros(num_patterns)
    overlap = 0.0
    if use_second_inhibitory:
        S_inh_2 = S[neuron_types == 2]  # States of second inhibitory population
        N_I2 = S_inh_2.shape[0]
        m_inh_2 = np.zeros(num_patterns)
        h_inh_2 = np.zeros(N_I2)
        #print("type 2 neuron pop",pattern[neuron_types == 2])
        #print("type 2 neurons state", S_inh_2)
    # Compute overlaps for each pattern
    m_exc = np.dot(pattern[neuron_types == 1].reshape(1, -1), S_exc.reshape(-1, 1)) / N

    if N_I1 != 0:
        m_inh_1 = np.sum(S_inh_1.reshape(-1, 1)) / N_I1
    else:
        m_inh_1 = 0.0
    if use_second_inhibitory and np.mean(S_exc) > a:
        if N_I2 != 0:
            m_inh_2 = np.sum(S_inh_2.reshape(-1, 1)) / N_I2
        else:
            m_inh2 = 0

    # Compute external input
    if use_external_input:
        h_ext = np.zeros(N_exc)
        for i in range(N_exc):
            h_ext[i] = J * (pattern[neuron_types == 1][i] - np.mean(patterns[:, neuron_types == 1][:, i]))
        #print(h_ext)

    h_exc = np.zeros(N_exc)
    for i in range(N_exc):
        h_exc[i] = np.sum(pattern[neuron_types == 1][i] * (c * m_exc - c * a * m_inh_1))
        overlap = c * m_exc - c * a * m_inh_1
        #print("before external",h_exc[i])
        if use_external_input:
            h_exc[i] += h_ext[i]
            #print("after external",h_exc[i])
        if use_second_inhibitory and np.mean(S_exc) > a:
            h_exc[i] = np.sum(pattern[neuron_types == 1][i] * c * (m_exc - a * m_inh_1 - a * m_inh_2))
            overlap = c * m_exc - c * a * m_inh_1 - a * m_inh_2
    # Compute total input to first inhibitory neurons
    h_inh_1 = np.zeros(N_I1)
    for k in range(N_I1):
        h_inh_1[k] = 1 / K * S_inh_1[k]

    if use_second_inhibitory and np.sum(S_exc) / N > a:
        for k in range(N_I2):
            h_inh_2[k] = 1 / K * S_inh_2[k]

    # Combine the total inputs
    h = np.zeros_like(S, dtype=float)
    h[neuron_types == 1] = h_exc.reshape(-1, 1)
    h[neuron_types == 0] = h_inh_1.reshape(-1, 1)
    """print("h_exc",h_exc)
    print("h_inh_1",h_inh_1)"""
    if use_second_inhibitory:
        h[neuron_types == 2] = h_inh_2.reshape(-1, 1)

    return h, overlap


def synchronous_update(state, pattern, neuron_types, a, c, K, beta, theta, patterns, use_second_inhibitory=False,
                       use_external_input=False, J=2.0):
    state = state.reshape(-1, 1)
    h, overlap = compute_h_in_terms_of_overlaps(pattern, state, neuron_types, a, c, K, patterns, use_second_inhibitory,
                                                use_external_input, J)

    excitatory_indices = (neuron_types == 1).reshape(-1)

    new_state_exc = np.tanh(beta * (h[excitatory_indices] - theta))

    inhibitory_indices = (neuron_types == 0).reshape(-1)
    new_state_inh = h[inhibitory_indices]

    new_state_exc = new_state_exc.reshape(-1, 1)
    new_state_inh = new_state_inh.reshape(-1, 1)

    new_state = np.zeros_like(state, dtype=float)

    new_state[excitatory_indices] = new_state_exc
    new_state[inhibitory_indices] = new_state_inh

    new_state = stochastic_spike_variable(new_state)
    return new_state, h, overlap


def sequential_update(state, pattern, neuron_types, a, c, K, beta, theta, patterns, use_second_inhibitory=False,
                      use_external_input=False, J=2.0):
    h_inh, overlap = \
        compute_h_in_terms_of_overlaps(pattern, state, neuron_types, a, c, K, patterns, use_second_inhibitory,
                                       use_external_input, J)
    state[neuron_types == 0] = stochastic_spike_variable(h_inh[neuron_types == 0])
    # Then update excitatory neurons with the new state of inhibitory neurons
    h, overlap = compute_h_in_terms_of_overlaps(pattern, state, neuron_types, a, c, K, patterns, use_second_inhibitory,
                                                use_external_input, J)
    h_exc = h[neuron_types == 1]
    state[neuron_types == 1] = stochastic_spike_variable(np.tanh(beta * (h_exc - theta)))
    return state, h, overlap


def run_network_exc_inh_synchronous(N, M, N_i, K, a, c, theta, exc_prob, T, beta, plot=True,
                                    use_second_inhibitory=False, use_external_input=False, J=2.0):
    patterns = generate_low_activity_patterns(N, M, a)
    retrieval_accuracies = []
    neuron_types_list = initialize_neuron_types(N, M, exc_prob, use_second_inhibitory)
    retrieval_errors = []
    retrieval_counts = []
    state = flip_bits_V3(patterns, a, neuron_types_list)
    #print("Patterns", patterns)
    #print("State", state)
    S_exc_history = {mu: [] for mu in range(M)}
    overlaps = {mu: [] for mu in range(M)}

    for mu in range(M):

        neuron_types = neuron_types_list[mu]
        pattern = patterns[mu]
        S = state[mu].reshape(-1, 1)
        for i in range(T):
            if use_external_input and i < 5:
                S, h, overlap = synchronous_update(S, pattern, neuron_types, a, c, K, beta, theta, patterns,
                                                   use_second_inhibitory=use_second_inhibitory,
                                                   use_external_input=use_external_input, J=J)
            else:
                S, h, overlap = synchronous_update(S, pattern, neuron_types, a, c, K, beta, theta, patterns,
                                                   use_second_inhibitory=use_second_inhibitory,
                                                   use_external_input=False, J=J)

            overlaps[mu].append(overlap.item())
            print(overlap.item())
            S_exc_history[mu].append(S.copy())
        accuracy = 1 - hamming_distance_V3(S[neuron_types == 1], np.array(pattern)[neuron_types == 1]) / N
        retrieval_accuracies.append(accuracy)
        retrieval_errors.append(hamming_distance_V3(S[neuron_types == 1], np.array(pattern)[neuron_types == 1]))
        retrieval_counts.append(hamming_distance_V3(S[neuron_types == 1], np.array(pattern)[neuron_types == 1]) <= 0.05)
    if plot:
        plot_exc_inh_external(patterns, S_exc_history, neuron_types_list, N, overlaps)

    mean_retrieval_accuracies = np.mean(retrieval_accuracies)
    mean_retrieval_errors = np.mean(retrieval_errors)
    std_retrieval_errors = np.std(retrieval_errors)
    counts = np.sum(retrieval_counts)
    return mean_retrieval_accuracies, mean_retrieval_errors, std_retrieval_errors, counts


def run_network_exc_inh_sequential(N, M, N_i, K, a, c, theta, exc_prob, T, beta, plot=True, use_second_inhibitory=False,
                                   use_external_input=False, J=2.0):
    patterns = generate_low_activity_patterns(N, M, a)
    retrieval_accuracies = []
    neuron_types_list = initialize_neuron_types(N, M, exc_prob, use_second_inhibitory)
    retrieval_errors = []
    retrieval_counts = []
    state = flip_bits_V3(patterns, a, neuron_types_list)
    S_history = {mu: [] for mu in range(M)}
    overlaps = {mu: [] for mu in range(M)}
    for mu in range(M):

        neuron_types = neuron_types_list[mu]
        pattern = patterns[mu]
        S = state[mu].reshape(-1, 1)
        for i in range(T):
            if use_external_input and i < 5:
                S, h, overlap = sequential_update(S, pattern, neuron_types, a, c, K, beta, theta, patterns,
                                                  use_second_inhibitory=use_second_inhibitory,
                                                  use_external_input=use_external_input,
                                                  J=J)
            else:
                S, h, overlap = sequential_update(S, pattern, neuron_types, a, c, K, beta, theta, patterns,
                                                  use_second_inhibitory=use_second_inhibitory,
                                                  use_external_input=False,
                                                  J=J)
            overlaps[mu].append(overlap.item())
            S_history[mu].append(S.copy())

        accuracy = 1 - hamming_distance_V3(S[neuron_types == 1], np.array(pattern)[neuron_types == 1]) / N
        #print(np.unique(hamming_distance_test(S[neuron_types==1],np.array(pattern)[neuron_types==1])))
        #print(np.unique(hamming_distance_test(S[neuron_types==1], np.array(pattern)[neuron_types==1])))
        retrieval_errors.append(hamming_distance_V3(S[neuron_types == 1], np.array(pattern)[neuron_types == 1]))
        retrieval_counts.append(hamming_distance_V3(S[neuron_types == 1], np.array(pattern)[neuron_types == 1]) <= 0.05)
        retrieval_accuracies.append(np.mean(accuracy))

        #print(retrieval_accuracies)

    if plot:
        plot_exc_inh_external(patterns, S_history, neuron_types_list, N, overlaps)

    mean_retrieval_accuracies = np.mean(retrieval_accuracies)
    mean_retrieval_errors = np.mean(retrieval_errors)
    std_retrieval_errors = np.std(retrieval_errors)
    counts = np.sum(retrieval_counts)
    return mean_retrieval_accuracies, mean_retrieval_errors, std_retrieval_errors, counts


def capacity_study_exc_inh(N_values, loading_values, N_i_values, K_values, update_type, use_second_population=False,
                           use_external_input=False):
    """Study the capacity of Hopfield networks across different sizes and loadings."""
    results = {N: [] for N in N_values}
    print(use_second_population)

    a = 0.1
    c = 2 / (a * (1 - a))
    theta = 1.

    T = 50
    beta = 4.

    for i, N in enumerate(N_values):
        N_i = N_i_values[i]
        K = K_values[i]
        exc_prob = (N - N_i) / N
        print("N", N)
        for L in loading_values:
            print("L", L)
            M = int(L * N)
            print("M", M)
            mean_count = []
            for j in range(5):
                if update_type == "synchronous":
                    retrieval_rates, mean_error, std_error, count = run_network_exc_inh_synchronous(
                        N, M, N_i, K, a, c, theta, exc_prob, T, beta, plot=False,
                        use_second_inhibitory=use_second_population, use_external_input=use_external_input)
                    print("Synchronous", j, "", mean_error, count, std_error)
                else:
                    retrieval_rates, mean_error, std_error, count = run_network_exc_inh_sequential(
                        N, M, N_i, K, a, c, theta, exc_prob, T, beta, plot=False,
                        use_second_inhibitory=use_second_population, use_external_input=use_external_input)
                    print("Sequential", j, "", mean_error, count, std_error)
                mean_count.append(count)

            mean_count_value = np.mean(mean_count)
            std_count_value = np.std(mean_count)
            results[N].append((mean_count_value, std_count_value))

    return results


def plot_capacity_study(results, N_values, loading_values, title):
    print("Results", results)
    plt.figure(figsize=(10, 8))
    for N in N_values:
        means, stds = zip(*results[N])
        means = np.array(means) / N  # Convert to array for element-wise division
        stds = np.std(means)
        plt.errorbar(
            loading_values, means, yerr=stds,
            label=f'N = {N}',
            fmt='o-',  # Marker style
            capsize=5,  # Length of the error bar caps
            elinewidth=1.5,  # Width of the error bar lines
            markeredgewidth=1.5,  # Width of the marker edges
            capthick=1.5,  # Thickness of the caps
            markersize=6  # Size of the markers
        )

    plt.xlabel('Loading L = M/N')
    plt.ylabel('Average Retrieved Patterns / N')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def real_test():
    N = 300
    M = 3

    N_i = 80
    K = 60
    a = 0.1
    c = 2 / (a * (1 - a))
    theta = 1.0
    inh_prob = (N - N_i) / N
    T = 5
    beta = 4.

    retrivals_sequential = run_network_exc_inh_sequential(N, M, N_i, K, a, c, theta, inh_prob, T, beta, plot=True,
                                                          use_second_inhibitory=False)
    retrievals_synchronous = run_network_exc_inh_synchronous(N, M, N_i, K, a, c, theta, inh_prob, T, beta, plot=True,
                                                             use_second_inhibitory=False)
    return retrivals_sequential, retrievals_synchronous