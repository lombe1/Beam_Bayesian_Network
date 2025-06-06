import numpy as np
import pysmile_license
import pysmile
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Enter the file name with information about buildings and their properties and load the data
filename = 'GCadastre.csv'
df = pd.read_csv(filename)

# Function to filter the DataFrame for specified maximum and minimum values in a specified column
def filter_column_values(df, column, min_value, max_value):
    """
    Filter the DataFrame to only have values between min_value and max_value in the specified column.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    column (str): The column to filter.
    min_value (int): The minimum value for the filter.
    max_value (int): The maximum value for the filter.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    #create a new dataframe
    df_new = pd.DataFrame()
    df_new[column] = df[column].copy()
    # Filter
    filtered_df = df[(df_new[column] >= min_value) & (df_new[column] <= max_value)]
    return filtered_df

def filter_column_values_by_twice_std(df, column):
    """
    Filter the DataFrame to only have values within two standard deviations of the mean in the specified column.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    column (str): The column to filter.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Calculate the mean and standard deviation
    mean = df[column].mean()
    std = df[column].std()
    
    # Filter the DataFrame
    filtered_df = df[(df[column] >= mean - 2 * std) & (df[column] <= mean + 2 * std)]
    
    return filtered_df
# Function 

def calculate_length_width(df, area_col, perimeter_col):
    """
    Calculate the length and width from the area and perimeter columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the area and perimeter columns.
    area_col (str): The name of the column containing the area values.
    perimeter_col (str): The name of the column containing the perimeter values.

    Returns:
    pd.DataFrame: The DataFrame with two new columns for length and width.
    """
    # Calculate the length and width
    #df.loc[:, 'length'] = (df[perimeter_col] + ((df[perimeter_col]**2 - 16 * df[area_col])**0.5)) / 4
    #df.loc[:, 'width'] = (df[perimeter_col] - ((df[perimeter_col]**2 - 16 * df[area_col])**0.5)) / 4
    df_area_perimeter = df.copy()
    df_area_perimeter = df_area_perimeter.dropna(subset=[area_col, perimeter_col])
    df_length_width = pd.DataFrame()
    df_length_width['length'] = (df_area_perimeter[perimeter_col].copy() + ((df_area_perimeter[perimeter_col].copy()**2 - 16 * df_area_perimeter[area_col].copy())**0.5)) / 4
    df_length_width['width'] = (df_area_perimeter[perimeter_col].copy() - ((df_area_perimeter[perimeter_col].copy()**2 - 16 * df_area_perimeter[area_col].copy())**0.5)) / 4
    #df_all = df_length_width.dropna(subset=['length'])

    return df_length_width

# Define the round_to_one function
def round_to_one(value):
    return int(np.floor(value))
# Define the round_to_even function
def round_to_even(value):
    return int(np.round(value / 2) * 2)

def count_values_within_range(df, column):
    """
    Find the count of values for all column values that are within defined min and max values.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column (str): The name of the column to analyze.

    Returns:
    pd.Series: The count of values within the defined min and max, sorted by the column values.
    """
    # Calculate the 25th and 75th percentile values
    #min = df[column].describe()["25%"] #percentile_25
    #max = df[column].describe()["75%"] #percentile_75

   # Calculate the 95% of the values lie in this range
    min = df[column].describe()["mean"]-2*df[column].describe()["std"]  
    max = df[column].describe()["mean"]+2*df[column].describe()["std"] 
    
    # Filter the DataFrame to include only values within these percentiles
    filtered_df = df[(df[column] >= min) & (df[column] <= max)]
    
    # Count the occurrences of each value in the filtered DataFrame
    value_counts = filtered_df[column].value_counts().sort_index()

    # Convert the counts to a DataFrame
    value_counts_df = pd.DataFrame(value_counts)

    # Reset the index
    value_counts_df = value_counts_df.reset_index()

    # Rename the columns
    value_counts_df.columns = [column, 'count']
    
    return value_counts_df

def add_normalized_column_by_sum(df, new_column_name):
    """
    Add a column of normalized values to the DataFrame based on the sum of the count column.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    new_column_name (str): The name of the new column to add.

    Returns:
    pd.DataFrame: The DataFrame with the new normalized column.
    """
    # Calculate the sum of the column
    column_sum = df['count'].sum()
    
    # Normalize the values in the specified column
    #df.loc[:,new_column_name] = df['count'] / column_sum
    #df[new_column_name] = df.loc['count'] / column_sum
    df[new_column_name] = (df['count'].copy() / column_sum).round(4)
    return df

def count_values_in_column(df, column):
    """
    Count the values in a column and return a new DataFrame with the counts.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame to use.
    column (str): The column to count values in.

    Returns:
    pd.DataFrame: A DataFrame with the counts of unique values in the column.
    """
    # Count the values in the column
    value_counts = df[column].value_counts().sort_index()

    # Convert the counts to a DataFrame
    value_counts_df = pd.DataFrame(value_counts)

    # Reset the index
    value_counts_df = value_counts_df.reset_index()

    # Rename the columns
    value_counts_df.columns = [column, 'count']

    return value_counts_df

# Filter database to residential buildings only
df_GCadastre=filter_column_values(df, 'bygningsty', 100, 170)

# Filter database to buildings constructed after 1969 and before 2020 and calculate length and width
df_GCadastre_1969_2020=filter_column_values(df_GCadastre, 'Constructi', 1970, 2019)
df_decade_geom = df_GCadastre_1969_2020.copy()
df_decade_geom = calculate_length_width(df_GCadastre_1969_2020, 'Shape_Area', 'Shape_Leng')

# Remove entries with NaN values in the relevant columns
df_decade_geom = df_decade_geom.dropna(subset=['length'])

# Filter the DataFrame to remove outliers based on the length and width columns
df_decade_geom = filter_column_values_by_twice_std(df_decade_geom, 'length')
df_decade_geom = filter_column_values_by_twice_std(df_decade_geom, 'width')

df_new = pd.DataFrame()

# Round and assign each value to a category
df_new['Decade'] = df_GCadastre_1969_2020['Constructi'].apply(lambda x: (x//10)*10)
df_new['Length'] = df_decade_geom['length'].apply(round_to_even).astype(int)
df_new['Width'] = df_decade_geom['width'].apply(round_to_one).astype(int)

# Verify if there are any NaN values in the DataFrame
print("NaN values in df_new before dropping:")
print(df_new.isna().sum())

# Remove entries with empty values from the 'Length' column
df_new = df_new.dropna(subset=['Length'])

# Verify if there are any NaN values in the DataFrame after dropping
print("NaN values in df_new after dropping:")
print(df_new.isna().sum())

## Count the number of buildings in each decade
df_decade_counts = count_values_in_column(df_new, 'Decade')

## Count the number of buildings in each length value within range
df_length_counts = count_values_in_column(df_new, 'Length')

## Count the number of buildings in each width value within range
df_width_counts = count_values_in_column(df_new, 'Width')

## Find the percentage of buildings in each length and width value and add it to the DataFrame
df_length_counts_perc = add_normalized_column_by_sum(df_length_counts, 'priors')
df_width_counts_perc = add_normalized_column_by_sum(df_width_counts, 'priors')
## Find the percentage of buildings in each decade and add it to the DataFrame
df_decade_counts_perc = add_normalized_column_by_sum(df_decade_counts, 'priors')

print(df_decade_counts_perc)
print(df_length_counts_perc)
print(df_width_counts_perc)

## Send results to a csv file
df_length_counts_perc.to_csv('Length_priors.csv', index=False)
df_width_counts_perc.to_csv('Width_priors.csv', index=False)
df_decade_counts_perc.to_csv('Decade_priors.csv', index=False)

df_BN = pd.DataFrame()
# Add a prefix to each column to convert to string
# Add the string at the start of each number in the 'number' column
df_BN['Decade'] = df_new['Decade'].apply(lambda x: f"Decade_{x}")
df_BN['Length'] = df_new['Length'].apply(lambda x: f"Length_{int(x)}")
df_BN['Width'] = df_new['Width'].apply(lambda x: f"Width_{int(x)}")

df_BN.to_csv('Decade_Length_Width.csv', index=False) # Save the DataFrame to a CSV file without the index column

# Create a class to learn the structure of the network based on data (*some of the explanatory txt is taken from Pysmile Wrapper Tutorial 10)
class decadelengthwidthBN:
    def __init__(self):
       
        # Load the data
        print("Starting loading...")
        ds = pysmile.learning.DataSet()
        try:
            ds.read_file("Decade_length_width.csv")
        except pysmile.SMILEException:
            print("Dataset load failed")
            return
        print(f"Dataset has {ds.get_variable_count()} variables (columns) "
            + f"and {ds.get_record_count()} records (rows)")
        
        # Bayesian search structural learning. It is hill climbing procedure with random restarts, guided by log-likelihood scoring heuristic.
        bayes_search = pysmile.learning.BayesianSearch()
        bayes_search.set_iteration_count(100) #random restarts set to 50
        bayes_search.set_rand_seed(9876543) #random seed set to 9876543 to ensure reproducibility
        try:
            net1 = bayes_search.learn(ds)
        except pysmile.SMILEException:
            print("Bayesian Search failed")
            return
        print(f"1st Bayesian Search finished, structure score: {bayes_search.get_last_score()}")
        net1.write_file("Decade_length_width-BS1.xdsl")
        
        # Bayesian search structural learning again with a different random seed for the random number generator.
        bayes_search.set_rand_seed(3456789)
        try:
            net2 = bayes_search.learn(ds)
        except pysmile.SMILEException:
            print("Bayesian Search failed")
            return
        print(f"2nd Bayesian Search finished, structure score: {bayes_search.get_last_score()}")
        net2.write_file("Decade_length_width-BS2.xdsl")

        # Tree-augmented Naive Bayes. It is a hybrid of Naive Bayes and Bayesian network.
        """ The TAN algorithm starts with a Naive Bayes structure 
        (i.e., one in which the class variable is the only parent of all remaining, feature variables)
        and adds connections between the feature variables to account for possible dependence between them,
        conditional on the class variable. The algorithm imposes the limit of only one additional parent of every feature
        variable (additional to the class variable, which is a parent of every feature variable). We need to specify the class
        variable with its textual identifier. In this example Decade is the class variable. """

        tan = pysmile.learning.TAN()
        tan.set_rand_seed(777999)
        tan.set_class_variable_id("Decade")
        try:
            net3 = tan.learn(ds)
        except pysmile.SMILEException:
            print("TAN failed")
            return
        print("Tree-augmented Naive Bayes finished")
        net3.write_file("Decade_length_width-TAN.xdsl")
        
        # PC algorithm. It is a constraint-based structure learning algorithm.
        """ PC, which uses independences observed in data (established by
        means of classical independence tests) to infer the structure that has generated them. The output of the PC
        algorithm is a pattern, which is an adjacency matrix, and does not necessarily represent a directed acyclic graph
        (DAG). The pattern can be converted to network with the makeNetwork method, which enforces the DAG
        criterion on the pattern, and copies the variables and edges to the network. """
        
        pc = pysmile.learning.PC()
        try:
            pattern = pc.learn(ds)
        except pysmile.SMILEException:
            print("PC failed")
            return
        net4 = pattern.make_network(ds)
        print("PC finished, proceeding to parameter learning")
        #net5.write_file("Decade_length_width-PC.xdsl")
        
        
        em = pysmile.learning.EM()
        try:
            matching = ds.match_network(net4)
        except pysmile.SMILEException:
            print("Can't automatically match network with dataset")
            return
        em.set_uniformize_parameters(False)
        em.set_randomize_parameters(False)
        em.set_eq_sample_size(0)
        try:
            em.learn(ds, net4, matching)
        except pysmile.SMILEException:
            print("EM failed")
            return
        print("EM finished")
        net4.write_file("Decade_length_width-PC_EM.xdsl")
        print("Learning complete.")
        
decadelengthwidthBN() # Run the class to learn the structure of the network based on data


def plot_bayesian_network(xdsl_file, title):
    # Load the network from the .xdsl file
    net = pysmile.Network()
    net.read_file(xdsl_file)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for node_id in net.get_all_nodes():
        node_name = net.get_node_name(node_id)
        G.add_node(node_name)
    
    # Add edges to the graph
    for node_id in net.get_all_nodes():
        node_name = net.get_node_name(node_id)
        for child_id in net.get_children(node_id):
            child_name = net.get_node_name(child_id)
            G.add_edge(node_name, child_name)
    
    # Plot the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold', arrowsize=20)
    plt.title(f"Bayesian Network Structure: {title}")
    plt.show()

# Plot output
xdsl_file1 = "Decade_length_width-PC_EM.xdsl"
title1 = "PC-EM Algorithm"
plot_bayesian_network(xdsl_file1, title1)

xdsl_file2 = "Decade_length_width-TAN.xdsl"
title2 = "TAN Algorithm"
plot_bayesian_network(xdsl_file2, title2)

xdsl_file3 = "Decade_length_width-BS1.xdsl"
title3 = "Bayesian Search Algorithm 1"
plot_bayesian_network(xdsl_file3, title3)

xdsl_file4 = "Decade_length_width-BS2.xdsl"
title4 = "Bayesian Search Algorithm 2"
plot_bayesian_network(xdsl_file4, title4)

# Create a class to read the file and output priors and posteriors
class read_BN_file_send_priors_to_csv:
    def __init__(self):
        print("Starting reading file...")
        net = pysmile.Network()
        # load the network created before
        net.read_file("Decade_length_width-TAN.xdsl")

        # Print the node outcomes for each node
       
        outcomes_length = self.ls_node_outcomes(net, net.get_node("Length"))
        outcomes_width = self.ls_node_outcomes(net, net.get_node("Width"))
        outcomes_decade = self.ls_node_outcomes(net, net.get_node("Decade"))

        print(f'The outcomes for Length are: {outcomes_length}')
        print(f'The outcomes for Width are: {outcomes_width}')
        print(f'The outcomes for Decade are: {outcomes_decade}')
        print()
        # Print posterior probabilities for each node with no evidence set
        print("Posteriors with no evidence set:")
        net.update_beliefs()
        ls_posteriors_length = self.ls_posteriors(net, net.get_node("Length"))
        ls_posteriors_width = self.ls_posteriors(net, net.get_node("Width"))
        ls_posteriors_decade = self.ls_posteriors(net, net.get_node("Decade"))

        print(f'The posteriors for Decade are: {ls_posteriors_decade}')
        print(f'The posteriors for Length are: {ls_posteriors_length}')
        print(f'The posteriors for Width are: {ls_posteriors_width}')
        print()
        # Print posterior probabilities for Length given one width
        ls_posteriors_width_given_length =  self.change_evidence_and_update_node(net, "Length", "Length_8", "Width")
        print(f'The posteriors for Width given Length=Length_8 are: {ls_posteriors_width_given_length}')
        net.clear_evidence("Length")
        net.update_beliefs()
        
        self.posteriors_node1_given_node2(net, "Decade", "Width")
        net.clear_evidence("Width")
        net.update_beliefs()
        self.posteriors_node1_given_node2(net, "Length", "Width")
        net.clear_evidence("Width")
        net.update_beliefs()
        self.posteriors_node1_given_node2(net, "Width", "Length")
        net.clear_evidence("Length")
        net.update_beliefs()
        self.posteriors_node1_given_node2(net, "Decade", "Length")
        net.clear_evidence("Length")
        net.update_beliefs()
        self.posteriors_node1_given_node2(net, "Length", "Decade")
        net.clear_evidence("Decade")
        net.update_beliefs()
        self.posteriors_node1_given_node2(net, "Width", "Decade")
        net.clear_evidence("Decade")
        net.update_beliefs()


    # Function to find the posteriors for a node given another node
    def posteriors_node1_given_node2(self, net, node1, node2):
        df = pd.DataFrame()
        node1_outcomes = self.ls_node_outcomes(net, net.get_node(node1))
        node2_outcomes = self.ls_node_outcomes(net, net.get_node(node2))
        for i in range(len(node2_outcomes)):
            print()
            print(f"Posteriors for {node1} given {node2}={node2_outcomes[i]}:")
            #print()
            ls_posteriors_node1_given_node2 = self.change_evidence_and_update_node(net, node2, node2_outcomes[i], node1)
            print(ls_posteriors_node1_given_node2)
            d = {node2_outcomes[i]: ls_posteriors_node1_given_node2}
            
            
            df = pd.concat([df, pd.DataFrame(d)], axis=1)
        # Make the first column the node1 outcomes and the rest the posteriors for node1 given node2
        df.insert(0, node1, node1_outcomes)
        # Remove the string from node1_outcomes column
        df[node1] = df[node1].str.split('_').str[1].astype(int)
        # send to csv
        df.to_csv(f'{node1}_given_{node2}.csv')


    def sort_df(self, net, df, column):
        return df.reindex(df[column].str.split('_').str[1].astype(int).sort_values().index)
    
    # Create a function that returns a sorted list of the node outcomes for a given node
    def ls_node_outcomes(self, net, node_handle):
        node_outcomes = net.get_outcome_ids(node_handle)
        return sorted(node_outcomes, key=lambda x: (int(x.split('_')[1]), x.split('_')[0]))
    
    # Create a function the returns a list of posteriors for a given node
    def ls_posteriors(self, net, node_handle):
        node_id = net.get_node_id(node_handle)
        if net.is_evidence(node_handle):
            print(f"{node_id} has evidence set ({net.get_evidence_id(node_handle)})")
        else :
            posterior_list = []
            outcome_id_list = []
            posteriors = net.get_node_value(node_handle)
            for i in range(0, len(posteriors)):
                posterior_list.append(posteriors[i])
                outcome_id_list.append(net.get_outcome_id(node_handle, i))
            d = {node_id: outcome_id_list, 'Posteriors': posterior_list}
            df = pd.DataFrame(d)
            df_sorted = self.sort_df(net, df, node_id)
            #convert the sorted df to a list and return outcomes only
            return df_sorted['Posteriors'].tolist()
    # Create a function to return all posteriors for all nodes in the network as lists

    def ls_all_posteriors(self, net):
        all_posteriors = []
        for handle in net.get_all_nodes():
            ls_all_posteriors = self.ls_posteriors(net, handle)
            all_posteriors.append(ls_all_posteriors)
        return all_posteriors
    
    # Change evidence for one node and update and show posteriors for another node¨
    def change_evidence_and_update_node(self, net, node_id1, outcome_id1, node_id2):
        if outcome_id1 is not None:
            net.set_evidence(node_id1, outcome_id1)
        else:
            net.clear_evidence(node_id1)
        net.update_beliefs()
        return self.ls_posteriors(net, net.get_node(node_id2))

read_BN_file_send_priors_to_csv()


