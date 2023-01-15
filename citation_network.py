import networkx as nx
import pandas as pd
import community
import plotly.graph_objects as go
import requests

file = "data/test_citations.csv"

with open(file, 'r') as temp_f:
    # get No of columns in each line
    col_count = [len(l.split(",")) for l in temp_f.readlines()]

# Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
column_names = [i for i in range(0, max(col_count))]

# Read csv
df = pd.read_csv(file, header=None, delimiter=",", names=column_names)
df = df.rename(columns={df.columns[0]: 'case_name', df.columns[1]: 'cited_cases'})

# Rename CSV from ID's to actual names
# Load the data frame and the metadata
metadata = pd.read_csv('data/metadata.csv')

# Create a mapping from id to name_abbreviation
id_to_name = dict(zip(metadata['id'], metadata['name_abbreviation']))

# Create a new empty data frame with the same columns as df
modified_df = pd.DataFrame(columns=df.columns)

# Create a dataframe with ID and Dataframe to use later for hover information
id_name_df = pd.DataFrame(columns=['id', 'name_abbreviation'])
for i, row in df.iterrows():
    for col in df.columns:
        id = row[col]
        if pd.isna(id):
            # Skip cells with a nan value
            continue
        id_name_df = id_name_df.append({'id': id, 'name_abbreviation': id_to_name[id]}, ignore_index=True)

id_name_df = id_name_df.drop_duplicates()
id_name_df['id'] = id_name_df['id'].astype(int)

key = 'Token 519e093c5a6dbbbd322f9a0e931d7b12e4f9d471'

# Create an empty dataframe to store the results
results_df = pd.DataFrame()

# Iterate through the rows of the id_name_df
for i, row in id_name_df.iterrows():
    id = row['id']
    response = requests.get(url=f"https://api.case.law/v1/cases/{id}",
                            headers={'Authorization': key})

    data = response.json()
    # Normalize the JSON response
    df_temp = pd.json_normalize(data)

    # Append the new rows to the results_df
    results_df = pd.concat([results_df, df_temp], ignore_index=True)
    print(f"Case {id} Downloaded")

"""
Use this lambda to clean up the name if it runs too long but for now I'm just deleting it

results_df['name'] = results_df['name'].apply(
    lambda x: x[:x.find(" ", 85)] + "<br>" + x[x.find(" ", 85):] if x.find(" ", 85) != -1 else x)
results_df['name'] = results_df['name'].apply(lambda x: "Full Name: " + x)
"""
del results_df['name']

# Need to get rid of all the un-necessary information
del results_df['id']
del results_df['url']
results_df['decision_date'] = results_df['decision_date'].apply(lambda x: "Decision Date: " + x)
results_df['docket_number'] = results_df['docket_number'].apply(lambda x: "Docket Number: " + x)
del results_df['first_page']
del results_df['last_page']
del results_df['citations']
del results_df['cites_to']
del results_df['frontend_url']
del results_df['frontend_pdf_url']
del results_df['preview']
del results_df['last_updated']
del results_df['volume.url']
results_df['volume.volume_number'] = results_df['volume.volume_number'].apply(lambda x: "Volume Number: " + x)
results_df['volume.barcode'] = results_df['volume.barcode'].apply(lambda x: "Volume Barcode: " + x)
del results_df['reporter.url']
results_df['reporter.full_name'] = results_df['reporter.full_name'].apply(lambda x: "Reporter Full Name: " + x)
del results_df['reporter.id']
del results_df['court.url']
del results_df['court.name_abbreviation']
del results_df['court.slug']
del results_df['court.name']
del results_df['court.id']
del results_df['jurisdiction.id']
results_df['jurisdiction.name_long'] = results_df['jurisdiction.name_long'].apply(lambda x: "Jurisdiction Name: " + x)
del results_df['jurisdiction.url']
del results_df['jurisdiction.slug']
del results_df['jurisdiction.whitelisted']
del results_df['jurisdiction.name']
del results_df['analysis.cardinality']
del results_df['analysis.char_count']
del results_df['analysis.ocr_confidence']
del results_df['analysis.pagerank.raw']
del results_df['analysis.pagerank.percentile']
del results_df['analysis.sha256']
del results_df['analysis.simhash']
del results_df['analysis.word_count']
del results_df['analysis.random_id']
del results_df['analysis.random_bucket']
del results_df['provenance.date_added']
results_df['provenance.source'] = results_df['provenance.source'].apply(lambda x: "Source: " + x)
results_df['provenance.batch'] = results_df['provenance.batch'].apply(lambda x: "Batch: " + x)

# Iterate through the rows and columns of the data frame
for i, row in df.iterrows():
    modified_row = {}
    for col in df.columns:
        # Replace the id with the name_abbreviation
        id = row[col]
        if pd.isna(id):
            # Skip cells with a nan value
            continue
        modified_row[col] = id_to_name[id]
    # Append the modified row to the modified data frame
    modified_df = modified_df.append(modified_row, ignore_index=True)

df = modified_df

# Create an empty graph
G = nx.Graph()

# Iterate over the rows in the dataframe
for index, row in df.iterrows():
    # Extract the case name and the list of cited cases
    case_name = row['case_name']
    row = df.loc[df['case_name'] == case_name]
    cited_cases = row.values.tolist()[0]

    # Add the case to the graph as a node
    G.add_node(case_name)

    # Cut off values once nan
    cutoff_index = len(cited_cases)
    for i, val in enumerate(cited_cases):
        if pd.isna(val):
            cutoff_index = i
            break

    cited_cases_list = cited_cases[:cutoff_index]
    del cited_cases_list[0]  # Delete first value because it is the case name

    # Add an edge between the current case and each of the cited cases
    for cited_case in cited_cases_list:
        G.add_edge(case_name, cited_case)

# Compute the degree centrality of each node and add it to the dataframe with all the case information
degree_centrality = nx.degree_centrality(G)
# Compute the eigenvector centrality of each node
eigenvector_centrality = nx.eigenvector_centrality(G)
eigenvector_centrality = {k: round(v * 1000, 2) for k, v in eigenvector_centrality.items()}  # Round

results_df['degree_c'] = results_df['name_abbreviation'].map(degree_centrality).fillna("Unavailable")
results_df['degree_c'] = results_df['degree_c'].map(lambda x: "<br>Degree Centrality: " + str(x))

results_df['eigen'] = results_df['name_abbreviation'].map(eigenvector_centrality).fillna("Unavailable")
results_df['eigen'] = results_df['eigen'].map(lambda x: "Eigenvector Centrality: " + str(x))

# Print the degree centrality of the most central node
most_central_node = max(degree_centrality, key=degree_centrality.get)
sub_1 = f'the most central node is "{most_central_node}" with a degree centrality of {degree_centrality[most_central_node]:.2f} '

# Use the Louvain community detection algorithm to identify communities in the network
partition = community.best_partition(G)

# Print the number of communities identified
num_communities = len(set(partition.values()))
sub_2 = f'There are {num_communities} communities in the network'

subtitle = sub_2 + ". " + sub_1

# Compute the 3D layout of the network using the Fruchterman Reingold layout function
pos = nx.fruchterman_reingold_layout(G, dim=3)

# Extract the x, y, and z coordinates of the nodes
Xv = [pos[k][0] for k in pos.keys()]
Yv = [pos[k][1] for k in pos.keys()]
Zv = [pos[k][2] for k in pos.keys()]

annotated_case_text = {}
for key, value in pos.items():
    # Find the row in the results_df dataframe where the "name_abbreviation" column matches the key
    row = results_df[results_df['name_abbreviation'] == key]
    # Concatenate the values of the row except for the 'url' column with "<br>" in between
    new_key = key + "<br>" + "<br>".join(row.applymap(str).values[0])
    annotated_case_text[new_key] = value

# Create a Scatter 3D trace with the node coordinates and labels
trace = go.Scatter3d(x=Xv, y=Yv, z=Zv, text=[k for k in annotated_case_text.keys()],
                     hovertemplate='%{text}',
                     mode='markers',
                     marker=dict(size=6, color='#0066cc', line=dict(color='#cccccc', width=0.5)))

# Create a Figure object
fig = go.Figure(data=[trace])

# Set the layout and title of the figure
fig.update_layout(title='Citation Network (Communities)',
                  xaxis_title='x-coordinate',
                  yaxis_title='y-coordinate',
                  annotations=[dict(text=subtitle,
                                    x=0.5, y=-0.1,
                                    xref='paper', yref='paper',
                                    align='center', showarrow=False)])

# Show the figure
fig.show()

"""
# Graph To View Degree Centrality

# Create a Scatter 3D trace with the node coordinates and labels
trace = go.Scatter3d(x=Xv, y=Yv, z=Zv, text=[k for k in pos.keys()],
                     mode='markers',
                     marker=dict(size=6, color='#0066cc', line=dict(color='#cccccc', width=0.5)),
                     textposition='bottom center')

# Set the marker size to the degree centrality of each node
trace.marker.size = [degree_centrality[k] * 1000 for k in pos.keys()]

# Set the text to show the node name and degree centrality on hover
trace.text = [f'{k}: {degree_centrality[k]:.2f}' for k in pos.keys()]

# Create a Figure object
fig = go.Figure(data=[trace])

# Set the layout and title of the figure
fig.update_layout(title='Degree Centrality',
                  xaxis_title='x-coordinate',
                  yaxis_title='y-coordinate',
                  scene=dict(aspectmode='data'))

# Show the figure
fig.show()

# Graph To View Eigenvector Centrality
# This measure is based on the concept of
# "eigenvectors" in linear algebra. It assigns a score to each node based on the scores of its neighbors, with higher
# scores being given to nodes that are connected to other high-scoring nodes. A node with a high eigenvector
# centrality is one that is connected to many other important or influential nodes in the network.


# The communities identified by the layout do not necessarily correspond to any inherent structure in the data. They
# are simply a byproduct of the layout algorithm and should be interpreted with caution


# Create a Scatter 3D trace with the node coordinates and labels
trace = go.Scatter3d(x=Xv, y=Yv, z=Zv,
                     text=[f"{k}: {eigenvector_centrality[k]:.2f}" for k in pos.keys()],
                     mode='markers',
                     marker=dict(size=6, color='#0066cc', line=dict(color='#cccccc', width=0.5)))

# Create a Figure object
fig = go.Figure(data=[trace])

# Set the layout and title of the figure
fig.update_layout(title='Eigenvector Centrality',
                  xaxis_title='x-coordinate',
                  yaxis_title='y-coordinate')

# Show the figure
fig.show()
"""

print("Successful!")
