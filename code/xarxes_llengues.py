# %%
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
# %%
# lang_df = pd.read_excel('fonologia_transposat.ods',
#                         index_col=0,
#                         header=0)
# lang_df.head()
# %%
lang_df = pd.read_excel('../data/fonologia transposat_complet_MDPN.xlsx',index_col=None)
lang_df = lang_df.set_index('Language').T
lang_df.head()
lang_df_nofam = lang_df.drop(columns=['Família', 'Gènere'])
lang_df_nofam.head()
# %% do correlations of all pairs of languages
lang_df_nofam = lang_df_nofam.astype(float)
corr_mat = lang_df_nofam.T.corr('pearson')
print(corr_mat)
# %%
plt.imshow(corr_mat)
# %%
# %% weight distribution
plt.figure(figsize=(8,6))
correlation_values = corr_mat.values.flatten()
correlation_values = correlation_values[correlation_values != 1]
plt.grid(zorder=0)
sns.histplot(correlation_values, kde=False, binwidth=0.05, stat='probability',zorder=10)
plt.ylabel('$P(\\rho)$', fontsize = 25)
plt.xlabel('$\\rho$', fontsize = 25)
plt.yticks(fontsize=20)
plt.xticks(np.linspace(-1,1,5),fontsize=20)
plt.xlim(-1,1)
plt.savefig('../results/figures/correlation_distribution_langs.png',bbox_inches='tight')
# %%
langs = list(corr_mat.index)
print(langs)
# %% do percolation study
# order all pairs in decreasing order of correlation, excluding self-correlations
# create network in networkx with all the languages and no interactions.

# keep adding interactions and counting the size of the LCC and of the second one

# plot results

# Step 1: Compute correlation matrix
# Assuming df is your DataFrame containing the correlation matrix
# Replace df with your actual DataFrame containing the correlation matrix
# Make sure the index is set to languages
# Example: df = pd.DataFrame(data, index=languages, columns=languages)

# Step 2: Sort correlations in decreasing order, excluding self-correlations
correlation_pairs = []
for i, language1 in enumerate(lang_df.index):
    for j, language2 in enumerate(lang_df.index):
        if i != j:
            correlation_pairs.append((language1, language2, corr_mat[language1][language2]))

sorted_correlation_pairs = sorted(correlation_pairs, key=lambda x: x[2], reverse=True)

# Step 3: Create network in NetworkX with all languages and no interactions
G = nx.Graph()
languages = lang_df.index
N = len(languages)
G.add_nodes_from(languages)
# Initialize variables for measurement
measurement_interval = 50
sizes_lcc = []
sizes_second_largest = []
correlation_thresholds = []

# Step 4: Iterate through sorted correlations, adding interactions and measuring sizes
for idx, pair in enumerate(sorted_correlation_pairs):
    language1, language2, correlation = pair
    if idx % measurement_interval == 0:
        # Measure the network properties
        lcc_size = len(max(nx.connected_components(G), key=len))
        sizes_lcc.append(lcc_size/N)
        
        sorted_cc = sorted(nx.connected_components(G), key=len, reverse=True)
        second_largest_size = len(sorted_cc[1]) if len(sorted_cc) > 1 else 0
        sizes_second_largest.append(second_largest_size/N)
        correlation_thresholds.append(correlation)
    if correlation < 0.5:
        break  # Stop adding pairs if correlation falls below 0.5
    G.add_edge(language1, language2)

# Step 5: Plot results
x = np.array(range(0, len(sizes_lcc) * measurement_interval, measurement_interval))/(N*(N-2))
plt.figure(figsize=(8,6))
plt.plot(x, sizes_lcc, label='LCC',lw=2)
plt.plot(x, sizes_second_largest, label='Second LCC', lw=2, c='r')
plt.xlabel('Fraction of interactions added', fontsize=25)
plt.ylabel('Fraction of component', fontsize=25)
plt.xlim(0,0.1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=15)
plt.savefig('../results/figures/percolation_fraction_of_interactions_added.png',bbox_inches='tight')
# %%
plt.figure(figsize=(8,6))
plt.plot(correlation_thresholds, sizes_lcc, label='LCC', lw=2)
plt.plot(correlation_thresholds, sizes_second_largest, label='Second LCC', lw=2, c='r')
plt.xlabel('Correlation Threshold', fontsize=25)
plt.ylabel('Fraction of Component', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0.7,1.0)
plt.legend(fontsize=15)
plt.savefig('../results/figures/percolation_correlation_threshold.png', bbox_inches='tight')

# %%


# %%



# %% create network
thres = 0.75
G_lang = nx.Graph()
for index, row in corr_mat.iterrows():
    for lang in langs:
        if lang != index:
            if np.abs(row[lang]) > thres:
                G_lang.add_edge(index, lang, weight=row[lang])
# %%
print(list(lang_df.iloc[10])[1:])
# %% add family and genre to nodes in network
family_dict = dict(zip(langs,list(lang_df['Família'])))
nx.set_node_attributes(G_lang,family_dict,'Familia')
family_dict = dict(zip(langs,list(lang_df['Gènere'])))
nx.set_node_attributes(G_lang,family_dict,'Genere')
# %% save network to use community detection method that takes into account negative edges



# %%
weights = np.array([G_lang.edges[edge]['weight'] for edge in G_lang.edges()])
layout = nx.spring_layout(G_lang)
# %%
plt.figure(figsize=(20, 20))
nx.draw(G_lang,
        pos=layout,
        with_labels=True,
        edge_cmap=mpl.colormaps['Reds'],
        edge_color=weights,
        width=weights)
plt.savefig('../results/figures/lang_net_075.png', bbox_inches='tight')

# %% Get communities and plot network
from infomap import Infomap
im = Infomap()
mapping = im.add_networkx_graph(G_lang)
im.run()
# %%
communities_dict = {}
for node in im.tree:
    if node.is_leaf:
        communities_dict[im.names[node.node_id]] = node.module_id
N_comm = np.max(list(communities_dict.values()))
communities_im = N_comm*[[]]
for i in range(N_comm):
    communities_im[i] = [node for node in communities_dict.keys() if communities_dict[node]==i+1]
i = 0
# for comm in communities_im:
#     for x in comm:
#         print(descrip_dict[x],i)
#     i += 1

print(communities_im)
print(communities_dict)
# %%
nx.set_node_attributes(G_lang,communities_dict,'Community')
# %% save network to plot with cytoscape
nx.write_gml(G_lang,'../results/lang_network_075.gml')
# %%
layout_2 = nx.fruchterman_reingold_layout(G_lang)
# %% pintar la red con las comunidades
plt.figure(figsize=(20,20))
nx.draw(G_lang,
        pos=layout_2,
        edge_color=weights,
        width=np.array(weights),
        with_labels=True,
        node_color=[communities_dict[node] for node in G_lang.nodes()],
        edge_cmap=plt.cm.Reds,
        cmap=plt.cm.Accent)

plt.box(False)
plt.savefig('../results/figures/red_lang_075_comunidades.png',bbox_inches='tight')
# %%
familias = set(family_dict.values())


# Function to generate a random color
def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Dictionary of colors for each language using a dictionary comprehension
color_dict = {familia: random_color() for familia in familias}

print(color_dict)

# %% pintar la red
plt.figure(figsize=(20,20))
nx.draw(G_lang,
        pos=layout_2,
        edge_color=weights,
        width=3*np.sqrt(np.array(weights)),
        with_labels=True,
        node_color=[color_dict[family_dict[node]] for node in G_lang.nodes()],
        edge_cmap=plt.cm.Reds,
        cmap=plt.cm.Accent)

plt.box(False)
plt.savefig('../results/figures/red_lang_075_families.png',bbox_inches='tight')
# %% get average profile for each community (with std)

# %% get network família - comunidad fonològica
# nodes_top = 
# nodes_bottom = 


# %% get composition of communities in terms of Família and Gènere


# # %%
# import plotly.graph_objects as go



# # Position nodes using NetworkX's spring layout
# pos = nx.spring_layout(G_lang)

# # Create edges
# edge_x = []
# edge_y = []
# for edge in G_lang.edges():
#     x0, y0 = pos[edge[0]]
#     x1, y1 = pos[edge[1]]
#     edge_x.extend([x0, x1, None])
#     edge_y.extend([y0, y1, None])

# # Create nodes
# node_x = [pos[node][0] for node in G.nodes()]
# node_y = [pos[node][1] for node in G.nodes()]

# # Create Plotly trace for edges
# edge_trace = go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines')

# # Create Plotly trace for nodes
# node_trace = go.Scatter(
#     x=node_x, y=node_y,
#     mode='markers',
#     hoverinfo='text',
#     marker=dict(showscale=True, size=10, color=list(dict(G.degree()).values()),
#                 colorbar=dict(title='Node Connections'),
#                 colorscale='Viridis'))

# # Create the Plotly plot
# fig = go.Figure(data=[edge_trace, node_trace],
#                 layout=go.Layout(showlegend=False, hovermode='closest',
#                                  margin=dict(b=0,l=0,r=0,t=0),
#                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
# fig.show()

# %%
