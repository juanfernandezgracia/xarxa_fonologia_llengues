# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
# %%
def get_colors(n): 
    color = []
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    return color

# %%

lang_df = pd.read_csv('../data/fonologia_with_family_genre.csv')
lang_df.head()
# %%
lang_df_cut = lang_df.drop(index=[10,11])
lang_df_cut = lang_df_cut.set_index('Language')
lang_df_cut.head()
# %%
features = np.array(lang_df_cut.columns)[:-2]
print(features)
# %%
from sklearn.preprocessing import StandardScaler
x = lang_df_cut.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
# %%
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_lang = pd.DataFrame(x,columns=feat_cols)
normalised_lang.head()
# %%
from sklearn.decomposition import PCA
pca_lang = PCA(n_components=3)
principalComponents_lang = pca_lang.fit_transform(x)
# %%
principal_lang_Df = pd.DataFrame(data = principalComponents_lang
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
principal_lang_Df = principal_lang_Df.set_index(lang_df_cut.index)
print(principal_lang_Df)
print(lang_df_cut)
# %%
print('Explained variation per principal component: {}'.format(pca_lang.explained_variance_ratio_))
# %%
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
# plt.figure(figsize=(10,10))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel('Principal Component - 1',fontsize=20)
# plt.ylabel('Principal Component - 2',fontsize=20)
# plt.ylabel('Principal Component - 3',fontsize=20)
# plt.title("Principal Component Analysis of Language phonology Dataset",fontsize=20)
targets = lang_df_cut['Família'].unique()
colors = get_colors(len(targets))
for target, color in zip(targets,colors):
    indicesToKeep = lang_df_cut['Família'] == target
    ax.scatter(principal_lang_Df.loc[indicesToKeep, 'principal component 1']
               , principal_lang_Df.loc[indicesToKeep, 'principal component 2'], principal_lang_Df.loc[indicesToKeep, 'principal component 3'], c = color, s = 50)
plt.show()
#plt.legend(targets,prop={'size': 15})
# %%
for i in range(3):
    print('Principal component '+str(i+1))
    a = zip(features,pca_lang.components_[i])
    for x in a:
        print(x)
    print('------------------')
    print('------------------')
    print('------------------')


# %%









# %%
from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
breast_data = breast.data
breast_data.shape
breast_labels = breast.target
breast_labels.shape
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
final_breast_data.shape
breast_dataset = pd.DataFrame(final_breast_data)
breast_dataset.head()
# %%
