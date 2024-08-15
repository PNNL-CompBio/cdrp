import coderdata as cd
import pandas as pd
import os
import umap
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st



# if os.path.exists('joined_dataset3'):
#     hcmi = cd.DatasetLoader('hcmi')
#     beataml = cd.DatasetLoader('beataml')
#     cptac = cd.DatasetLoader('cptac')
#     depmap = cd.DatasetLoader('broad_sanger')
#     mpnst = cd.DatasetLoader('mpnst')

#     # Join BeatAML and HCMI
#     joined_dataset0 = cd.join_datasets(beataml, hcmi)

#     # Join DepMap and CPTAC
#     joined_dataset1 = cd.join_datasets(depmap, cptac)

#     # Join Datasets
#     joined_dataset2 = cd.join_datasets(joined_dataset0,joined_dataset1)

#     # Final Join
#     joined_dataset3 = cd.join_datasets(joined_dataset2,mpnst)

#     joined_dataset3.transcriptomics= joined_dataset3.transcriptomics[["improve_sample_id", "transcriptomics", "entrez_id", "source", "study"]]




#     model_type_sample_map = dict(zip(joined_dataset3.samples['improve_sample_id'], joined_dataset3.samples['model_type']))
#     common_name_sample_map = dict(zip(joined_dataset3.samples['improve_sample_id'], joined_dataset3.samples['common_name']))
#     cancer_type_sample_map = dict(zip(joined_dataset3.samples['improve_sample_id'], joined_dataset3.samples['cancer_type']))
#     study_sample_map = dict(zip(joined_dataset3.transcriptomics['improve_sample_id'], joined_dataset3.transcriptomics['study']))


    # df['cancer_type'] = df.improve_sample_id.map(cancer_type_sample_map)
    # df['study'] = df.improve_sample_id.map(study_sample_map)
    # df['model_type'] = df.improve_sample_id.map(model_type_sample_map)

st.header('Using Proteomics data')
@st.cache_data(persist="disk")
def get_data():
    if os.path.exists('proteomics.csv'):
        df = pd.read_csv('proteomics.csv')

    embedding_t_full_data = pd.read_pickle('umap_prot.pkl')
    X = pd.read_pickle('pca_prot.pkl')

    return df, embedding_t_full_data, X
# else:    
#     df = joined_dataset3.transcriptomics.pivot_table(index='improve_sample_id', columns='entrez_id', values='transcriptomics')
#     df = df.dropna(axis=1)
#     df.to_csv('a.csv')

df, embedding_t_full_data, X = get_data()

reducer = umap.UMAP()
t_full_data = df.drop(['improve_sample_id', 'model_type', 'cancer_type', 'study'], axis=1)


# scaled_t_full_data = StandardScaler().fit_transform(t_full_data)
# embedding_t_full_data = reducer.fit_transform(scaled_t_full_data)
# embedding_t_full_data.shape



#plot umap. Hide/unhide unknowns

legend_handles = [
    mpatches.Patch(color=sns.color_palette()[0], label='Tumor'),  
    mpatches.Patch(color=sns.color_palette()[1], label='Organoid'),
    mpatches.Patch(color=sns.color_palette()[2], label='Cell Line')
    # mpatches.Patch(color=sns.color_palette()[3], label='Other')  # Uncomment this to include the Other model type legend label.
]

color_map_list = { 'study': {'Sanger & Broad Cell Lines RNASeq': 0, 'DepMap': 1, 'CPTAC3': 2, np.nan: 3},
             'model_type': {"tumor": 0, "organoid": 1, "cell line": 2} }

palette = sns.color_palette()
def get_legend_handles(color_map):
    legend_handles_tmp=[]
    for k,v in color_map.items():
        legend_handles_tmp.append(mpatches.Patch(color=palette[v], label=k))

    return legend_handles_tmp


# This is used to hide the Unknown model types.
# alphas = [0 if x == 3 else 1 for x in df.model_type.map({"tumor": 0, "organoid": 1, "cell line": 2, np.nan: 3})]

color_by = st.selectbox('Color by', ['study', 'model_type'])

if color_by == 'study':
    color_map = color_map_list['study']
elif color_by == 'model_type':
    color_map = color_map_list['model_type']

legend_handles = get_legend_handles(color_map)


study_by = st.selectbox(
    "Study by",
    ['cancer_type', 'model_type', 'study'] )

if study_by == 'cancer_type':

    cancer_type = st.selectbox(
        "Cancer type",
        df.cancer_type.unique() )

    mask = (df.cancer_type == cancer_type).values
elif study_by == 'model_type':
    model_type = st.selectbox(
        "Model type",
        df.model_type.unique() )

    mask = (df.model_type == model_type).values
elif study_by == 'study':
    study = st.selectbox(
        "study",
        df.study.unique() )

    mask = (df.study == study).values



def plot_UMAP(mask):
        
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    # model_types = df.loc[mask, 'model_type'].map({"tumor": 0, "organoid": 1, "cell line": 2, np.nan: 3})
    model_types = df.loc[mask, color_by].map(color_map)
    ax[0].scatter(
        # embedding_t_full_data[:, 0],
        # embedding_t_full_data[:, 1],
        embedding_t_full_data[mask, 0],
        embedding_t_full_data[mask, 1],
        
        # c=[sns.color_palette()[x] for x in df.model_type.map({"tumor": 0, "organoid": 1, "cell line": 2, np.nan: 3})],
        c=[sns.color_palette()[x] for x in model_types],
        # alpha=alphas,  # Apply the alpha values here
        s=3
    )
    plt.gca().set_aspect('equal', 'datalim')

    ax[0].legend(handles=legend_handles, title=color_by)

    
    ax[1].scatter(
        embedding_t_full_data[:, 0],
        embedding_t_full_data[:, 1],
        c=[sns.color_palette()[x] for x in df[color_by].map(color_map)],
        s=3
    )
    plt.gca().set_aspect('equal', 'datalim')

    ax[0].legend(handles=legend_handles, title=color_by)
    ax[1].legend(handles=legend_handles, title=color_by)


    st.pyplot(fig)



def plot_PCA(mask):

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    model_types = df.loc[mask, color_by].map(color_map)
    ax[0].scatter(
        X[mask, 0],
        X[mask, 1],
        # c=[sns.color_palette()[x] for x in df.model_type.map({"tumor": 0, "organoid": 1, "cell line": 2, np.nan: 3})],
        c=[sns.color_palette()[x] for x in model_types],
        # alpha=alphas,  # Apply the alpha values here
        s=3
    )
    plt.gca().set_aspect('equal', 'datalim')

    ax[1].scatter(
        X[:, 0],
        X[:, 1],
        c=[sns.color_palette()[x] for x in df[color_by].map(color_map)],
        s=3
    )
    ax[0].legend(handles=legend_handles, title=color_by)
    ax[1].legend(handles=legend_handles, title=color_by)


    st.pyplot(fig)




def color_by_cancer():

    top10 = list( df.cancer_type.value_counts().head(10).index )

    color_map_cancer = {v:k for k,v in dict(enumerate(top10)).items() }


    legend_handles = get_legend_handles(color_map_cancer)
    # color_map_cancer.update({np.nan:len(color_map_cancer)})

    mask = (df.cancer_type.isin(top10)).values
    model_types = df.loc[mask, 'cancer_type'].map( color_map_cancer )


    fig, ax = plt.subplots()

    ax.scatter(
        X[mask, 0],
        X[mask, 1],
        # c=[sns.color_palette()[x] for x in df.model_type.map({"tumor": 0, "organoid": 1, "cell line": 2, np.nan: 3})],
        c=[sns.color_palette()[x] for x in model_types],
        # alpha=alphas,  # Apply the alpha values here
        s=3
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(handles=legend_handles, title='Cancer Type', ncol=1, bbox_to_anchor=[0, 0])

    st.pyplot(fig)


def color_by_cancer_umap():

    top10 = list( df.cancer_type.value_counts().head(10).index )

    color_map_cancer = {v:k for k,v in dict(enumerate(top10)).items() }


    legend_handles = get_legend_handles(color_map_cancer)
    # color_map_cancer.update({np.nan:len(color_map_cancer)})

    mask = (df.cancer_type.isin(top10)).values
    model_types = df.loc[mask, 'cancer_type'].map( color_map_cancer )


    fig, ax = plt.subplots()

    ax.scatter(
        embedding_t_full_data[mask, 0],
        embedding_t_full_data[mask, 1],
        # c=[sns.color_palette()[x] for x in df.model_type.map({"tumor": 0, "organoid": 1, "cell line": 2, np.nan: 3})],
        c=[sns.color_palette()[x] for x in model_types],
        # alpha=alphas,  # Apply the alpha values here
        s=3
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(handles=legend_handles, title='Cancer Type', ncol=1, bbox_to_anchor=[0, 0])

    st.pyplot(fig)





st.subheader('UMAP')
plot_UMAP(mask)

st.subheader('PCA')
plot_PCA(mask)

st.subheader('Cancer Type clusters UMAP')
color_by_cancer_umap()

st.subheader('Cancer Type clusters PCA')
color_by_cancer()