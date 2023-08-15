# coding: utf-8
import os
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
import backoff
import time
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.50

# Where to save the figures, and dataset locations
PROJECT_ROOT_DIR = "../"
IMAGE_PATH = os.path.join(PROJECT_ROOT_DIR, "result_images")
Feature_PATH = os.path.join(PROJECT_ROOT_DIR, "feature_extraction")

os.makedirs(IMAGE_PATH, exist_ok=True)
os.makedirs(Feature_PATH, exist_ok=True)

## function for automatically save the diagram/graph into the folder 
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGE_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
def get_all_files(directory, pattern):
    return [f for f in Path(directory).glob(pattern)]
def id2name(id):
    id = id.lower()
    # id = id.rstrip()
    if id == 'metal_non-ferrous':
        return 'non-ferrous metal'
    elif id == 'metal_ferrous':
        return 'ferrous metal'
    elif id == 'metal_ferrous_steel':
        return 'steel'
    elif id == 'metal_aluminum':
        return 'aluminum'
    elif id in ['other', 'wood', 'plastic']:
        return id
    else:
        raise f"Non-default id! {id}"
        # return 'other'
def plot_data_distribution(data, variable, title, filename, figure_size = ((7,6))):
    # Initialize the plot
    plt.figure(figsize = figure_size)
    
    # Create the countplot
    ax = sns.countplot(x=variable, data=data, palette='Set1', saturation=0.7, edgecolor='k', linewidth=1.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    # Set labels and title
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel(variable, fontsize=20, labelpad=12)
    ax.set_ylabel("Data volume", fontsize=20, labelpad=10)
    ax.tick_params(labelsize=15)
    
    # Save the figure
    plt.tight_layout()
    save_fig(f"{filename}.png")
    plt.show()

def plot_multi_label_distribution(data, variable, title, filename, figure_size = ((7,6))): 
    # Split the multi-label feature into individual labels
    all_labels = data[variable].str.split(',', expand=True).stack().reset_index(drop=True)
    
    # Initialize the plot
    plt.figure(figsize = figure_size)
    
    # Create the countplot for individual labels
    ax = sns.countplot(x=all_labels, palette='Set1', saturation=0.7, edgecolor='k', linewidth=1.5, order=all_labels.value_counts().index)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    # Set labels and title
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel(variable, fontsize=20, labelpad=12)
    ax.set_ylabel("Data volume", fontsize=20, labelpad=10)
    ax.tick_params(labelsize=15)
    
    # Save the figure
    plt.tight_layout()
    save_fig(f"{filename}.png")
    plt.show()


def plot_multi_label_distribution_selected(data, variable, title, filename, figure_size=(7,6), top_n=5): 
    # Split the multi-label feature into individual labels
    all_labels = data[variable].str.split(',', expand=True).stack().reset_index(drop=True)
    
    # Get the top_n labels based on their counts
    top_labels = all_labels.value_counts().index[:top_n]
    
    # Initialize the plot
    plt.figure(figsize=figure_size)
    
    # Create the countplot for individual labels
    ax = sns.countplot(x=all_labels, palette='Set1', saturation=0.7, edgecolor='k', linewidth=1.5, 
                       order=top_labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    # Set labels and title
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlabel(variable, fontsize=20, labelpad=12)
    ax.set_ylabel("Data volume", fontsize=20, labelpad=10)
    ax.tick_params(labelsize=15)
    
    # Save the figure
    plt.tight_layout()
    save_fig(f"{filename}.png")
    plt.show()
# Example usage (you can run this in your local environment):
# plot_multi_label_distribution(consolidated_dataset, 'assembly_industries', 
    # 'Distribution of Fusion Gallery Dataset per Industry', 'industries_distribution')
def show_confusion_matrix(y_true, y_pred, classes=None, normalize=None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.show()
# Load and inspect the contents of the provided assembly.json file again
with open("D:\Dataset\\ASME-Hackathon-2023-Autodesk\\train_data\\Fusion360GalleryDataset_23hackathon_train\\7780_6c885e81\\assembly.json", "r") as file:
    assembly_data = json.load(file)
# Display some key-value pairs from the assembly_data dictionary for inspection
dict(list(assembly_data.items())[:2])  # Display the first two entries for brevity
# Display the top-level keys of the assembly_data dictionary
assembly_data.keys()
# Display a sample from the 'bodies' section for inspection
sample_body = list(assembly_data['bodies'].values())[3]
sample_body
# Checking the type and structure of 'bodies' and 'properties'
bodies_type = type(assembly_data['bodies'])
properties_type = type(assembly_data['properties'])

# If they are dictionaries, let's see some of their keys
bodies_keys = list(assembly_data['bodies'].keys()) if isinstance(assembly_data['bodies'], dict) else None
properties_keys = list(assembly_data['properties'].keys()) if isinstance(assembly_data['properties'], dict) else None

bodies_type, bodies_keys, properties_type, properties_keys
def extract_features(data):
    # Extracting global features from 'properties'
    global_properties = data['properties']
    bounding_box = global_properties.get('bounding_box', {})
    max_point = bounding_box.get('max_point', {})
    min_point = bounding_box.get('min_point', {})
    
    # Bounding box dimensions
    x_dim = max_point.get('x', 0) - min_point.get('x', 0)
    y_dim = max_point.get('y', 0) - min_point.get('y', 0)
    z_dim = max_point.get('z', 0) - min_point.get('z', 0)
    
    # Derived features
    aspect_ratio_xy = x_dim / (y_dim + 1e-10)
    aspect_ratio_xz = x_dim / (z_dim + 1e-10)
    aspect_ratio_yz = y_dim / (z_dim + 1e-10)
    
    moments_of_inertia = global_properties.get('xyz_moments_of_inertia', {})
    
    # Global features
    global_features = {
        'bounding_box_max_x' : max_point.get('x'),
        'bounding_box_max_y' : max_point.get('y'),
        'bounding_box_max_z' : max_point.get('z'),
        'bounding_box_min_x' : min_point.get('x'),
        'bounding_box_min_y' : min_point.get('y'),
        'bounding_box_min_z' : min_point.get('z'),
        'vertex_count' : global_properties.get("vertex_count"),
        'edge_count': global_properties.get("edge_count"),
        'face_count': global_properties.get("face_count"),
        'loop_count': global_properties.get("loop_count"),
        'shell_count': global_properties.get("shell_count"),
        'body_count': global_properties.get("body_count"),
        'assembly_area': global_properties.get('area'),
        'assembly_volume': global_properties.get('volume'),
        'assembly_mass': global_properties.get('mass'),
        'assembly_density': global_properties.get('density'),
        'assembly_center_of_mass_x' : global_properties.get('center_of_mass', {}).get('x'),
        'assembly_center_of_mass_y' : global_properties.get('center_of_mass', {}).get('y'),
        'assembly_center_of_mass_z' : global_properties.get('center_of_mass', {}).get('z'),
        'assembly_likes_count' : global_properties.get('likes_count'),
        'assembly_comments_count' : global_properties.get('comments_count'),  
        'assembly_views_count' : global_properties.get('views_count'),
        'assembly_design_type': global_properties.get('design_type'),
        'assembly_categories': ",".join(global_properties.get('categories', [])),
        'assembly_industries': ",".join(global_properties.get('industries', [])),
        'bounding_box_x_dim': x_dim,
        'bounding_box_y_dim': y_dim,
        'bounding_box_z_dim': z_dim,
        'aspect_ratio_xy': aspect_ratio_xy,
        'aspect_ratio_xz': aspect_ratio_xz,
        'aspect_ratio_yz': aspect_ratio_yz,
        'moment_of_inertia_xx': moments_of_inertia.get('xx'),
        'moment_of_inertia_yy': moments_of_inertia.get('yy'),
        'moment_of_inertia_zz': moments_of_inertia.get('zz'),
        'moment_of_inertia_xy': moments_of_inertia.get('xy'),
        'moment_of_inertia_yz': moments_of_inertia.get('yz'),
        'moment_of_inertia_xz': moments_of_inertia.get('xz'),
        'vertex_to_edge_ratio': global_properties.get("vertex_count") / (global_properties.get("edge_count") + 1e-10),
        'face_to_loop_ratio': global_properties.get("face_count") / (global_properties.get("loop_count") + 1e-10)
    }
    
    # Extracting features from 'bodies'
    bodies_data = data['bodies']
    features_list = []
    for uuid, body_info in bodies_data.items():
        features = {}
        features['uuid'] = uuid
        features['name'] = body_info['name']
        features['type'] = body_info['type']
        
        # Extracting physical properties
        physical_properties = body_info.get('physical_properties', {})
        com_body_x = physical_properties.get('center_of_mass', {}).get('x')
        com_body_y = physical_properties.get('center_of_mass', {}).get('y')
        com_body_z = physical_properties.get('center_of_mass', {}).get('z')
        
        # Compute Euclidean distance between body's COM and assembly's COM
        com_assembly_x = global_features['assembly_center_of_mass_x']
        com_assembly_y = global_features['assembly_center_of_mass_y']
        com_assembly_z = global_features['assembly_center_of_mass_z']
        
        features['com_distance'] = np.sqrt((com_body_x - com_assembly_x)**2 + 
                                           (com_body_y - com_assembly_y)**2 + 
                                           (com_body_z - com_assembly_z)**2)
        
        features['center_of_mass_x'] = com_body_x
        features['center_of_mass_y'] = com_body_y
        features['center_of_mass_z'] = com_body_z
        features['body_area'] = physical_properties.get('area')
        features['body_volume'] = physical_properties.get('volume')
        features['material_category'] = body_info.get('material_category')
        
        # Calculate the volume fraction
        features['volume_fraction'] = features['body_volume'] / (global_features['assembly_volume'] + 1e-10)
        
        # Combining local and global features
        features.update(global_features)
        
        features_list.append(features)
    
    # Convert to a DataFrame
    df = pd.DataFrame(features_list)
    return df

# Extract enhanced features from the data
enhanced_features_df_updated = extract_features(assembly_data)
enhanced_features_df_updated.head()
# input_dir = r"D:\FusionGallery\Fusion360GalleryDataset_23hackathon"  
input_dir = r"D:\Dataset\ASME-Hackathon-2023-Autodesk\train_data\Fusion360GalleryDataset_23hackathon_train"
def process_assemblies(input_dir):
    # Retrieve all the assembly JSON files from the directory
    input_jsons = get_all_files(input_dir, "*/assembly.json")
    
    # This will store all the extracted features from all files
    all_features = []
    
    for input_json in tqdm(input_jsons):
        with open(input_json, "r", encoding="utf-8") as f:
            assembly_data = json.load(f)

        # Extract features using our function
        extracted_features = extract_features(assembly_data)
        
        # # Add the assembly filename as an additional feature for traceability
        # extracted_features['assembly_filename'] = str(input_json.parts[-2])
        
        all_features.append(extracted_features)
    
    # Concatenate all features into a single DataFrame
    consolidated_df = pd.concat(all_features, ignore_index=True)
    return consolidated_df

# Modify the path according to your directory structure
input_dir = r"D:\Dataset\ASME-Hackathon-2023-Autodesk\train_data\Fusion360GalleryDataset_23hackathon_train"
# To test the function, you'd run:
consolidated_dataset = process_assemblies(input_dir)
consolidated_dataset
consolidated_dataset['material_category'].unique()
consolidated_dataset['assembly_categories'].unique()
consolidated_dataset['type'].unique()
consolidated_dataset['assembly_design_type'].unique()
import random

def inspect_body_names(df, sample_size=10):
    """
    Randomly selects and returns body names from the dataset.
    
    Parameters:
    - df: DataFrame containing the assembly data
    - sample_size: Number of body names to randomly select (default is 10)
    
    Returns:
    - List of randomly selected body names
    """
    # Ensure that sample size doesn't exceed the number of available rows
    sample_size = min(sample_size, len(df))
    
    return random.sample(df['name'].tolist(), sample_size)

# Randomly inspect 10 body names from the enhanced features dataframe
random_body_names = inspect_body_names(consolidated_dataset)
random_body_names
plot_data_distribution(consolidated_dataset, 'material_category', 
                       'Distribution of Dataset per material category', 'material_category_distrubution')
plot_multi_label_distribution(consolidated_dataset, 'assembly_industries', 
                              'Distribution of Dataset per Industry', 'industries_distribution', figure_size = (10,8))
plot_multi_label_distribution_selected(consolidated_dataset, 'assembly_categories', 
                              'Distribution of Dataset per Categories', 'Category Distrubution', 
                            figure_size = (15,6), top_n=20)
import numpy as np

# Check for infinite values again
infinite_values = (consolidated_dataset == np.inf) | (consolidated_dataset == -np.inf)

# Count the infinite values in each column
infinite_counts = infinite_values.sum()

# Filter out columns with infinite values
infinite_counts = infinite_counts[infinite_counts > 0]

infinite_counts
# Identify columns with very large values
large_values_threshold = 1e+30

# Filter numeric columns only
numeric_columns = consolidated_dataset.select_dtypes(include=[np.number])

# Identify columns with very large values in the numeric subset
large_values = (numeric_columns.abs() > large_values_threshold)

# Count the large values in each column
large_value_counts = large_values.sum()

# Filter out columns with large values
large_value_columns = large_value_counts[large_value_counts > 0]

large_value_columns
abnormal_rows = consolidated_dataset[(consolidated_dataset['assembly_mass'] > large_values_threshold)]

# Display the sample indices and the abnormal values
abnormal_values = abnormal_rows[['assembly_mass']]
abnormal_values
# Remove the identified abnormal samples
consolidated_dataset_cleaned = consolidated_dataset.drop(abnormal_values.index)

# Check the shape of the new cleaned dataset
consolidated_dataset_cleaned.shape
# Filter numeric columns only
numeric_columns = consolidated_dataset_cleaned.select_dtypes(include=[np.number])

# Identify columns with very large values in the numeric subset
large_values = (numeric_columns.abs() > large_values_threshold)

# Count the large values in each column
large_value_counts = large_values.sum()

# Filter out columns with large values
large_value_columns = large_value_counts[large_value_counts > 0]

large_value_columns
consolidated_dataset_cleaned.to_csv(os.path.join(Feature_PATH, "consolidated_dataset.csv"))
consolidated_dataset_cleaned.to_csv(os.path.join(Feature_PATH, "consolidated_dataset.csv"))
def transform_features(dataset):
    # Drop uuid and name columns
    dataset = dataset.drop(columns=['uuid', 'name','type', 'assembly_design_type' ])
    
    # Convert the categorical target 'material_category' to ordinal numbers
    dataset['Material Category (Target)'] = dataset['material_category'].astype('category').cat.codes
    
    # One-Hot Encoding for 'type' and 'assembly_design_type'
    # dataset = pd.get_dummies(dataset, columns=['type', 'assembly_design_type'])
    
    # Multi-label Binarization for 'assembly_categories' and 'assembly_industries' with cleaned labels
    for col in ['assembly_categories', 'assembly_industries']:
        # Split the comma-separated string into a list of labels, handle NaNs and strip whitespace
        dataset[col] = dataset[col].str.split(',').fillna('').apply(lambda x: [item.strip() for item in x])
        mlb = MultiLabelBinarizer()
        col_encoded = mlb.fit_transform(dataset[col])
        col_df = pd.DataFrame(col_encoded, columns=mlb.classes_, index=dataset.index)
        dataset = pd.concat([dataset, col_df], axis=1)
        dataset = dataset.drop(columns=[col])
    
    return dataset

# Apply the transformations
transformed_dataset = transform_features(consolidated_dataset_cleaned)
# Drop one of the redundant 'Architecture' columns
transformed_dataset_cleaned = transformed_dataset.drop(columns=['Architecture'])
transformed_dataset_cleaned
transformed_dataset_cleaned.columns
transformed_dataset_cleaned.to_csv(os.path.join(Feature_PATH, "feature_transformed_cleaned.csv"))
df_transformed_X = transformed_dataset_cleaned.drop(columns=['Material Category (Target)', 'material_category'])
df_transformed_X_cleaned = df_transformed_X.fillna(df_transformed_X.median())
def plot_custom_style_top_correlated_features(dataset, target_column, top_n=10):
    # Compute correlations with the target column using Spearman correlation
    correlations_with_target = dataset.corr(method='spearman')[target_column].drop(target_column)
    
    # Get top N absolute correlations (including the target)
    top_correlations = correlations_with_target.abs().nlargest(top_n - 1).index
    top_correlations = top_correlations.insert(0, target_column)
    top_correlation_values = dataset[top_correlations].corr(method='spearman')
    
    # Plot the heatmap with the provided style
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        top_correlation_values,
        vmin=-1, vmax=1, center=0,
        cmap="plasma",
        square=True,
        ax=ax,
        annot=True,
        fmt='.2f',
        cbar_kws={"shrink": .9},
        linewidths=0.5, linecolor='black'
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontsize=16
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=16
    )
    ax.axhline(y=0, color='k',linewidth=4)
    ax.axhline(y=top_correlation_values.shape[1], color='k',linewidth=4)
    ax.axvline(x=0, color='k',linewidth=4)
    ax.axvline(x=top_correlation_values.shape[0], color='k',linewidth=4)
    plt.title(f"Top {top_n} Features Correlated with Target Material Category", fontsize=18, y=1.02)
    # plt.show()

# Visualize the top 10 features most correlated with 'material_category_encoded' including the target itself
# Using the modified dataset and custom style
plot_custom_style_top_correlated_features(transformed_dataset_cleaned, 'Material Category (Target)', top_n=12)
save_fig("correlation matrix - spearman")
# Define the dataset and target variable
y = transformed_dataset_cleaned['Material Category (Target)']

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model.fit(df_transformed_X_cleaned, y)

# Extract feature importances
feat_importances = pd.Series(model.feature_importances_, index=df_transformed_X_cleaned.columns)


# Plotting the feature importances
fig = plt.figure(figsize=(4,3), dpi=600)
ax = plt.gca()
widths = 2
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(widths)
    
color = ['darkred','yellow', 'blue','green', 'steelblue','orange', 'olive', 'midnightblue', 'darkkhaki', 
         "lightblue", "purple", "darkblue", "gray", "black", "lightyellow"]

feat_importances.nlargest(15).plot(kind='barh', color=color, edgecolor='k', linewidth=0.5)
plt.title("RF Feature Importance", fontsize=12, y=1.02)
plt.xlabel('Relative Importance', fontsize=12, labelpad=2)
plt.tick_params(axis='both', labelsize=8, pad=3)

# Displaying the top 15 feature importances
top_feature_importances = feat_importances.nlargest(15)
top_feature_importances
save_fig("RF feature importance ranking")
from sklearn.preprocessing import StandardScaler

df_normalized=(df_transformed_X_cleaned.select_dtypes(include=np.number) - df_transformed_X_cleaned.mean(numeric_only = True)) / df_transformed_X_cleaned.std(numeric_only = True)
df_normalized.head()
# dimensionality reduction with 18 features
pca= PCA(n_components=15)
pca.fit(df_normalized)
print(pca.explained_variance_ratio_)
# Determine explained variance using explained_variance_ration_ attribute
exp_var_pca = pca.explained_variance_ratio_

# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.

cum_sum_eigenvalues = np.cumsum(exp_var_pca)
#
# Create the visualization plot

plt.figure(figsize=(8, 5)) 
plt.grid(linestyle='--', linewidth=1, alpha=0.7)
plt.title("PCA feature analysis", fontsize = 20, pad = 12)
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=1.0, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio', fontsize = 18, labelpad=12)
plt.xlabel('Principal component index', fontsize = 18, labelpad=12)
plt.legend(loc='best', fontsize="large")
plt.tight_layout()
# plt.show()
save_fig('PCA_all.png')
import matplotlib.patches
    
def two_d_pca_projection (df_normalized, name):
    pca_2 = PCA(n_components=2)
    pca_2.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_2 = pca_2.transform(df_normalized)

    plt.figure(figsize=(6, 4))
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

    xdata = data_pca_2[:, 0]
    ydata = data_pca_2[:, 1]

    plt.scatter(xdata, ydata, c=colors, alpha=0.8, s=10)
    plt.xlabel('Principal component 1',fontsize=16, labelpad=13)
    plt.ylabel('Principal component 2',fontsize=16, labelpad=13)
    # plt.title('PCA Projection ' + name,fontsize=20, pad=12)
    # plt.gca().set(xlabel='Principle component 1', ylabel='Principle component 2', title='PCA dimensionality reduction', set_fontsize=12)
    plt.legend(handles=handles, loc='upper left', fontsize='large') # title='Color'
    save_fig("PCA_2D_" + name)

def three_d_pca_projection (df_normalized, name):
    pca_3 = PCA(n_components=3)
    pca_3.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_3 = pca_3.transform(df_normalized)

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')

    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]


    # Data for three-dimensional scattered points
    xdata = data_pca_3[:, 1]
    ydata = data_pca_3[:, 2]
    zdata = data_pca_3[:, 0]
    # ax.scatter3D(xdata, ydata, zdata,
    #              c=df_denoised_normalized["laser power"], edgecolor='none', alpha=0.9, s=40,
    #              cmap=plt.cm.get_cmap('Set1', 4)); # cmap='Greens'

    ax.scatter3D(xdata, ydata, zdata,edgecolor='none', alpha=0.8, s=20,
                 c=colors); # cmap='Greens'

    ax.set_xlabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_ylabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_zlabel('PCA 1',fontsize=16, labelpad=13)
    # ax.set_title("PCA 3D projection " + name, fontsize = 20, pad = 5)
    # plt.colorbar();
    plt.legend(handles=handles, loc='best', fontsize='Large') #; upper left # title='Color'
    save_fig("PCA_3D_" + name)
three_d_pca_projection(df_normalized, "all_feature")
import matplotlib.patches
    
def two_d_pca_projection (df_normalized, name):
    pca_2 = PCA(n_components=2)
    pca_2.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_2 = pca_2.transform(df_normalized)

    plt.figure(figsize=(6, 4))
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

    xdata = data_pca_2[:, 0]
    ydata = data_pca_2[:, 1]

    plt.scatter(xdata, ydata, c=colors, alpha=0.8, s=10)
    plt.xlabel('Principal component 1',fontsize=16, labelpad=13)
    plt.ylabel('Principal component 2',fontsize=16, labelpad=13)
    # plt.title('PCA Projection ' + name,fontsize=20, pad=12)
    # plt.gca().set(xlabel='Principle component 1', ylabel='Principle component 2', title='PCA dimensionality reduction', set_fontsize=12)
    plt.legend(handles=handles, loc='upper left', fontsize='large') # title='Color'
    save_fig("PCA_2D_" + name)

def three_d_pca_projection (df_normalized, name):
    pca_3 = PCA(n_components=3)
    pca_3.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_3 = pca_3.transform(df_normalized)

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')

    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]


    # Data for three-dimensional scattered points
    xdata = data_pca_3[:, 1]
    ydata = data_pca_3[:, 2]
    zdata = data_pca_3[:, 0]
    # ax.scatter3D(xdata, ydata, zdata,
    #              c=df_denoised_normalized["laser power"], edgecolor='none', alpha=0.9, s=40,
    #              cmap=plt.cm.get_cmap('Set1', 4)); # cmap='Greens'

    ax.scatter3D(xdata, ydata, zdata,edgecolor='none', alpha=0.8, s=20,
                 c=colors); # cmap='Greens'

    ax.set_xlabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_ylabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_zlabel('PCA 1',fontsize=16, labelpad=13)
    # ax.set_title("PCA 3D projection " + name, fontsize = 20, pad = 5)
    # plt.colorbar();
    plt.legend(handles=handles, loc='best', fontsize=14) #; upper left # title='Color'
    save_fig("PCA_3D_" + name)
three_d_pca_projection(df_normalized, "all_feature")
import matplotlib.patches
    
def two_d_pca_projection (df_normalized, name):
    pca_2 = PCA(n_components=2)
    pca_2.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_2 = pca_2.transform(df_normalized)

    plt.figure(figsize=(6, 4))
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

    xdata = data_pca_2[:, 0]
    ydata = data_pca_2[:, 1]

    plt.scatter(xdata, ydata, c=colors, alpha=0.8, s=10)
    plt.xlabel('Principal component 1',fontsize=16, labelpad=13)
    plt.ylabel('Principal component 2',fontsize=16, labelpad=13)
    # plt.title('PCA Projection ' + name,fontsize=20, pad=12)
    # plt.gca().set(xlabel='Principle component 1', ylabel='Principle component 2', title='PCA dimensionality reduction', set_fontsize=12)
    plt.legend(handles=handles, loc='upper left', fontsize='large') # title='Color'
    save_fig("PCA_2D_" + name)

def three_d_pca_projection (df_normalized, name):
    pca_3 = PCA(n_components=3)
    pca_3.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_3 = pca_3.transform(df_normalized)

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')

    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]


    # Data for three-dimensional scattered points
    xdata = data_pca_3[:, 1]
    ydata = data_pca_3[:, 2]
    zdata = data_pca_3[:, 0]
    # ax.scatter3D(xdata, ydata, zdata,
    #              c=df_denoised_normalized["laser power"], edgecolor='none', alpha=0.9, s=40,
    #              cmap=plt.cm.get_cmap('Set1', 4)); # cmap='Greens'

    ax.scatter3D(xdata, ydata, zdata,edgecolor='none', alpha=0.8, s=20,
                 c=colors); # cmap='Greens'

    ax.set_xlabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_ylabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_zlabel('PCA 1',fontsize=16, labelpad=13)
    # ax.set_title("PCA 3D projection " + name, fontsize = 20, pad = 5)
    # plt.colorbar();
    plt.legend(handles=handles, loc='best', fontsize=10) #; upper left # title='Color'
    save_fig("PCA_3D_" + name)
three_d_pca_projection(df_normalized, "all_feature")
import matplotlib.patches
    
def two_d_pca_projection (df_normalized, name):
    pca_2 = PCA(n_components=2)
    pca_2.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_2 = pca_2.transform(df_normalized)

    plt.figure(figsize=(6, 4))
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

    xdata = data_pca_2[:, 0]
    ydata = data_pca_2[:, 1]

    plt.scatter(xdata, ydata, c=colors, alpha=0.8, s=10)
    plt.xlabel('Principal component 1',fontsize=16, labelpad=13)
    plt.ylabel('Principal component 2',fontsize=16, labelpad=13)
    # plt.title('PCA Projection ' + name,fontsize=20, pad=12)
    # plt.gca().set(xlabel='Principle component 1', ylabel='Principle component 2', title='PCA dimensionality reduction', set_fontsize=12)
    plt.legend(handles=handles, loc='upper left', fontsize='large') # title='Color'
    save_fig("PCA_2D_" + name)

def three_d_pca_projection (df_normalized, name):
    pca_3 = PCA(n_components=3)
    pca_3.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_3 = pca_3.transform(df_normalized)

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')

    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]


    # Data for three-dimensional scattered points
    xdata = data_pca_3[:, 1]
    ydata = data_pca_3[:, 2]
    zdata = data_pca_3[:, 0]
    # ax.scatter3D(xdata, ydata, zdata,
    #              c=df_denoised_normalized["laser power"], edgecolor='none', alpha=0.9, s=40,
    #              cmap=plt.cm.get_cmap('Set1', 4)); # cmap='Greens'

    ax.scatter3D(xdata, ydata, zdata,edgecolor='none', alpha=0.8, s=30,
                 c=colors); # cmap='Greens'

    ax.set_xlabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_ylabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_zlabel('PCA 1',fontsize=16, labelpad=13)
    # ax.set_title("PCA 3D projection " + name, fontsize = 20, pad = 5)
    # plt.colorbar();
    plt.legend(handles=handles, loc='best', fontsize=10) #; upper left # title='Color'
    save_fig("PCA_3D_" + name)
import matplotlib.patches
    
def two_d_pca_projection (df_normalized, name):
    pca_2 = PCA(n_components=2)
    pca_2.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_2 = pca_2.transform(df_normalized)

    plt.figure(figsize=(6, 4))
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

    xdata = data_pca_2[:, 0]
    ydata = data_pca_2[:, 1]

    plt.scatter(xdata, ydata, c=colors, alpha=0.8, s=10)
    plt.xlabel('Principal component 1',fontsize=16, labelpad=13)
    plt.ylabel('Principal component 2',fontsize=16, labelpad=13)
    # plt.title('PCA Projection ' + name,fontsize=20, pad=12)
    # plt.gca().set(xlabel='Principle component 1', ylabel='Principle component 2', title='PCA dimensionality reduction', set_fontsize=12)
    plt.legend(handles=handles, loc='upper left', fontsize='large') # title='Color'
    save_fig("PCA_2D_" + name)

def three_d_pca_projection (df_normalized, name):
    pca_3 = PCA(n_components=3)
    pca_3.fit(df_normalized)
    # dimensionality reduction -> output is reduced feature space
    data_pca_3 = pca_3.transform(df_normalized)

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')

    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [plt.cm.tab10(i) for i in levels] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]


    # Data for three-dimensional scattered points
    xdata = data_pca_3[:, 1]
    ydata = data_pca_3[:, 2]
    zdata = data_pca_3[:, 0]
    # ax.scatter3D(xdata, ydata, zdata,
    #              c=df_denoised_normalized["laser power"], edgecolor='none', alpha=0.9, s=40,
    #              cmap=plt.cm.get_cmap('Set1', 4)); # cmap='Greens'

    ax.scatter3D(xdata, ydata, zdata,edgecolor='none', alpha=0.8, s=30,
                 c=colors); # cmap='Greens'

    ax.set_xlabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_ylabel('PCA 1',fontsize=16, labelpad=13)
    ax.set_zlabel('PCA 1',fontsize=16, labelpad=13)
    # ax.set_title("PCA 3D projection " + name, fontsize = 20, pad = 5)
    # plt.colorbar();
    plt.legend(handles=handles, loc='best', fontsize=11) #; upper left # title='Color'
    save_fig("PCA_3D_" + name)
three_d_pca_projection(df_normalized, "all_feature")
X = df_transformed_X_cleaned.to_numpy()
y = transformed_dataset_cleaned['Material Category (Target)'].to_numpy()
y.shape
X.shape
# 2D visualization function
# color_list = [plt.cm.tab10(2), plt.cm.tab10(4),plt.cm.tab10(3), plt.cm.tab10(0)]
# color_list = ["#0fa14a", "#7a7c7f","#c24553", "#faaf42"]
# color_list = ["#0fa14a", "#7a7c7f", plt.cm.tab10(3), "#faaf42"]
color_list = [ "#faaf42", "#7a7c7f", plt.cm.tab10(3), "#0fa14a","#0fa14a", 
              "#c24553",plt.cm.tab10(3), plt.cm.tab10(0), plt.cm.tab10(4)]

def plot_embedding_2d(X, method_name, title=None):
    #MinMax scaled to [0,1]
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    # After dimension reduction, each point is:（X[i, 0], X[i, 1]）
    plt.figure(figsize=(6, 5))
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [color_list[i] for i in levels] # using the "plt.cm.tab10(i)" colormap
    handles = [matplotlib.patches.Patch(color=color_list[i], label=c) for i, c in enumerate(categories)]
    
    xdata = X[:, 0]
    ydata = X[:, 1]
    plt.scatter(xdata, ydata, c=colors, alpha=0.9, s=10, linewidths=0.1, edgecolors='k')
    xlabel = method_name + "-1"
    ylabel = method_name + "-2"
    plt.xlabel(xlabel,fontsize=20, labelpad=13)
    plt.ylabel(ylabel,fontsize=20, labelpad=13)
    plt.legend(handles=handles, loc='best', fontsize='large') #; upper left # title='Color'
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if title is not None:
        plt.title(title,fontsize=20, pad=12)

# 3D visualization function
def plot_embedding_3d(X, method_name, title=None):
    #MinMax scaled to [0,1]
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    # After dimension reduction, each point is:（X[i, 0], X[i, 1], X[i, 2]）
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')
    
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [color_list[i] for i in levels] # using the "plt.cm.tab10(i)" colormap
    handles = [matplotlib.patches.Patch(color=color_list[i], label=c) for i, c in enumerate(categories)]
    
    xdata = X[:, 0]
    ydata = X[:, 1]
    zdata = X[:, 2]
    
    ax.scatter3D(xdata, ydata, zdata,linewidths=0.1, edgecolor='k', alpha=0.9, s=14, c=colors); # cmap='Greens'
    xlabel = method_name + "-1"
    ylabel = method_name + "-2"
    zlable = method_name + "-3"

    ax.set_xlabel(xlabel,fontsize=16, labelpad=13)
    ax.set_ylabel(ylabel,fontsize=16, labelpad=13)
    ax.set_zlabel(zlable,fontsize=16, labelpad=13)
    if title is not None:
        ax.set_title(title,fontsize=20, pad = 5)
    plt.legend(handles=handles, loc='best', fontsize='large') #; upper left # title='Color'
n_neighbors=10

# # # Random Projection
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=3, random_state=42)
# X_projected = rp.fit_transform(X)
# plot_embedding_2d(X_projected, "Random Projection", title="Random Projection")
# save_fig("Random Projection")
# plot_embedding_3d(X_projected,'Random Projection',"Random Projection 3D")
# save_fig("Random Projection-3D")

########------------------------------------#########
########---------------PCA------------------#########
########------------------------------------#########
print("Computing PCA projection")
t0 = time.time()
X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X)
plot_embedding_2d(X_pca[:,0:2], 'PCA', title="PCA 2D")
save_fig("PCA_2D")
plot_embedding_3d(X_pca,'PCA', title ="PCA 3D (time %.2fs)" %(time.time() - t0))
save_fig("PCA_3D")


# ########------------------------------------#########
# ########---------------LDA------------------#########
# ########------------------------------------#########
print("Computing LDA projection")
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
t0 = time.time()
X_lda = LDA(n_components=3).fit_transform(X2, y)
plot_embedding_2d(X_lda[:,0:2],'LDA',  "LDA 2D" )
save_fig("LDA_2D")
plot_embedding_3d(X_lda,'LDA', "LDA 3D")
save_fig("LDA_3D")

# ########------------------------------------#########
# ########---------------Isomap---------------#########
# # ########------------------------------------#########
# print("Computing Isomap embedding")
# t0 = time.time()
# X_iso = manifold.Isomap(n_neighbors = n_neighbors, n_components=3).fit_transform(X)
# print("Done.")
# plot_embedding_2d(X_iso,"Isomap", "Isomap 2D")
# save_fig("Isomap_2D")
# plot_embedding_3d(X_iso,'Isomap', "Isomap 3D")
# save_fig("Isomap_3D")
# ########------------------------------------#########
# ########-----------------MDS----------------#########
# ########------------------------------------#########
print("Computing MDS embedding")
clf = manifold.MDS(n_components=3, n_init=1, max_iter=100)
t0 = time.time()
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
plot_embedding_2d(X_mds,'MDS', "MDS (time %.2fs)" %(time.time() - t0))
save_fig("MDS-2D")
plot_embedding_3d(X_mds,'MDS', "MDS 3D")
save_fig("MDS_3D")


# ########------------------------------------#########
# ########-------------Random Trees-----------#########
# ########------------------------------------#########
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,max_depth=5)
t0 = time.time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=3)
X_reduced = pca.fit_transform(X_transformed)
plot_embedding_2d(X_reduced,'Random Trees', "Random Trees (time %.2fs)" %(time.time() - t0))
save_fig("Random_trees-2D")
plot_embedding_3d(X_reduced,'Random Trees', "Random Trees 3D")
save_fig("Random Trees_3D")
# ########------------------------------------#########
# ########-----------------MDS----------------#########
# ########------------------------------------#########
# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=3, n_init=1, max_iter=100)
# t0 = time.time()
# X_mds = clf.fit_transform(X)
# print("Done. Stress: %f" % clf.stress_)
# plot_embedding_2d(X_mds,'MDS', "MDS (time %.2fs)" %(time.time() - t0))
# save_fig("MDS-2D")
# plot_embedding_3d(X_mds,'MDS', "MDS 3D")
# save_fig("MDS_3D")


# ########------------------------------------#########
# ########-------------Random Trees-----------#########
# ########------------------------------------#########
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,max_depth=5)
t0 = time.time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=3)
X_reduced = pca.fit_transform(X_transformed)
plot_embedding_2d(X_reduced,'Random Trees', "Random Trees (time %.2fs)" %(time.time() - t0))
save_fig("Random_trees-2D")
plot_embedding_3d(X_reduced,'Random Trees', "Random Trees 3D")
save_fig("Random Trees_3D")
########------------------------------------#########
########-------------------TSNE--------------#########
########------------------------------------#########
# t-SNE
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=3, perplexity = 200, n_iter=5000, init='pca', random_state=0, n_jobs=-1)
t0 = time.time()
X_tsne = tsne.fit_transform(X)
print (X_tsne.shape)
plot_embedding_2d(X_tsne[:,0:2],'TSNE', "t-SNE 2D")
save_fig("TSNE-2D")
plot_embedding_3d(X_tsne,'TSNE',"t-SNE 3D (time %.2fs)" %(time.time() - t0))
save_fig("TSNE-3D")
X = df_normalized.to_numpy()
y = transformed_dataset_cleaned['Material Category (Target)'].to_numpy()
y.shape
X.shape
# 2D visualization function
# color_list = [plt.cm.tab10(2), plt.cm.tab10(4),plt.cm.tab10(3), plt.cm.tab10(0)]
# color_list = ["#0fa14a", "#7a7c7f","#c24553", "#faaf42"]
# color_list = ["#0fa14a", "#7a7c7f", plt.cm.tab10(3), "#faaf42"]
color_list = [ "#faaf42", "#7a7c7f", plt.cm.tab10(3), "#0fa14a","#0fa14a", 
              "#c24553",plt.cm.tab10(3), plt.cm.tab10(0), plt.cm.tab10(4)]

def plot_embedding_2d(X, method_name, title=None):
    #MinMax scaled to [0,1]
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    # After dimension reduction, each point is:（X[i, 0], X[i, 1]）
    plt.figure(figsize=(6, 5))
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [color_list[i] for i in levels] # using the "plt.cm.tab10(i)" colormap
    handles = [matplotlib.patches.Patch(color=color_list[i], label=c) for i, c in enumerate(categories)]
    
    xdata = X[:, 0]
    ydata = X[:, 1]
    plt.scatter(xdata, ydata, c=colors, alpha=0.9, s=10, linewidths=0.1, edgecolors='k')
    xlabel = method_name + "-1"
    ylabel = method_name + "-2"
    plt.xlabel(xlabel,fontsize=20, labelpad=13)
    plt.ylabel(ylabel,fontsize=20, labelpad=13)
    plt.legend(handles=handles, loc='best', fontsize='large') #; upper left # title='Color'
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if title is not None:
        plt.title(title,fontsize=20, pad=12)

# 3D visualization function
def plot_embedding_3d(X, method_name, title=None):
    #MinMax scaled to [0,1]
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    # After dimension reduction, each point is:（X[i, 0], X[i, 1], X[i, 2]）
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')
    
    levels, categories = pd.factorize(transformed_dataset_cleaned['material_category'])
    colors = [color_list[i] for i in levels] # using the "plt.cm.tab10(i)" colormap
    handles = [matplotlib.patches.Patch(color=color_list[i], label=c) for i, c in enumerate(categories)]
    
    xdata = X[:, 0]
    ydata = X[:, 1]
    zdata = X[:, 2]
    
    ax.scatter3D(xdata, ydata, zdata,linewidths=0.1, edgecolor='k', alpha=0.9, s=14, c=colors); # cmap='Greens'
    xlabel = method_name + "-1"
    ylabel = method_name + "-2"
    zlable = method_name + "-3"

    ax.set_xlabel(xlabel,fontsize=16, labelpad=13)
    ax.set_ylabel(ylabel,fontsize=16, labelpad=13)
    ax.set_zlabel(zlable,fontsize=16, labelpad=13)
    if title is not None:
        ax.set_title(title,fontsize=20, pad = 5)
    plt.legend(handles=handles, loc='best', fontsize='large') #; upper left # title='Color'
n_neighbors=10

# # # Random Projection
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=3, random_state=42)
# X_projected = rp.fit_transform(X)
# plot_embedding_2d(X_projected, "Random Projection", title="Random Projection")
# save_fig("Random Projection")
# plot_embedding_3d(X_projected,'Random Projection',"Random Projection 3D")
# save_fig("Random Projection-3D")

########------------------------------------#########
########---------------PCA------------------#########
########------------------------------------#########
print("Computing PCA projection")
t0 = time.time()
X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X)
plot_embedding_2d(X_pca[:,0:2], 'PCA', title="PCA 2D")
save_fig("PCA_2D")
plot_embedding_3d(X_pca,'PCA', title ="PCA 3D (time %.2fs)" %(time.time() - t0))
save_fig("PCA_3D")


# ########------------------------------------#########
# ########---------------LDA------------------#########
# ########------------------------------------#########
print("Computing LDA projection")
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
t0 = time.time()
X_lda = LDA(n_components=3).fit_transform(X2, y)
plot_embedding_2d(X_lda[:,0:2],'LDA',  "LDA 2D" )
save_fig("LDA_2D")
plot_embedding_3d(X_lda,'LDA', "LDA 3D")
save_fig("LDA_3D")

# ########------------------------------------#########
# ########---------------Isomap---------------#########
# # ########------------------------------------#########
# print("Computing Isomap embedding")
# t0 = time.time()
# X_iso = manifold.Isomap(n_neighbors = n_neighbors, n_components=3).fit_transform(X)
# print("Done.")
# plot_embedding_2d(X_iso,"Isomap", "Isomap 2D")
# save_fig("Isomap_2D")
# plot_embedding_3d(X_iso,'Isomap', "Isomap 3D")
# save_fig("Isomap_3D")
# ########------------------------------------#########
# ########-----------------MDS----------------#########
# ########------------------------------------#########
# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=3, n_init=1, max_iter=100)
# t0 = time.time()
# X_mds = clf.fit_transform(X)
# print("Done. Stress: %f" % clf.stress_)
# plot_embedding_2d(X_mds,'MDS', "MDS (time %.2fs)" %(time.time() - t0))
# save_fig("MDS-2D")
# plot_embedding_3d(X_mds,'MDS', "MDS 3D")
# save_fig("MDS_3D")


# ########------------------------------------#########
# ########-------------Random Trees-----------#########
# ########------------------------------------#########
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,max_depth=5)
t0 = time.time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=3)
X_reduced = pca.fit_transform(X_transformed)
plot_embedding_2d(X_reduced,'Random Trees', "Random Trees (time %.2fs)" %(time.time() - t0))
save_fig("Random_trees-2D")
plot_embedding_3d(X_reduced,'Random Trees', "Random Trees 3D")
save_fig("Random Trees_3D")
########------------------------------------#########
########-------------------TSNE--------------#########
########------------------------------------#########
# t-SNE
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=3, perplexity = 200, n_iter=5000, init='pca', random_state=0, n_jobs=-1)
t0 = time.time()
X_tsne = tsne.fit_transform(X)
print (X_tsne.shape)
plot_embedding_2d(X_tsne[:,0:2],'TSNE', "t-SNE 2D")
save_fig("TSNE-2D")
plot_embedding_3d(X_tsne,'TSNE',"t-SNE 3D (time %.2fs)" %(time.time() - t0))
save_fig("TSNE-3D")
# ########------------------------------------#########
# ########-------------------TSNE--------------#########
# ########------------------------------------#########
# # t-SNE
# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=3, perplexity = 200, n_iter=5000, init='pca', random_state=0, n_jobs=-1)
# t0 = time.time()
# X_tsne = tsne.fit_transform(X)
# print (X_tsne.shape)
# plot_embedding_2d(X_tsne[:,0:2],'TSNE', "t-SNE 2D")
# save_fig("TSNE-2D")
# plot_embedding_3d(X_tsne,'TSNE',"t-SNE 3D (time %.2fs)" %(time.time() - t0))
# save_fig("TSNE-3D")
consolidated_dataset['assembly_industries'].unique()
plot_data_distribution(consolidated_dataset_cleaned, 'material_category', 
                       'Distribution of Dataset per material category', 'material_category_distrubution')
plot_multi_label_distribution(consolidated_dataset_cleaned, 'assembly_industries', 
                              'Distribution of Dataset per Industry', 'industries_distribution', figure_size = (10,8))
plot_multi_label_distribution_selected(consolidated_dataset_cleaned, 'assembly_categories', 
                              'Distribution of Dataset per Categories', 'Category Distrubution', 
                            figure_size = (15,6), top_n=20)
get_ipython().run_line_magic('store', 'transformed_dataset_cleaned')
def transform_features(dataset):
    # Drop uuid and name columns
    dataset = dataset.drop(columns=['uuid', 'name','type', 'assembly_design_type' ])
    
    # Convert the categorical target 'material_category' to ordinal numbers
    # Initialize the encoder
    encoder = LabelEncoder()
    # Fit the encoder on the 'material_category' column
    encoder.fit(dataset['material_category'])
    dataset['Material Category (Target)'] = encoder.transform(dataset['material_category'])
    
    # One-Hot Encoding for 'type' and 'assembly_design_type'
    # dataset = pd.get_dummies(dataset, columns=['type', 'assembly_design_type'])
    
    # Multi-label Binarization for 'assembly_categories' and 'assembly_industries' with cleaned labels
    for col in ['assembly_categories', 'assembly_industries']:
        # Split the comma-separated string into a list of labels, handle NaNs and strip whitespace
        dataset[col] = dataset[col].str.split(',').fillna('').apply(lambda x: [item.strip() for item in x])
        mlb = MultiLabelBinarizer()
        col_encoded = mlb.fit_transform(dataset[col])
        col_df = pd.DataFrame(col_encoded, columns=mlb.classes_, index=dataset.index)
        dataset = pd.concat([dataset, col_df], axis=1)
        dataset = dataset.drop(columns=[col])
    
    return dataset, encoder

# Apply the transformations
encoder, transformed_dataset = transform_features(consolidated_dataset_cleaned)
# Drop one of the redundant 'Architecture' columns
transformed_dataset_cleaned = transformed_dataset.drop(columns=['Architecture'])
transformed_dataset_cleaned
import os
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
import backoff
import time
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
def transform_features(dataset):
    # Drop uuid and name columns
    dataset = dataset.drop(columns=['uuid', 'name','type', 'assembly_design_type' ])
    
    # Convert the categorical target 'material_category' to ordinal numbers
    # Initialize the encoder
    encoder = LabelEncoder()
    # Fit the encoder on the 'material_category' column
    encoder.fit(dataset['material_category'])
    dataset['Material Category (Target)'] = encoder.transform(dataset['material_category'])
    
    # One-Hot Encoding for 'type' and 'assembly_design_type'
    # dataset = pd.get_dummies(dataset, columns=['type', 'assembly_design_type'])
    
    # Multi-label Binarization for 'assembly_categories' and 'assembly_industries' with cleaned labels
    for col in ['assembly_categories', 'assembly_industries']:
        # Split the comma-separated string into a list of labels, handle NaNs and strip whitespace
        dataset[col] = dataset[col].str.split(',').fillna('').apply(lambda x: [item.strip() for item in x])
        mlb = MultiLabelBinarizer()
        col_encoded = mlb.fit_transform(dataset[col])
        col_df = pd.DataFrame(col_encoded, columns=mlb.classes_, index=dataset.index)
        dataset = pd.concat([dataset, col_df], axis=1)
        dataset = dataset.drop(columns=[col])
    
    return dataset, encoder

# Apply the transformations
encoder, transformed_dataset = transform_features(consolidated_dataset_cleaned)
# Drop one of the redundant 'Architecture' columns
transformed_dataset_cleaned = transformed_dataset.drop(columns=['Architecture'])
transformed_dataset_cleaned
def transform_features(dataset):
    # Drop uuid and name columns
    dataset = dataset.drop(columns=['uuid', 'name','type', 'assembly_design_type' ])
    
    # Convert the categorical target 'material_category' to ordinal numbers
    # Initialize the encoder
    encoder = LabelEncoder()
    # Fit the encoder on the 'material_category' column
    encoder.fit(dataset['material_category'])
    dataset['Material Category (Target)'] = encoder.transform(dataset['material_category'])
    
    # One-Hot Encoding for 'type' and 'assembly_design_type'
    # dataset = pd.get_dummies(dataset, columns=['type', 'assembly_design_type'])
    
    # Multi-label Binarization for 'assembly_categories' and 'assembly_industries' with cleaned labels
    for col in ['assembly_categories', 'assembly_industries']:
        # Split the comma-separated string into a list of labels, handle NaNs and strip whitespace
        dataset[col] = dataset[col].str.split(',').fillna('').apply(lambda x: [item.strip() for item in x])
        mlb = MultiLabelBinarizer()
        col_encoded = mlb.fit_transform(dataset[col])
        col_df = pd.DataFrame(col_encoded, columns=mlb.classes_, index=dataset.index)
        dataset = pd.concat([dataset, col_df], axis=1)
        dataset = dataset.drop(columns=[col])
    
    return dataset, encoder

# Apply the transformations
transformed_dataset, encoder = transform_features(consolidated_dataset_cleaned)
# Drop one of the redundant 'Architecture' columns
transformed_dataset_cleaned = transformed_dataset.drop(columns=['Architecture'])
transformed_dataset_cleaned
