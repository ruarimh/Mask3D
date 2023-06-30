#%%
import numpy as np
import pandas as pd
import laspy
import shutil
import os
import random
import torch
from matplotlib import pyplot as plt
import open3d as o3d
import seaborn as sns
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



#%%
# This .py computes some basic ecological metrics and summary statistics
# for the train/val/test datasets, and then plots them

test_plots = [1, 3, 6, 10, 15, 27, 32, 34, 35, 48, 49, 52, 53, 58, 60]
validation_plots = [59, 26, 21, 8, 20, 57, 12]
all_plots = [1, 2, 3, 6, 8, 9, 10, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25,
             26, 27, 28, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

train_ground_truth_tree_heights = []
val_ground_truth_tree_heights = []
test_predicted_tree_heights = []
test_ground_truth_tree_heights = []

train_zASL = []
val_zASL = []
test_zASL = []

train_num_understory_instances = []
train_num_instances = []
val_num_understory_instances = []
val_num_instances = []
test_num_understory_instances = []
test_num_instances = []
test_num_predicted_understory_instances = []
test_num_predicted_instances = []

train_species_counts = [0, 0, 0, 0, 0, 0]
val_species_counts = [0, 0, 0, 0, 0, 0]
test_species_counts = [0, 0, 0, 0, 0, 0]

data_location = "D:/Downloads/MRes Data/processed/las/visualizations_full_500_rgb_test/"
raw_data_location = "D:/Downloads/MRes Data/FORinstance_dataset/NIBIO2/"

#%%
for plot_num in all_plots:

    
    
    subplot_num = 0
    
    if plot_num in test_plots:
        plot_sorted_masks = np.load(data_location + f"plot{plot_num}_annotated.las_{subplot_num}.txt_sorted_masks.npy")
        plot_full_res_coords = np.load(data_location + f"plot{plot_num}_annotated.las_{subplot_num}.txt_full_res_coords.npy")
        plot_sort_classes = np.load(data_location + f"plot{plot_num}_annotated.las_{subplot_num}.txt_sort_classes.npy")
    
    with laspy.open(raw_data_location + f"/plot{plot_num}_annotated.las") as fh:
        las = fh.read()
        
    full_data = las.points.array
    
    # rescale X, Y, Z to be in the range (0, 490) as in stpls3d
    full_data["X"] = ((full_data["X"] - full_data["X"].min()) * 
                     (1/(full_data["X"].max() - full_data["X"].min()) * 490))
    
    full_data["Y"] = ((full_data["Y"] - full_data["Y"].min()) * 
                     (1/(full_data["Y"].max() - full_data["Y"].min()) * 490))
    
    full_data["Z"] = ((full_data["Z"] - full_data["Z"].min()) * 
                     (1/(full_data["Z"].max() - full_data["Z"].min()) * 490))
    
    max_z = np.max(full_data["Z"])
    min_z = np.min(full_data["Z"])
    
    max_zASL = np.max(full_data["zASL"])
    min_zASL = np.min(full_data["zASL"])
    
    
    # init each plot with 0 instances
    if plot_num in validation_plots:
        val_num_understory_instances.append(0) 
    elif plot_num in test_plots:
        test_num_understory_instances.append(0)
        test_num_predicted_understory_instances.append(0)
    else:
        train_num_understory_instances.append(0)
    
    

    
    if plot_num in test_plots:
        pred_coords = []
        pred_normals = []
        pred_sem_color = []
        pred_inst_color = []
        
        for did in range(len(plot_sorted_masks)):
        
            for i in reversed(range(plot_sorted_masks[did].shape[1])):
                coords = plot_full_res_coords[plot_sorted_masks[did][:, i].astype(bool)]
        
                mask_coords = plot_full_res_coords[plot_sorted_masks[did][:,i].astype(bool)]
        
                label = plot_sort_classes[did][i]
        
                if len(mask_coords) == 0:
                    continue
        
                pred_coords.append(mask_coords)
                
        # get unique instance masks
        unique_pred_coords = []
        
        for arr in pred_coords:
            if not any(np.array_equal(arr, unique_arr) for unique_arr in unique_pred_coords):
                unique_pred_coords.append(arr)
        
        pred_coords = unique_pred_coords
        
        # filter out arrays that are subsets of other arrays
        filtered_pred_coords = []
        
        for i, array in enumerate(pred_coords):
            is_subset = False
            for j, other_array in enumerate(pred_coords):
                if i != j and array.shape[0] <= other_array.shape[0]:
                    if np.array_equal(array, other_array[:array.shape[0]]):
                        is_subset = True
                        break
            if not is_subset:
                filtered_pred_coords.append(array)
                
        pred_coords = filtered_pred_coords

        # for each plot, count the number of predicted trees
        test_num_predicted_instances.append(len(pred_coords) - 1)
        
        for predicted_instance in pred_coords:
            instance_z = np.max(predicted_instance[:, 2])
            
            test_predicted_tree_heights.append(instance_z)
            
            if instance_z < 0.25 * max_z:
                test_num_predicted_understory_instances[-1] += 1
        
    # count the number of ground truth trees
    if plot_num in validation_plots:
        val_num_instances.append(len(np.unique(full_data["treeID"])) - 1)
    elif plot_num in test_plots:
        test_num_instances.append(len(np.unique(full_data["treeID"])) - 1)
    else:
        train_num_instances.append(len(np.unique(full_data["treeID"])) - 1)
    
    # for each predicted and ground-truth tree, write down the species and height
    unique_instances = np.unique(full_data["treeID"])
    
    
    for unique_instance in unique_instances:
        if unique_instance > 0:
            tree_species = full_data["treeSP"][full_data["treeID"] == unique_instance]
            
            if plot_num in validation_plots:
                val_species_counts[tree_species[0].astype(int)] += 1
            elif plot_num in test_plots:
                test_species_counts[tree_species[0].astype(int)] += 1
            else:
                train_species_counts[tree_species[0].astype(int)] += 1
            
        instance_z = np.max(full_data["Z"][full_data["treeID"] == unique_instance])
        instance_zASL = np.max(full_data["zASL"][full_data["treeID"] == unique_instance]) - min_zASL
        
        if plot_num in validation_plots:
            val_ground_truth_tree_heights.append(instance_z)
            val_zASL.append(instance_zASL)
        elif plot_num in test_plots:
            test_ground_truth_tree_heights.append(instance_z)
            test_zASL.append(instance_zASL)
        else:
            train_ground_truth_tree_heights.append(instance_z)
            train_zASL.append(instance_zASL)
        
        if instance_z < 0.25 * max_z:
            if plot_num in validation_plots:
                val_num_understory_instances[-1] += 1
            elif plot_num in test_plots:
                test_num_understory_instances[-1] += 1
            else:
                train_num_understory_instances[-1] += 1
                
#%%
plotting_directory = "D:/One Drive/OneDrive - University of Essex/Ruari/Uni Stuff/PhD/Cambridge/AI4ER/MRes/Plots/"

plt.clf()

# Plot the distribution for train_ground_truth_tree_heights
sns.kdeplot(data=train_ground_truth_tree_heights, label='Train', bw = 0.2, 
            clip = (0, np.max(train_ground_truth_tree_heights)))

# Plot the distribution for val_ground_truth_tree_heights
sns.kdeplot(data=val_ground_truth_tree_heights, label='Validation', bw = 0.2,
            clip = (0, np.max(val_ground_truth_tree_heights)))

# Plot the distribution for test_ground_truth_tree_heights
sns.kdeplot(data=test_ground_truth_tree_heights, label='Test', bw = 0.2,
            clip = (0, np.max(test_ground_truth_tree_heights)))

# Set labels and title
plt.xlabel('Tree Height')
plt.ylabel('Density')

# Show the legend
plt.legend()

plt.gcf().subplots_adjust(left=0.2)

plt.savefig(plotting_directory + 'tree_height_distributions.png',dpi=600)

# Show the plot
plt.show()




#%%

plt.clf()

# Plot the distribution for test_ground_truth_tree_heights
sns.kdeplot(data=test_ground_truth_tree_heights, label='Ground Truth', bw = 0.2)

# Plot the distribution for test_ground_truth_tree_heights
sns.kdeplot(data=test_predicted_tree_heights, label='Predicted', bw = 0.2)

# Set labels and title
plt.xlabel('Tree Height (arbitrary units)')
plt.ylabel('Density')

# Show the legend
plt.legend()

plt.savefig(plotting_directory + 'tree_height_predicted_vs_gt.png',dpi=600)

# Show the plot
plt.show()


#%%
plt.clf()

data = {'x': test_num_understory_instances, 'y': test_num_predicted_understory_instances}
df = pd.DataFrame(data)

# Scatter plot with line of best fit and confidence bands
sns.lmplot(x='x', y='y', data=df, ci=95)

biggest_value = max(max(test_num_understory_instances), max(test_num_predicted_understory_instances))
plt.plot([0, biggest_value], [0, biggest_value], '--', color='grey', zorder = 0)

# Set labels and title
plt.xlabel('Number of understory trees')
plt.ylabel('Number of predicted understory trees')

plt.xlim(-2)
plt.ylim(-5)

# Save the plot to disk
plt.savefig(plotting_directory + 'predicted_vs_gt_understory.png', dpi=600)

# Show the plot
plt.show()

#%%
plt.clf()

train_species_counts = train_species_counts[0:4]
val_species_counts = val_species_counts[0:4]
test_species_counts = test_species_counts[0:4]

# Define the categories (x-axis labels)
categories = ['Unclassified', 'Picea abies', 'Pinus silvestris', 'Betula pendula']

# Calculate the total counts for each dataset
train_total = sum(train_species_counts)
val_total = sum(val_species_counts)
test_total = sum(test_species_counts)

# Calculate the percentages
train_percentages = [count / train_total * 100 for count in train_species_counts]
val_percentages = [count / val_total * 100 for count in val_species_counts]
test_percentages = [count / test_total * 100 for count in test_species_counts]

# Set the width of each bar
bar_width = 0.25

# Set the positions of the bars on the x-axis
train_positions = np.arange(len(categories))
val_positions = train_positions + bar_width
test_positions = val_positions + bar_width

# Create the grouped bar plot
plt.bar(train_positions, train_percentages, width=bar_width, label='Train')
plt.bar(val_positions, val_percentages, width=bar_width, label='Validation')
plt.bar(test_positions, test_percentages, width=bar_width, label='Test')

# Add labels and title
plt.xlabel('Tree Species')
plt.ylabel('% of species in dataset')
plt.xticks(val_positions, categories)

# Add legend
plt.legend()

plt.savefig(plotting_directory + 'tree_species_distribution.png',dpi=600)

# Show the plot
plt.show()

#%%

plotting_directory = "D:/One Drive/OneDrive - University of Essex/Ruari/Uni Stuff/PhD/Cambridge/AI4ER/MRes/Plots/"

plt.clf()

# Plot the distribution for train_ground_truth_tree_heights
sns.kdeplot(data=train_zASL, label='Train', bw = 0.2, 
            clip = (0, np.max(train_zASL)))

# Plot the distribution for val_ground_truth_tree_heights
sns.kdeplot(data=val_zASL, label='Validation', bw = 0.2,
            clip = (0, np.max(val_zASL)))

# Plot the distribution for test_ground_truth_tree_heights
sns.kdeplot(data=test_zASL, label='Test', bw = 0.2,
            clip = (0, np.max(test_zASL)))

# Set labels and title
plt.xlabel('Tree Height (m)')
plt.ylabel('Density')

# Show the legend
plt.legend()

plt.gcf().subplots_adjust(left=0.2)

plt.savefig(plotting_directory + 'tree_zASL_distributions.png',dpi=600)

# Show the plot
plt.show()

