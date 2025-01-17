# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:12:39 2025

@author: THINKPAD
"""
# %%
import rasterio
import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# %%
# file_of_paths = r"E:\ufl\Aim\Aim_1\tiff_labels.xlsx"
# # Read the Excel file into a DataFrame
# df = pd.read_excel(file_of_paths)
# df['TIFF_Path'] = df['TIFF_Path'].str.replace(
#     r'G:\\YUWANGDATA\\PHD\\test_sep\\', 
#     r'/blue/xuwang1/gao.k/Aim/Aim_1/Segment_Output/', 
#     regex=True
# )

# df.to_csv('E:\\ufl\\Aim\\updated_tiff_paths_2.csv', index=False)

file_of_paths = r"E:\ufl\Aim\updated_tiff_paths.csv"
# Read the Excel file into a DataFrame
df = pd.read_csv(file_of_paths)

segment_dict = {
    f"{row['File_Name']}":
                        {
                        "Sample": row['Sample'],
                        "Brix Level": (
                            "6-7.9 Brix" if 6 <= row['Brix'] < 8 else
                            "8-9.9 Brix" if 8 <= row['Brix'] < 10 else
                            "10-11.9 Brix" if 10 <= row['Brix'] < 12 else
                            "12-13.9 Brix" if 12 <= row['Brix'] < 14 else
                            "14-16 Brix"
                        ),
                        "Brix":row['Brix'],
                        "Acid":row['Acid'],
                        "TIFF_Path": row['TIFF_Path'],
                        
                        }
    for index, row in df.iterrows()
}


# %%


def load_image(filepath):
    with rasterio.open(filepath) as src:
        image = src.read(masked=True)  # Assuming the mask is applied in the .TIFF
    return image


def multi_dimention_to_1d_array(image, wavelengths, important_bands):
    # Ensure wavelengths and image have the same number of bands
    assert len(wavelengths) == image.shape[0], "Number of wavelengths must match the number of bands in the image."
    
    # print(wavelengths)
    # print(len(wavelengths))
    
    # Get indices of important bands
    important_indices = [wavelengths.index(band) for band in important_bands if band in wavelengths]
    
    # print(important_indices)
    
    # Extract only the important bands
    extracted_bands = image[important_indices, :, :]
    
    # print(extracted_bands.shape)
    
    # Flatten the extracted bands to a 1D array
    flattened_array = extracted_bands.reshape(len(important_indices), -1).flatten()

    # Number of pixels per band
    length_per_band = image.shape[1] * image.shape[2]

    return length_per_band, flattened_array

def average_spectral_per_band_excluding_zeros(image):
    # Create a mask where the image is not equal to zero
    # mask = image != 0
    # Use np.ma.array to create a masked array where the mask is applied
    masked_image = np.ma.array(image)
    # Average over the x and y dimensions, considering only non-zero elements
    return np.ma.mean(masked_image, axis=(1, 2))


def downsample_image(masked_image, factor=2):
    """
    Downsample the spatial dimensions of an image by a given factor.
    """
    return masked_image[:, ::factor, ::factor]  # Reduce the spatial dimensions


def resize_hyperspectral_image(image, target_shape=(100, 100)):
    """
    Resize a hyperspectral image to a fixed spatial size.
    
    Parameters:
        image (numpy array): Hyperspectral image with shape (band, height, width).
        target_shape (tuple): Target spatial shape (height, width).
    
    Returns:
        numpy array: Resized hyperspectral image with shape (band, target_height, target_width).
    """
    bands, height, width = image.shape
    resized_image = np.zeros((bands, target_shape[0], target_shape[1]))
    for b in range(bands):
        resized_image[b] = resize(image[b], target_shape, mode='reflect', anti_aliasing=True)
    return resized_image


# Function to process images stored in a dictionary and save the results to an Excel file
def process_images_to_excel(segment_dict, important_bands, reshape_x, reshape_y, downsample_factor=2):
    image_size = []
    label = []
    all_results = []
    wavelengths = np.arange(400, 1003, 2.7)
    formatted_wavelengths = [f"{round(wavelength):.1f}" for wavelength in wavelengths]
    # Iterate over each entry in the dictionary
    for file_name, data in segment_dict.items():
        filepath = data['TIFF_Path']
        file_brix = data['Brix']
        hyperspectral_image = load_image(filepath)
        resized_image = resize_hyperspectral_image(hyperspectral_image, target_shape=(reshape_x, reshape_y))
        image_size.append(resized_image.shape)
        # downsampled_image = downsample_image(image_size, downsample_factor=2)  # Reduce size by 2
        length_per_band, values_1d = multi_dimention_to_1d_array(resized_image, formatted_wavelengths, important_bands)
        label.append(file_brix)
        all_results.append(values_1d)
        # Prepare the dictionary with additional spectral data
        # result = {
        #     "File Name": file_name,
        #     "Sample": data['Sample'],
        #     "Brix Level": data['Brix Level'],
        #     "Brix": data['Brix'],
        #     "Acid": data['Acid']
        # }
        # i = 0
        # count = 0
        # for value in values_1d:
        #     cur_band = wavelengths[count//length_per_band]
        #     result[f'{cur_band}_{i}'] = value
        #     i += 1
        #     count += 1
        # all_results.append(result)
    
    # Convert results to a DataFrame
    # df = pd.DataFrame(all_results)
    
    return image_size, all_results, label
    

# %%
# output_excel_path = 'updated_spectral_data_summary_HV.xlsx'
# all_data, max_bands, wavelength_labels = process_images_to_excel(segment_dict, output_excel_path)

# image = load_image(r'E:/ufl/Aim/Aim_1/Segment_Output/0_104_2_interior_vertical.tiff')
# mask = image != 0
# masked_image = np.ma.array(image, mask=~mask)
# mean_array = np.ma.mean(masked_image, axis=(1, 2))
# flattened_array = masked_image.compressed()


# # Example usage, assuming 'segment_dict' is defined as described
# output_excel_path = 'updated_spectral_data_summary_HV.xlsx'
# df = process_images_to_excel(segment_dict, output_excel_path)
# # Write the DataFrame to an Excel file
# df.to_excel(output_excel_path, index=False, engine='openpyxl')

important_bands = [524.0, 802.0, 405.0, 689.0, 551.0, 527.0, 576.0, 800.0, 816.0, 557.0, 403.0, 530.0, 538.0, 532.0, 554.0, 486.0, 495.0, 592.0, 589.0, 929.0, 522.0, 516.0, 962.0, 408.0, 454.0, 927.0, 603.0, 459.0, 505.0, 500.0, 813.0, 468.0, 492.0, 465.0, 581.0, 400.0, 549.0, 900.0, 519.0, 470.0, 997.0, 478.0, 565.0, 994.0, 451.0, 924.0, 978.0, 503.0, 705.0, 497.0, 546.0, 513.0, 732.0, 975.0, 905.0, 951.0, 754.0, 594.0, 665.0, 746.0, 792.0, 651.0, 810.0]
important_bands = [f"{band:.1f}" for band in important_bands]
image_size, all_results, label = process_images_to_excel(segment_dict, important_bands, 100, 100, 4)

# %%

# Convert all_results and label to NumPy arrays
images_2d = np.array(all_results)  # Shape: (100, 224 * 300 * 300)
labels = np.array(label)  # Shape: (100,)
# %%
# 1. Normalize data
# scaler = StandardScaler()
# images_scaled = scaler.fit_transform(images_2d)

# # 2. Dimensionality reduction (PCA)
# pca = PCA(n_components=100)  # Choose number of principal components
# images_pca = pca.fit_transform(images_scaled)

# # 3. Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     images_pca, labels, test_size=0.2, random_state=42
# )

# List to store results
# List to store results
results = []

for n_pca in [5, 10, 15, 20, 25, 30, 35, 50, 100]:
    
    pca = PCA(n_components=n_pca)
    images_pca = pca.fit_transform(images_2d)
    
    X_train, X_test, y_train, y_test = train_test_split(
        images_pca, labels, test_size=0.2, random_state=42
    )
    
    pls = PLSRegression()

    # Define the hyperparameter grid
    param_grid = {
        'n_components': list(range(1, min(n_pca, 50) + 1, 5)) 
    }
    
    # Define a custom scoring function (e.g., negative mean squared error)
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    
    # Perform the grid search with cross-validation
    grid_search_pls = GridSearchCV(
        estimator=pls,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,  # Number of cross-validation folds
        verbose=2
    )
    
    # Fit the grid search to the training data
    grid_search_pls.fit(X_train, y_train)
    
    # Get the best parameters and estimator
    best_params_pls = grid_search_pls.best_params_
    best_pls = grid_search_pls.best_estimator_
    
    # Predict using the best model
    y_pred_pls = best_pls.predict(X_test)
    
    # Evaluate the performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_pls))
    r2 = r2_score(y_test, y_pred_pls)
    
    # Print the results
    print(f"PCA n_components={n_pca}, Best Parameters={best_params_pls}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # Append results to the list
    results.append({
        "PCA n_components": n_pca,
        "Best PLS n_components": best_params_pls['n_components'],
        "RMSE": rmse,
        "R²": r2
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
output_csv_path = "E:\\ufl\\Aim\\pls_results_pandas.csv"
results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
    
# %%
# 4. Train model (PLSR)
plsr = PLSRegression(n_components=10)  # Choose number of components
plsr.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = plsr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print results
print(f"RMSE: {rmse:.3f}")
print(f"R-squared: {r2:.3f}")