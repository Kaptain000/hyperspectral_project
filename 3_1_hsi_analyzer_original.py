# -*- coding: utf-8 -*-
"""3.1_HSI_Analyzer_Updated.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eW0-XcoL-Lj1ORuK5YUmsy2PHcH5C0ot
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from chemotools.feature_selection import RangeCut
from chemotools.baseline import LinearCorrection
from chemotools.derivative import SavitzkyGolay
from chemotools.smooth import MeanFilter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV

"""# Loading the Data"""

specta_file_path = r"E:/ufl/Aim/Aim_1/HSI_spectra_124_Updated.xlsx"
spectra = pd.read_excel(specta_file_path)
# C:\Users\Robert\PHD\Python\My work\spectra.xlsx
# G:\YUWANGDATA\PHD\Python\My work\spectra.xlsx
label_file_path = r"E:/ufl/Aim/Aim_1/HSI_label_Brix_124_Updated.xlsx"
label = pd.read_excel(label_file_path)
# C:\Users\Robert\PHD\Python\My work\label.xlsx
# G:\YUWANGDATA\PHD\Python\My work\label.xlsx
print(f"Number of samples: {spectra.shape[0]}")
print(f"Number of wavenumbers: {spectra.shape[1]}")

label.describe()

# Convert the spectra pandas.DataFrame to numpy.ndarray
spectra_np = spectra.to_numpy()
# Convert the wavenumbers pandas.columns to numpy.ndarray
wavenumbers = spectra.columns.to_numpy(dtype=np.float64)

# Convert the label pandas.DataFrame to numpy.ndarray
label_np = label.to_numpy()

# Creating this to visualize the spectra.

def plot_spectra(spectra: np.ndarray, wavenumbers: np.ndarray, label: np.ndarray, chartTitle = "Citrus Reflectance"):
    # Define a colormap
    cmap = plt.get_cmap("jet")

    # Define a normalization function to scale glucose concentrations between 0 and 1
    normalize = Normalize(vmin=label.min(), vmax=label.max())
    colors = [cmap(normalize(value)) for value in label]

    # Plot the spectra
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, row in enumerate(spectra):
        ax.plot(wavenumbers, row, color = colors[i])

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Brix')

    # Add labels
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.set_title(chartTitle)

    plt.show()

plot_spectra(spectra_np, wavenumbers, label_np)

"""# Preprocessing

## Savitzky-Golay
"""

# create a pipeline that scales the data
# Savitzky-Golay filtering to smooth high-frequency noise
SG_preprocessing = make_pipeline(
    RangeCut(start=400, end=1000, wavenumbers=wavenumbers),
    LinearCorrection(),
    SavitzkyGolay(window_size=20, polynomial_order=2, derivate_order=2),
    StandardScaler(with_std=False)
)

SG_spectra_preprocessed = SG_preprocessing.fit_transform(spectra_np)

# get the wavenumbers after the range cut
wavenumbers_cut = SG_preprocessing.named_steps['rangecut'].wavenumbers_


# plot the preprocessed spectra
plot_spectra(SG_spectra_preprocessed, wavenumbers_cut, label_np, chartTitle="Savitzky-Golay Reflectance")

# fig, ax = plt.subplots(figsize=(10, 4))
# ax.plot(param_grid['n_components'], np.abs(grid_search.cv_results_['mean_test_score']), marker='o', color='b')
# ax.set_xlabel('Number of components')
# ax.set_ylabel('Mean absolute error')
# ax.set_title('Cross validation results')

"""## Mean Filter"""

# create a pipeline that scales the data
MF_preprocessing = make_pipeline(
    RangeCut(start=400, end=1000, wavenumbers=wavenumbers),
    LinearCorrection(),
    MeanFilter(window_size=5),
    StandardScaler(with_std=False)
)

MF_spectra_preprocessed = MF_preprocessing.fit_transform(spectra_np)

# get the wavenumbers after the range cut
wavenumbers_cut = MF_preprocessing.named_steps['rangecut'].wavenumbers_


# plot the preprocessed spectra
plot_spectra(MF_spectra_preprocessed, wavenumbers_cut, label_np, chartTitle="Mean Filtered Reflectance")


''' original dataset '''
print("====================== PLSRegression with original dataset ===========================")
# instanciate a PLSRegression object
pls = PLSRegression(scale=False)

param_grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

grid_search = GridSearchCV(pls, param_grid, cv=10, scoring='neg_mean_absolute_error')

# MF_spectra_preprocessed: HSI_spectra_124.xlsx => Spectra => spectra_np => MF_spectra_preprocessed
grid_search.fit(spectra_np, label_np)

# print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", np.abs(grid_search.best_score_))

# instanciate a PLSRegression object with 10 components
pls = PLSRegression(n_components=10, scale=False)

# fit the model to the data
pls.fit(spectra_np, label_np)
# predict the Brix
label_pred = pls.predict(spectra_np)

# plot
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(label_np, label_pred, color='blue')
ax.plot([7, 15], [7, 15], color='magenta')
ax.set_xlabel('Measured Brix')
ax.set_ylabel('Predicted Brix')
ax.set_title('original PLSR')


from sklearn.metrics import r2_score, mean_squared_error
print("====================== polyfit with original dataset ===========================")
# Fit a linear trendline
z = np.polyfit(label_np.ravel(), label_pred.ravel(), 1)
p = np.poly1d(z)
ax.plot(label_np, p(label_np), color='red', label='Trendline')

# Calculate R² and MSE
r2 = r2_score(label_np, label_pred)
mse = mean_squared_error(label_np, label_pred)

# Add R² value to the plot
ax.text(0.05, 0.95, f'R²: {r2:.3f}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Print the metrics
print(f"R²: {r2:.3f}")
print(f"MSE: {mse:.3f}")
variance_brix = np.var(label_np)
print(f"Variance of Brix: {variance_brix:.3f}")

plt.show()

min_label = np.min(label_np)
max_label = np.max(label_np)
mean_label = np.mean(label_np)
std_label = np.std(label_np)

# Print the values
print(f"Min measured Brix: {min_label:.3f}")
print(f"Max measured Brix: {max_label:.3f}")
print(f"Mean measured Brix: {mean_label:.3f}")
print(f"Standard deviation of measured Brix: {std_label:.3f}")

# Convert predicted labels to a NumPy array
label_pred = np.array(label_pred)

# Calculate min, max, mean, and standard deviation
min_pred = np.min(label_pred)
max_pred = np.max(label_pred)
mean_pred = np.mean(label_pred)
std_pred = np.std(label_pred)

# Print the values
print(f"Min predicted Brix: {min_pred:.3f}")
print(f"Max predicted Brix: {max_pred:.3f}")
print(f"Mean predicted Brix: {mean_pred:.3f}")
print(f"Standard deviation of predicted Brix: {std_pred:.3f}")




"""# Prediction

Testing different methods and troubleshooting any potential issues.

## PLSR
Testing PLSR to see what works and compare SG to MF.

### SG Preprocessed
"""


print("====================== PLSRegression with SG_spectra_preprocessed dataset ===========================")
# instanciate a PLSRegression object
pls = PLSRegression(scale=False)

param_grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

grid_search = GridSearchCV(pls, param_grid, cv=10, scoring='neg_mean_absolute_error')

''' X is SG_spectra_preprocessed: HSI_spectra_124.xlsx => Spectra => spectra_np => SG_spectra_preprocessed'''
''' Y is label_np, HSI_label_124.xlsx => Label => label_np'''
grid_search.fit(SG_spectra_preprocessed, label_np)

# print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", np.abs(grid_search.best_score_))

# instanciate a PLSRegression object with 10 components
pls = PLSRegression(n_components=10, scale=False)

# fit the model to the data
pls.fit(SG_spectra_preprocessed, label_np)
# predict the Brix
SG_label_pred = pls.predict(SG_spectra_preprocessed)

# plot the predictions
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(label_np, SG_label_pred, color='blue')
ax.plot([7, 15], [7, 15], color='magenta')
ax.set_xlabel('Measured Brix')
ax.set_ylabel('Predicted Brix')
ax.set_title('Savitzky-Golay PLSR')


from sklearn.metrics import r2_score, mean_squared_error
print("====================== polyfit with SG_spectra_preprocessed dataset ===========================")
# Fit a linear trendline
z = np.polyfit(label_np.ravel(), SG_label_pred.ravel(), 1)
p = np.poly1d(z)
ax.plot(label_np, p(label_np), color='red', label='Trendline')

# Calculate R² and MSE
SG_r2 = r2_score(label_np, SG_label_pred)
SG_mse = mean_squared_error(label_np, SG_label_pred)

# Add R² value to the plot
ax.text(0.05, 0.95, f'R²: {SG_r2:.3f}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Print the metrics
print(f"R²: {SG_r2:.3f}")
print(f"MSE: {SG_mse:.3f}")
variance_brix = np.var(label_np)
print(f"Variance of Brix: {variance_brix:.3f}")

plt.show()

min_label = np.min(label_np)
max_label = np.max(label_np)
mean_label = np.mean(label_np)
std_label = np.std(label_np)

# Print the values
print(f"Min measured Brix: {min_label:.3f}")
print(f"Max measured Brix: {max_label:.3f}")
print(f"Mean measured Brix: {mean_label:.3f}")
print(f"Standard deviation of measured Brix: {std_label:.3f}")

# Convert predicted labels to a NumPy array
SG_label_pred = np.array(SG_label_pred)

# Calculate min, max, mean, and standard deviation
min_pred = np.min(SG_label_pred)
max_pred = np.max(SG_label_pred)
mean_pred = np.mean(SG_label_pred)
std_pred = np.std(SG_label_pred)

# Print the values
print(f"Min predicted Brix: {min_pred:.3f}")
print(f"Max predicted Brix: {max_pred:.3f}")
print(f"Mean predicted Brix: {mean_pred:.3f}")
print(f"Standard deviation of predicted Brix: {std_pred:.3f}")


"""### Mean Filter Preprocessed"""
print("====================== PLSRegression with MF_spectra_preprocessed dataset ===========================")
# instanciate a PLSRegression object
pls = PLSRegression(scale=False)

param_grid = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

grid_search = GridSearchCV(pls, param_grid, cv=10, scoring='neg_mean_absolute_error')

# MF_spectra_preprocessed: HSI_spectra_124.xlsx => Spectra => spectra_np => MF_spectra_preprocessed
grid_search.fit(MF_spectra_preprocessed, label_np)

# print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", np.abs(grid_search.best_score_))

# instanciate a PLSRegression object with 10 components
pls = PLSRegression(n_components=10, scale=False)

# fit the model to the data
pls.fit(MF_spectra_preprocessed, label_np)
# predict the Brix
MF_label_pred = pls.predict(MF_spectra_preprocessed)

# plot
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(label_np, MF_label_pred, color='blue')
ax.plot([7, 15], [7, 15], color='magenta')
ax.set_xlabel('Measured Brix')
ax.set_ylabel('Predicted Brix')
ax.set_title('Mean Filtered PLSR')


from sklearn.metrics import r2_score, mean_squared_error
print("====================== polyfit with MF_spectra_preprocessed dataset ===========================")
# Fit a linear trendline
z = np.polyfit(label_np.ravel(), MF_label_pred.ravel(), 1)
p = np.poly1d(z)
ax.plot(label_np, p(label_np), color='red', label='Trendline')

# Calculate R² and MSE
r2 = r2_score(label_np, MF_label_pred)
mse = mean_squared_error(label_np, MF_label_pred)

# Add R² value to the plot
ax.text(0.05, 0.95, f'R²: {r2:.3f}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Print the metrics
print(f"R²: {r2:.3f}")
print(f"MSE: {mse:.3f}")
variance_brix = np.var(label_np)
print(f"Variance of Brix: {variance_brix:.3f}")

plt.show()

min_label = np.min(label_np)
max_label = np.max(label_np)
mean_label = np.mean(label_np)
std_label = np.std(label_np)

# Print the values
print(f"Min measured Brix: {min_label:.3f}")
print(f"Max measured Brix: {max_label:.3f}")
print(f"Mean measured Brix: {mean_label:.3f}")
print(f"Standard deviation of measured Brix: {std_label:.3f}")

# Convert predicted labels to a NumPy array
MF_label_pred = np.array(MF_label_pred)

# Calculate min, max, mean, and standard deviation
min_pred = np.min(MF_label_pred)
max_pred = np.max(MF_label_pred)
mean_pred = np.mean(MF_label_pred)
std_pred = np.std(MF_label_pred)

# Print the values
print(f"Min predicted Brix: {min_pred:.3f}")
print(f"Max predicted Brix: {max_pred:.3f}")
print(f"Mean predicted Brix: {mean_pred:.3f}")
print(f"Standard deviation of predicted Brix: {std_pred:.3f}")

"""## SVM/R

### SG Pre
"""


print("====================== SVC ===========================")
# Instantiate an SVR model
svr = SVR()

# Define the parameter grid for SVR
param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1]}

# Create a grid search object
grid_search_svr = GridSearchCV(svr, param_grid, cv=10, scoring='neg_mean_absolute_error')

# Use ravel() to convert label_np to a 1D array
label_np_1d = label_np.ravel()

# Now fit the SVR model
grid_search_svr.fit(SG_spectra_preprocessed, label_np_1d)

'''================  ================='''
# Predict the Brix levels using the best SVR model
SG_label_pred_svr = grid_search_svr.best_estimator_.predict(SG_spectra_preprocessed)

# Plot the scatter plot
plt.scatter(label_np_1d, SG_label_pred_svr, color='blue', label='Data points')
plt.plot([7, 15], [7, 15], color='magenta', label='Ideal line')

# Add labels, title, and R² score
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('SVR Predictions')
plt.text(7.2, 14.8, f"$R^2$: {r2_score(label_np_1d, SG_label_pred_svr):.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.legend()
plt.show()

# Evaluate the model
print("Best parameters for SVR: ", grid_search_svr.best_params_)
print(f"R²: {r2_score(label_np_1d, SG_label_pred_svr):.3f}")
print(f"MSE: {mean_squared_error(label_np_1d, SG_label_pred_svr):.3f}")


print('fit with original data')
# Now fit the SVR model
grid_search_svr.fit(spectra_np, label_np_1d)

# Predict the Brix levels using the best SVR model
label_pred_svr = grid_search_svr.best_estimator_.predict(spectra_np)

# Plot the predictions (similar to PLSR)
plt.scatter(label_np_1d, label_pred_svr, color='blue')
plt.plot([7, 15], [7, 15], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('SVR Predictions with original data')
plt.text(7.2, 14.8, f"$R^2$: {r2_score(label_np_1d, label_pred_svr):.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()

# Evaluate the model
print("Best parameters for SVR: ", grid_search_svr.best_params_)
print(f"R²: {r2_score(label_np_1d, label_pred_svr):.3f}")
print(f"MSE: {mean_squared_error(label_np_1d, label_pred_svr):.3f}")

"""## Random Forest"""


print("====================== Random Forest ===========================")
# Convert label_np to 1D array using ravel()
label_np_1d = label_np.ravel()

# Define the RandomForest model
rf = RandomForestRegressor(random_state=42)

# Define the cross-validation strategy
cv = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold CV


r2_scores = cross_val_score(rf, SG_spectra_preprocessed, label_np_1d, cv=cv, scoring='r2')

# Perform cross-validation and get MSE scores (convert negative values to positive)
mse_scores = -cross_val_score(rf, SG_spectra_preprocessed, label_np_1d, cv=cv, scoring='neg_mean_squared_error')

# Print cross-validated performance
print(f"Mean R² across folds: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"Mean MSE across folds: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")

# Fit Random Forest on dataset
rf.fit(SG_spectra_preprocessed, label_np_1d)

# Predict on the entire dataset
SG_label_pred_rf = rf.predict(SG_spectra_preprocessed)

# Plot
plt.scatter(label_np_1d, SG_label_pred_rf, color='green')
plt.plot([7, 15], [7, 15], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('Random Forest Predictions (Full Data)')
plt.text(7.2, 14.8, f"$R^2$: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()


print('fit with original data')
r2_scores = cross_val_score(rf, spectra_np, label_np_1d, cv=cv, scoring='r2')

# Perform cross-validation and get MSE scores (convert negative values to positive)
mse_scores = -cross_val_score(rf, spectra_np, label_np_1d, cv=cv, scoring='neg_mean_squared_error')

# Print cross-validated performance
print(f"Mean R² across folds: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"Mean MSE across folds: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")

# Fit Random Forest on dataset
rf.fit(spectra_np, label_np_1d)

# Predict on the entire dataset
label_pred_rf = rf.predict(spectra_np)

# Plot
plt.scatter(label_np_1d, label_pred_rf, color='green')
plt.plot([7, 15], [7, 15], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('Random Forest Predictions (original data)')
plt.text(7.2, 14.8, f"$R^2$: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()


"""## ANN"""


print("====================== ANN ===========================")
# Define the parameter grid for tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100, 50)],  # Different layer sizes
    'activation': ['relu', 'tanh', 'logistic'],  # Activation functions
    'solver': ['adam', 'lbfgs', 'sgd'],  # Optimizers
    'alpha': [0.0001, 0.001, 0.01],  # Regularization strength
    'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
    'max_iter': [500, 1000, 2000]  # Maximum number of iterations
}

# Define the MLPRegressor model
mlp = MLPRegressor(random_state=42)

# Define the cross-validation strategy (10-fold CV)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform RandomizedSearchCV to tune the hyperparameters
random_search = RandomizedSearchCV(mlp, param_distributions=param_grid,
                                    n_iter=20, cv=cv, scoring='r2', n_jobs=-1, random_state=42)

# Fit the randomized search to data
random_search.fit(SG_spectra_preprocessed, label_np.ravel())

# Print the best parameters and the corresponding R² score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best R²: {random_search.best_score_:.3f}")

# get the best model
best_mlp_model = random_search.best_estimator_

# Predict with the best model
predicted_brix = best_mlp_model.predict(SG_spectra_preprocessed)

# Plot actual vs predicted values
import matplotlib.pyplot as plt
plt.scatter(label_np, predicted_brix, color='blue')
plt.plot([min(label_np), max(label_np)], [min(label_np), max(label_np)], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('MLP Neural Network Predictions (Best Model)')
plt.text(7.2, 14.8, f"$R^2$: {random_search.best_score_:.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()


print('fit with original data')
# Fit the randomized search to data
random_search.fit(spectra_np, label_np.ravel())

# Print the best parameters and the corresponding R² score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best R²: {random_search.best_score_:.3f}")

# get the best model
best_mlp_model = random_search.best_estimator_

# Predict with the best model
predicted_brix = best_mlp_model.predict(spectra_np)

# Plot actual vs predicted values
import matplotlib.pyplot as plt
plt.scatter(label_np, predicted_brix, color='blue')
plt.plot([min(label_np), max(label_np)], [min(label_np), max(label_np)], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('MLP Neural Network Predictions (Best Model with original data)')
plt.text(7.2, 14.8, f"$R^2$: {random_search.best_score_:.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()



"""## KNN"""


print("====================== KNN ===========================")
# Define the KNN Regressor model
knn = KNeighborsRegressor(n_neighbors=5)

# Convert label_np to 1D array
label_np_1d = label_np.ravel()

cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and get R² scores
r2_scores = cross_val_score(knn, SG_spectra_preprocessed, label_np_1d, cv=cv, scoring='r2')

# Perform cross-validation and get MSE scores
mse_scores = cross_val_score(knn, SG_spectra_preprocessed, label_np_1d, cv=cv, scoring='neg_mean_squared_error')
mse_scores = -mse_scores  # Convert negative MSE values to positive

# Print the cross-validated performance
print(f"Mean R² across folds: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"Mean MSE across folds: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")

knn.fit(SG_spectra_preprocessed, label_np_1d)

predicted_brix = knn.predict(SG_spectra_preprocessed)

# Plot actual vs predicted values
import matplotlib.pyplot as plt
plt.scatter(label_np_1d, predicted_brix, color='blue')
plt.plot([min(label_np_1d), max(label_np_1d)], [min(label_np_1d), max(label_np_1d)], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('KNN Regression Predictions')
plt.text(7.2, 14.8, f"$R^2$: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()


print('fit with original data')
# Perform cross-validation and get R² scores
r2_scores = cross_val_score(knn, spectra_np, label_np_1d, cv=cv, scoring='r2')

# Perform cross-validation and get MSE scores
mse_scores = cross_val_score(knn, spectra_np, label_np_1d, cv=cv, scoring='neg_mean_squared_error')
mse_scores = -mse_scores  # Convert negative MSE values to positive

# Print the cross-validated performance
print(f"Mean R² across folds: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"Mean MSE across folds: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")

knn.fit(spectra_np, label_np_1d)

predicted_brix = knn.predict(spectra_np)

# Plot actual vs predicted values
import matplotlib.pyplot as plt
plt.scatter(label_np_1d, predicted_brix, color='blue')
plt.plot([min(label_np_1d), max(label_np_1d)], [min(label_np_1d), max(label_np_1d)], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('KNN Regression Predictions(original data)')
plt.text(7.2, 14.8, f"$R^2$: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()


"""## XGBoost"""

# Define the XGBoost model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Convert label_np to 1D array
label_np_1d = label_np.ravel()

cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and get R² scores
r2_scores = cross_val_score(xgb, SG_spectra_preprocessed, label_np_1d, cv=cv, scoring='r2')

# Perform cross-validation and get MSE scores
mse_scores = cross_val_score(xgb, SG_spectra_preprocessed, label_np_1d, cv=cv, scoring='neg_mean_squared_error')
mse_scores = -mse_scores  # Convert negative MSE values to positive

# Print the cross-validated performance
print(f"Mean R² across folds: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"Mean MSE across folds: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")

xgb.fit(SG_spectra_preprocessed, label_np_1d)

predicted_brix = xgb.predict(SG_spectra_preprocessed)

# Plot actual vs predicted values
import matplotlib.pyplot as plt
plt.scatter(label_np_1d, predicted_brix, color='blue')
plt.plot([min(label_np_1d), max(label_np_1d)], [min(label_np_1d), max(label_np_1d)], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('XGBoost Predictions')
plt.text(7.2, 14.8, f"$R^2$: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()



print('fit with original data')
r2_scores = cross_val_score(xgb, spectra_np, label_np_1d, cv=cv, scoring='r2')

# Perform cross-validation and get MSE scores
mse_scores = cross_val_score(xgb, spectra_np, label_np_1d, cv=cv, scoring='neg_mean_squared_error')
mse_scores = -mse_scores  # Convert negative MSE values to positive

# Print the cross-validated performance
print(f"Mean R² across folds: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"Mean MSE across folds: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")

xgb.fit(spectra_np, label_np_1d)

predicted_brix = xgb.predict(spectra_np)

# Plot actual vs predicted values
import matplotlib.pyplot as plt
plt.scatter(label_np_1d, predicted_brix, color='blue')
plt.plot([min(label_np_1d), max(label_np_1d)], [min(label_np_1d), max(label_np_1d)], color='magenta')
plt.xlabel('Measured Brix')
plt.ylabel('Predicted Brix')
plt.title('XGBoost Predictions(original data)')
plt.text(7.2, 14.8, f"$R^2$: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}", fontsize=12, color='black')  # Add R² score at the top left corner
plt.show()

# %%
"""## Model Optimization

Now that I ironed out the compatibility issues, here is where actual model development starts.
"""

# Defining the parameters for each model
param_grids = {
    'PLSR': {'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'SVR': {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf']},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    'ANN': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100, 50)],  # Different hidden layer sizes
        'activation': ['relu', 'tanh'],  # Activation functions
        'solver': ['adam', 'lbfgs'],  # Optimizers
        'alpha': [0.0001, 0.001, 0.01]  # Regularization strength
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 10],  # Different numbers of neighbors
        'weights': ['uniform', 'distance'],  # Weighting methods
        'metric': ['euclidean', 'manhattan']  # Distance metrics
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],  # Number of boosting rounds
        'max_depth': [3, 5, 7],  # Maximum depth of trees
        'learning_rate': [0.01, 0.1, 0.2],  # Learning rate for boosting
        'subsample': [0.7, 0.8, 1.0],  # Subsampling for each tree
        'colsample_bytree': [0.7, 0.8, 1.0]  # Subsampling for columns
    }
}

# Define models to compare without parameters
pls = PLSRegression(scale=False)
svr = SVR()
rf = RandomForestRegressor()
ann = MLPRegressor(max_iter=1000)
knn = KNeighborsRegressor()
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

models = {
    'PLSR': pls,
    'SVR': svr,
    'Random Forest': rf,
    'ANN': ann,
    'KNN': knn,
    'XGBoost': xgb
}

# Convert label_np to 1D array
label_np_1d = label_np.ravel()

# Define the cross-validation strategy
cv = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold CV

# Parameter tuning and cross-validation for each model
results = {}

for model_name, model in models.items():
    print(f"Evaluating {model_name} with parameter tuning...")

    # Parameter tuning
    grid_search = GridSearchCV(model, param_grids[model_name], cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the grid search object to the data
    grid_search.fit(SG_spectra_preprocessed, label_np_1d)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Evaluate the best model using cross-validation with R² score
    r2_scores = cross_val_score(best_model, SG_spectra_preprocessed, label_np_1d, cv=cv, scoring='r2')
    mse_scores = cross_val_score(best_model, SG_spectra_preprocessed, label_np_1d, cv=cv, scoring='neg_mean_squared_error')
    mse_scores = -mse_scores  # Convert negative MSE values to positive

    results[model_name] = {
        'Best Params': grid_search.best_params_,
        'MSE Mean': np.mean(mse_scores),
        'MSE Std': np.std(mse_scores),
        'R² Mean': np.mean(r2_scores),
        'R² Std': np.std(r2_scores)
    }

# Print the results
for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print(f"Best Parameters: {result['Best Params']}")
    print(f"Mean MSE: {result['MSE Mean']:.3f} ± {result['MSE Std']:.3f}")
    print(f"Mean R²: {result['R² Mean']:.3f} ± {result['R² Std']:.3f}")

# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]  # Sort by importance

# # Plot the top features
# plt.figure(figsize=(10, 6))
# plt.bar(range(10), importances[indices[:10]], color="r", align="center")
# plt.xticks(range(10), indices[:10])
# plt.title('Top 10 Feature Importances (Random Forest)')
# plt.xlabel('Feature Index')
# plt.ylabel('Importance Score')
# plt.show()

"""# Final Model figures"""

## Training the final models
# Retrain each model using the best parameters from the grid search
final_models = {}

for model_name, result in results.items():
    best_params = result['Best Params']

    if model_name == 'PLSR':
        final_model = PLSRegression(**best_params)
    elif model_name == 'SVR':
        final_model = SVR(**best_params)
    elif model_name == 'Random Forest':
        final_model = RandomForestRegressor(**best_params)
    elif model_name == 'ANN':
        final_model = MLPRegressor(max_iter=1000, **best_params)
    elif model_name == 'KNN':
        final_model = KNeighborsRegressor(**best_params)
    elif model_name == 'XGBoost':
        final_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)

    # Fit the final model to the full dataset
    final_model.fit(SG_spectra_preprocessed, label_np_1d)

    final_models[model_name] = final_model

# # Predict using the final models
# for model_name, final_model in final_models.items():
#     predictions = final_model.predict(SG_spectra_preprocessed)

#     print(f"{model_name} predictions: {predictions[:5]}")  # Print first 5 predictions as an example

actual_brix = label_np.ravel()

# Predictions for each model
predicted_brix_plsr = final_models['PLSR'].predict(SG_spectra_preprocessed)
predicted_brix_svr = final_models['SVR'].predict(SG_spectra_preprocessed)
predicted_brix_rf = final_models['Random Forest'].predict(SG_spectra_preprocessed)
predicted_brix_ann = final_models['ANN'].predict(SG_spectra_preprocessed)
predicted_brix_knn = final_models['KNN'].predict(SG_spectra_preprocessed)
predicted_brix_xgb = final_models['XGBoost'].predict(SG_spectra_preprocessed)

# # Plotting actual vs predicted, add trendline, and R²
# def plot_with_trendline(ax, actual, predicted, model_name, color):
#     ax.scatter(actual, predicted, color=color)
#     ax.plot([min(actual), max(actual)], [min(actual), max(actual)], color='magenta', linestyle='--')

#     # Fit a linear trendline
#     z = np.polyfit(actual, predicted, 1)  # Fit a 1st degree polynomial (straight line)
#     p = np.poly1d(z)
#     ax.plot(actual, p(actual), color='red', label='Trendline')  # Add the trendline

#     # Calculate R²
#     r2 = r2_score(actual, predicted)
#     ax.text(0.05, 0.90, f'R²: {r2:.3f}', transform=ax.transAxes, fontsize=10,
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# # Title and axis labels with increased fontsize
#     ax.set_title(f'{model_name}: Predicted Brix', fontsize=16)
#     ax.set_xlabel('Measured 6-Methyl-5-Hepten-2-one', fontsize=14)
#     ax.set_ylabel('Predicted 6-Methyl-5-Hepten-2-one', fontsize=14)

#     # Move legend
#     ax.legend(loc='upper left', fontsize=12)

# fig, axs = plt.subplots(3, 2, figsize=(12, 18))

# # PLSR
# plot_with_trendline(axs[0, 0], actual_brix, predicted_brix_plsr, 'PLSR', 'blue')

# # SVR
# plot_with_trendline(axs[0, 1], actual_brix, predicted_brix_svr, 'SVR', 'green')

# # Random Forest
# plot_with_trendline(axs[1, 0], actual_brix, predicted_brix_rf, 'Random Forest', 'orange')

# # ANN
# plot_with_trendline(axs[1, 1], actual_brix, predicted_brix_ann, 'ANN', 'purple')

# # KNN
# plot_with_trendline(axs[2, 0], actual_brix, predicted_brix_knn, 'KNN', 'red')

# # XGBoost
# plot_with_trendline(axs[2, 1], actual_brix, predicted_brix_xgb, 'XGBoost', 'cyan')

# plt.tight_layout()

# # Show the combined figure
# plt.show()

# Plotting actual vs predicted, add trendline, and R²
def plot_with_trendline(actual, predicted, model_name, color):
    plt.figure(figsize=(5, 5))  # Create a new figure for each model
    plt.scatter(actual, predicted, color=color)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='magenta', linestyle='--')

    # Fit a linear trendline
    z = np.polyfit(actual, predicted, 1)  # Fit a 1st degree polynomial (straight line)
    p = np.poly1d(z)
    plt.plot(actual, p(actual), color='red', label='Trendline')  # Add the trendline

    # Calculate R²
    r2 = r2_score(actual, predicted)
    plt.text(0.05, 0.85, f'R²: {r2:.3f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Title and axis labels with increased fontsize
    plt.title(f'{model_name}: Predicted Brix-124', fontsize=16)
    plt.xlabel('Measured Brix', fontsize=14)
    plt.ylabel('Predicted Brix', fontsize=14)

    # Move legend
    plt.legend(loc='upper left', fontsize=12)

    # Show the figure
    plt.tight_layout()
    plt.show()

# Plot each model separately
plot_with_trendline(actual_brix, predicted_brix_plsr, 'PLSR', 'blue')
plot_with_trendline(actual_brix, predicted_brix_svr, 'SVR', 'green')
plot_with_trendline(actual_brix, predicted_brix_rf, 'Random Forest', 'orange')
plot_with_trendline(actual_brix, predicted_brix_ann, 'ANN', 'purple')
plot_with_trendline(actual_brix, predicted_brix_knn, 'KNN', 'red')
plot_with_trendline(actual_brix, predicted_brix_xgb, 'XGBoost', 'cyan')


# %%
"""## Feature Importance"""


def plot_feature_importance(final_models, model_name, x_labels):
    # Validate that the model exists and has feature_importances_
    if model_name not in final_models:
        raise ValueError(f"Model '{model_name}' not found in final_models.")
    feature_importances = final_models[model_name].feature_importances_
    
    # Validate x_labels length
    if len(x_labels) != len(feature_importances):
        raise ValueError("The length of x_labels must match the number of features in the model.")

    # Ensure x_labels is a NumPy array
    x_labels = np.array(x_labels)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, feature_importances, color='green')
    plt.title(f'{model_name} Feature Importance')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)  # Rotate x-tick labels for clarity
    plt.show()

    # Sort the feature importances and their corresponding wavelengths
    sorted_indices = np.argsort(feature_importances)[::-1]  # Indices for descending order
    sorted_importances = feature_importances[sorted_indices]  # Sorted importance scores
    sorted_x_labels = x_labels[sorted_indices].astype(str)  # Corresponding sorted wavelengths
    
    # Select the top N features
    top_n = 20
    sorted_x_labels_top = sorted_x_labels[:top_n]
    sorted_importances_top = sorted_importances[:top_n]
    
    # Plot only the top N features
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_x_labels_top, sorted_importances_top, color='green')
    plt.title(f'{model_name} Feature Importance (Top {top_n})')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Importance Score')
    
    # Explicitly set tick labels and adjust rotation for better readability
    plt.xticks(rotation=90, fontsize=10)  # Rotate x-ticks
    plt.show()
    
x_labels = wavenumbers_cut

for model_name in ['PLSR', 'SVR', 'Random Forest', 'ANN', 'KNN', 'XGBoost']:
    try:
        plot_feature_importance(final_models, model_name, x_labels)
    except AttributeError as e:
        print(f"Skipping {model_name}: {e}")
        
from sklearn.inspection import permutation_importance

def calculate_permutation_importance(model, X, y):
    result = permutation_importance(model, X, y, scoring='r2', n_repeats=10, random_state=42)
    return result.importances_mean




# # Calculate permutation importance for SVR
# result_svr = permutation_importance(final_models['SVR'], SG_spectra_preprocessed, label_np_1d, n_repeats=10, random_state=42)

# # Extract the importance scores
# feature_importances_svr = result_svr.importances_mean

# x_labels = np.arange(400, 999.4, 2.7)

# # Plot the feature importances for SVR
# plt.figure(figsize=(10, 6))
# plt.bar(x_labels, feature_importances_svr, color='purple')
# plt.title('SVR Feature Importance (Permutation Importance)')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Importance Score')
# plt.xticks(rotation=45)  # Rotate x-tick labels for clarity
# plt.show()

# # Set up a 2x3 grid of subplots (2 rows, 3 columns)
# fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # Adjust figsize for better layout

# # Function to plot error distribution for a given model in a specific axis
# def plot_error_distribution(ax, model_name, model, X, y_true):
#     predictions = model.predict(X)
#     errors = y_true - predictions

#     # Plot error distribution in the specific subplot (ax)
#     sns.histplot(errors, kde=True, color='blue', ax=ax)
#     ax.set_title(f'{model_name} Error Distribution', fontsize = 16)
#     ax.set_xlabel('Prediction Error (Brix)')
#     ax.set_ylabel('Density')

# # Model names for the subplots
# model_names = ['PLSR', 'SVR', 'Random Forest', 'ANN', 'KNN', 'XGBoost']

# # Plot each model's error distribution in the corresponding subplot
# for i, model_name in enumerate(model_names):
#     row = i // 2  # Calculate row position (0 or 1)
#     col = i % 2   # Calculate column position (0, 1, or 2)
#     plot_error_distribution(axs[row, col], model_name, final_models[model_name], MF_spectra_preprocessed, label_np_1d)

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Show the combined figure with all error distributions
# plt.show()