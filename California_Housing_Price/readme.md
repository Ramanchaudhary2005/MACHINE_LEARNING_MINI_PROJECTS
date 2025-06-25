# California Housing Price Prediction

This project predicts California housing prices using various regression models. The dataset is based on the California housing data and includes features such as location, median income, and proximity to the ocean.

## Project Structure

- [`regression.py`](regression.py): Main script for data preprocessing, model training, evaluation, and visualization.
- [`housing.csv`](housing.csv): Dataset containing California housing data.
- [`readme.md`](readme.md): Project documentation.

## Dataset

The dataset (`housing.csv`) contains the following columns:

- `longitude`: Longitude of the block.
- `latitude`: Latitude of the block.
- `housing_median_age`: Median age of houses in the block.
- `total_rooms`: Total number of rooms.
- `total_bedrooms`: Total number of bedrooms.
- `population`: Population of the block.
- `households`: Number of households.
- `median_income`: Median income in the block.
- `median_house_value`: Median house value (target variable).
- `ocean_proximity`: Proximity to the ocean (categorical).

## Workflow

1. **Data Loading**  
   The script loads the dataset using pandas.

2. **Data Cleaning**  
   - Missing values in `total_bedrooms` are imputed with the mean.

3. **Feature Encoding**  
   - The categorical column `ocean_proximity` is encoded using `LabelEncoder`.

4. **Feature Scaling**  
   - All features are standardized using `StandardScaler`.

5. **Exploratory Data Analysis**  
   - Boxen plots are generated for each feature to visualize distributions and outliers.

6. **Train-Test Split**  
   - The data is split into training and testing sets (80/20 split).

7. **Model Training & Evaluation**  
   - **Linear Regression**: Trained and evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
   - **Random Forest Regressor**: Trained and evaluated with the same metrics.

8. **Visualization**  
   - Scatter plot of `median_income` vs. `median_house_value` with regression predictions.

## How to Run

1. **Install Dependencies**

   ```sh
   pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
   ```

2. **Place the Dataset**

   Ensure `housing.csv` is in the same directory as `regression.py`.

3. **Run the Script**

   ```sh
   python regression.py
   ```

   The script will output evaluation metrics and display a plot.

## Results

The script prints the following metrics for each model:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Notes

- The script uses both linear and ensemble models for regression.
- You can extend the script to include more models or hyperparameter tuning.
- The dataset may contain missing values; these are handled automatically.

## References

- [scikit-learn California housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [pandas documentation](https://pandas.pydata.org/)
-