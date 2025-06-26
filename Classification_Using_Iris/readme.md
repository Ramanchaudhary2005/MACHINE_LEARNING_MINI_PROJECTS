# Iris Dataset Classification

This mini project demonstrates classification techniques using the classic Iris dataset. The project covers data visualization, dimensionality reduction, and model selection for classifying iris species.

## Dataset

The dataset used is [`Iris.csv`](Iris.csv), which contains 150 samples of iris flowers with the following features:
- **SepalLengthCm**
- **SepalWidthCm**
- **PetalLengthCm**
- **PetalWidthCm**
- **Species** (target variable: Iris-setosa, Iris-versicolor, Iris-virginica)

## Project Structure

- `Iris.csv` - The dataset file.
- `Classification.py` - Main Python script for data analysis, visualization, and classification.
- `pair_plot.png` - Pair plot visualization of features.
- `violin_plot.png` - Violin plot of Sepal Length by species.
- `pca_lda_plot.png` - PCA and LDA visualization of the dataset.

## Steps Performed

### 1. Data Visualization

- **Pair Plot:** Visualizes pairwise relationships between features, colored by species.
- **Violin Plot:** Shows the distribution of Sepal Length for each species.

### 2. Dimensionality Reduction

- **PCA (Principal Component Analysis):** Reduces feature space to 2D for visualization.
- **LDA (Linear Discriminant Analysis):** Projects data to maximize class separability.

### 3. Model Selection & Classification

- **Label Encoding:** Converts species names to numeric labels.
- **Train-Test Split:** Splits data into training and testing sets (80/20).
- **K-Nearest Neighbors (KNN):** Trains KNN classifiers for k=1 to 24 and evaluates accuracy.
- **Accuracy Evaluation:** Plots training and testing accuracy for different k values.

## How to Run

1. Ensure you have Python 3.x installed.
2. Install required libraries:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the script:
   ```
   python Classification.py
   ```
   This will generate and display the plots, and print model evaluation results.

## Results

- The visualizations help understand feature distributions and class separability.
- KNN classifier accuracy is evaluated for different values of k to select the best model.

## References

- [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

*This project is for educational purposes and hands-on practice with classification algorithms and data