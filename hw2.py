from pandas import *
from numpy import * 
from seaborn import heatmap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures , LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1
data_frame = read_csv('Housing.csv')  
data_frame.info()
print(data_frame.head())


#2 
label_encoder = LabelEncoder()
for column in data_frame.select_dtypes(include='object').columns:
    data_frame[column] = label_encoder.fit_transform(data_frame[column])

missing_v = data_frame.isnull().sum()
missing_v = missing_v[missing_v > 0]
print(missing_v)

def detect_outliers_iqr(df):
    outlier_info = {}
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            outlier_info[col] = {
                'count': len(outliers),
                'percent': round(len(outliers) / len(df) * 100, 2)
            }

    return outlier_info
outliers_summary = detect_outliers_iqr(data_frame)
for col, info in outliers_summary.items():
    print(f"🔍 Column: {col} | Outliers: {info['count']} rows ({info['percent']}%)")

def remove_outliers_iqr(df):
    df_clean = df.copy()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        iqr = Q3 - Q1
        lower_bound = Q1 - 1.5 * iqr
        upper_bound = Q3 + 1.5 * iqr

        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean
data_frame_cleaned = remove_outliers_iqr(data_frame)
modified_df = data_frame_cleaned

for column in modified_df.columns:
    missing_value = modified_df[column].isnull().sum()
    if missing_value > 0:
        if missing_value > 1000:
            modified_df.drop(columns=[column], inplace=True)
        else:
            if modified_df[column].dtype == 'float64' and modified_df[column].dtype == 'int64':
                median_data = modified_df[column].median()
                modified_df[column].fillna(median_data, inplace=True)
            else:
                mode_data = modified_df[column].mode()[0]
                modified_df[column].fillna(mode_data, inplace=True)
    top_freq_ratio = modified_df[column].value_counts(normalize=True).values[0]
    if top_freq_ratio > 0.70:
        modified_df.drop(columns=[column], inplace=True)
        
modified_df.drop(columns=["Order"], inplace=True)

missing_data2 = modified_df.isnull().sum()
missing_data2 = missing_data2[missing_data2 > 0]


#3
print(modified_df.describe())

#4
correlation_matrix = modified_df.corr(numeric_only=True)
correlation_with_price = correlation_matrix['SalePrice']
sorted_corr = correlation_with_price.sort_values(ascending=False)
strong_corr = sorted_corr[abs(sorted_corr) > 0.5]
print(strong_corr)

#5
plt.figure(figsize=(20,15))
heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

important_features = strong_corr.drop('SalePrice').index

top_features = important_features[:4]

for feature in top_features:
    plot = sns.jointplot(data=modified_df, x=feature, y='SalePrice', kind='reg', height=7, color='royalblue')
    plot.fig.suptitle(f'Jointplot of {feature} vs SalePrice', y=1.02)
    plt.show()

#6
X = modified_df.drop('SalePrice', axis=1)
y = modified_df['SalePrice']
X = get_dummies(X, drop_first=True)
scores = []
k_values = range(1, min(len(X.columns), 46))  
for k in k_values:
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    model = LinearRegression()
    cv_score = cross_val_score(model, X_selected, y, cv=5, scoring='r2').mean()
    scores.append(cv_score)

plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, marker='o')
plt.xlabel('Number of Selected Features (k)')
plt.ylabel('Average Model Accuracy (R²)')
plt.title('Selecting the Best Number of Features with SelectKBest')
plt.grid(True)
plt.show()

#7 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#8
# linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Lasso Regression
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Polynomial Regression (degree=2 for example)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)

#9
def evaluate_model(model, X_test, y_test, model_name="Model"):
    predictions = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    rms = sqrt(mean_squared_error(y_test, predictions))
    print(f"{model_name} => R²: {r2:.4f}, RMS: {rms:.2f}")

evaluate_model(linear_model, X_test, y_test, "Linear Regression")
evaluate_model(lasso_model, X_test, y_test, "Lasso Regression")
evaluate_model(ridge_model, X_test, y_test, "Ridge Regression")
evaluate_model(poly_model, X_test, y_test, "Polynomial Regression")

def plot_predictions(model, X_test, y_test, model_name="Model"):
    predictions = model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color='dodgerblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_predictions(linear_model, X_test, y_test, "Linear Regression")
plot_predictions(lasso_model, X_test, y_test, "Lasso Regression")
plot_predictions(ridge_model, X_test, y_test, "Ridge Regression")
plot_predictions(poly_model, X_test, y_test, "Polynomial Regression")

#10
random.seed(0)
X = sort(random.rand(100, 1) * 4 - 2, axis=0)
y = sin(X).ravel() + random.randn(100) * 0.2


degrees = [1, 3, 15]
plt.figure(figsize=(15, 4))
for i, d in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.subplot(1, 3, i + 1)
    plt.scatter(X, y, color='gray', label='Data')
    plt.plot(X, y_pred, color='red', label=f'Degree {d}')
    plt.title(f"Degree {d}")
    plt.legend()
plt.suptitle("Bias-Variance Tradeoff with Polynomial Models")
plt.tight_layout()
plt.show()
