import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import tree
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("xyz_dat.txt", header=None)
## y <=
df.columns = ['age','workclass','fnlwgt','education','education_num','marital-status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'salary']
df['y'] = np.where(df.salary==' <=50K', 0, 1)
df['workclass'] = np.where(df.workclass==' ?', 'Unknown', df.workclass)

num_columns = [n for n in df.columns[:-1] if (df[n].dtype==int) or (df[n].dtype==float)]
cat_columns = [n for n in df.columns[:-1] if (df[n].dtype==object)]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)

# Random Forests in `scikit-learn` (with N = 100)
rf = RandomForestClassifier(n_estimators=100,
                            random_state=0)
rf.fit(X_train, Y_train)

fn=data.feature_names
cn=data.target_names
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')





# Define numerical and categorical features
numerical_features = ['age', 'income']
categorical_features = ['gender', 'city']

# Define the preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Define a machine learning model (e.g., RandomForestClassifier)
model = RandomForestClassifier()

# Create a pipeline that first applies preprocessing, then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline with the training data
pipeline.fit(X_train, y_train)

# Evaluate the model
score = pipeline.score(X_test, y_test)
print(f"Test score: {score}")


df = pd.read_csv("xyz_dat.txt", header=None)
## y <=
df.columns = ['age','workclass','fnlwgt','education','education_num','marital-status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'salary']
df['y'] = np.where(df.salary==' <=50K', 0, 1)
df['workclass'] = np.where(df.workclass==' ?', 'Unknown', df.workclass)

num_columns = [n for n in df.columns[:-1] if (df[n].dtype==int) or (df[n].dtype==float)]
cat_columns = [n for n in df.columns[:-1] if (df[n].dtype==object)]

sns.histplot(df[(df.y==1)].age, kde=False, color='blue', label='>50K', bins=30, alpha=0.3)
sns.histplot(df[(df.y==0)].age,kde=False, color='orange', label='<=50K', bins=30, alpha=0.3)

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Overlayed Histograms')

# Add a legend
plt.legend()

# Show the plot
plt.show()


sns.countplot(x=df[(df.y==1)].workclass, palette='viridis', alpha=0.3)
sns.countplot(x=df[(df.y==0)].workclass, palette='viridis', alpha=0.3)

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Overlayed Histograms')

# Add a legend
plt.legend()

# Show the plot
plt.show()











import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define columns
num_columns = [n for n in df.columns[:-1] if (df[n].dtype==int) or (df[n].dtype==float)]
cat_columns = [n for n in df.columns[:-1] if (df[n].dtype==object)]

# Step 1: Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_columns),
        ('cat', OneHotEncoder(), cat_columns)
    ])

# Step 2: Create a pipeline with KMeans, preprocessing, and PCA for dimensionality reduction
kmeans_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=2))  # Reduce to 2D for plotting
])

# Step 3: Fit the pipeline to the data
kmeans_pipeline.fit_transform(df)

df['Cluster'] = labels

# Step 6: Plot the clusters using only the numerical features (Age vs Income)
plt.figure(figsize=(8, 6))

# Plot the data points, color-coded by the cluster labels
plt.scatter(df['Age'], df['Income'], c=df['Cluster'], cmap='viridis', s=100)

# Adding labels and title
plt.title('KMeans Clustering (Age vs Income)')
#plt.xlabel('Age')
#plt.ylabel('Income')

# Plot cluster centers (the means of the clusters)
centers = kmeans_pipeline.named_steps['kmeans'].cluster_centers_[:, :2]  # Only first two features: Age, Income
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.legend()
plt.show()












del df['salary']

x_spline = df[['age','']]




df[['age','workclass']].plot(kind="scatter", x='age', y='workclass', color='blue')
plt.show()




## percent <=50K
df[['age','salary']].value_counts()

df[['age','salary']].groupby('salary').agg(
    mn=('age','min')
)
