# https://safjan.com/lime-tutorial/
# LIME (Local Interpretable Model- Agnostic Explanations)

# LIME is a library for explaining the predictions of machine learning models.

# %%
# load packages
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# %%
# split dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
