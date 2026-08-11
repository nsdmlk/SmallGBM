import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from smallgbm import SmallGBMClassifier, SmallGBMRegressor

def test_classifier_fit_predict():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60, random_state=42)
    
    model = SmallGBMClassifier(n_estimators=50, max_depth=3, min_samples_leaf=3)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    assert len(pred) == len(y_test)
    assert proba.shape == (len(y_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert set(pred).issubset({0, 1})

def test_classifier_reproducibility():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    model1 = SmallGBMClassifier(random_state=42)
    model2 = SmallGBMClassifier(random_state=42)
    
    model1.fit(X, y)
    model2.fit(X, y)
    
    proba1 = model1.predict_proba(X)
    proba2 = model2.predict_proba(X)
    
    assert np.allclose(proba1, proba2)

def test_regressor_fit_predict():
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60, random_state=42)
    
    model = SmallGBMRegressor(n_estimators=50, max_depth=3, min_samples_leaf=3)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    
    assert len(pred) == len(y_test)
    assert pred.ndim == 1

def test_different_sigma_prior():
    X, y = make_classification(n_samples=80, n_features=5, random_state=42)
    
    model = SmallGBMClassifier(sigma_prior=10.0)
    model.fit(X, y)
    proba = model.predict_proba(X)
    
    assert proba.shape == (80, 2)
    assert not np.any(np.isnan(proba))