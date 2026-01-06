from imblearn.over_sampling import SMOTE


### SUBMIT THIS FILE TO THE BONUS EXTRA CREDIT GRADESCOPE SECTION ###
def apply_smote(X_train, y_train):
    """
    Apply the SMOTE algorithm to the training data
    For this function, you can use the SMOTE function from imblearn.over_sampling
    Set random_state=42 for Autograding purposes.

    Input: X_train, y_train
    Output: X_train_res, y_train_res
    """
    sm = SMOTE(random_state = 42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res
