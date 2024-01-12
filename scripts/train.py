def RandomForestRegressor():
    # Random Forest Regressor
    print("Random Forest Regressor")
    
def XGBoostRegressor():
    # XGBoost Regressor
    print("XGBoost Regressor")
    
def GradientBoostingRegressor():
    # Gradient Boosting Regressor
    print("Gradient Boosting Regressor")
    
def Lasso():
    # Lasso
    print("Lasso")

def DecisionTreeRegressor():
    # Decision Tree Regressor
    print("Decision Tree Regressor")
    
def ExtraTreesRegressor():
    # Extra Trees Regressor
    print("Extra Trees Regressor")
    
def AdaBoostRegressor():
    # Ada Boost Regressor
    print("Ada Boost Regressor")

models = {
        1: RandomForestRegressor,
        2: XGBoostRegressor,
        3: GradientBoostingRegressor,
        4: Lasso,
        5: DecisionTreeRegressor,
        6: ExtraTreesRegressor,
        7: AdaBoostRegressor,
    }

def train_model(model_id):
    
    if model_id in models:
        # Executes the corresponding model function
        result = models[model_id]()

        print(f"Training Model {models[model_id]}.")
    else:
        print(f"Invalid model_id: {model_id}")
    
train_model(1)
    
    