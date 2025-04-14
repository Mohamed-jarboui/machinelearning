from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_metrics(y_true, y_pred):
    """Calculate and print regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}") 
    print(f"R² Score: {r2:.2f}")
    
    return {"MAE": mae, "MSE": mse, "R2": r2}