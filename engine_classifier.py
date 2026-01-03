import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Set visual style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class EngineClassifier:
    """
    A Perceptron-based Linear Classifier built from scratch.
    It learns a decision boundary to separate Pass/Fail data.
    """
    def __init__(self, learning_rate=0.1, max_epochs=2000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None # Will be initialized during training
        
    def fit(self, X, y):
        """
        Trains the model using the Perceptron Learning Rule.
        X: Feature matrix (X1, X2)
        y: Target labels ("Pass"/"Fail")
        """
        print(f"--- Starting Training (Max Epochs: {self.max_epochs}) ---")
        
        # 1. Add Bias Term (X0 = 1)
        # This allows the line to shift up/down, not just rotate around (0,0)
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        # 2. Initialize Weights
        # [Bias_Weight, Weight_X1, Weight_X2]
        self.weights = np.array([10.0, 10.0, 10.0])
        
        # 3. Iterative Training Loop
        iteration = 0
        is_converged = False
        
        while not is_converged and iteration < self.max_epochs:
            is_converged = True # Assume we are done until proven otherwise
            
            for i in range(len(X_bias)):
                # Calculate prediction: dot product of features and weights
                prediction_score = np.dot(X_bias[i], self.weights)
                
                # Check actual label
                actual = y.iloc[i]
                
                # --- The Learning Rule ---
                # If Actual is Pass but we predicted Fail (score < 0)
                if actual == "Pass" and prediction_score < 0:
                    self.weights = self.weights + (self.learning_rate * X_bias[i])
                    is_converged = False # We had to fix an error, so keep training
                    
                # If Actual is Fail but we predicted Pass (score > 0)
                elif actual == "Fail" and prediction_score > 0:
                    self.weights = self.weights - (self.learning_rate * X_bias[i])
                    is_converged = False # We had to fix an error
            
            iteration += 1
            
            # Optional: Visualize progress every 100 epochs
            if iteration % 500 == 0:
                print(f"Epoch {iteration}: Weights updated to {self.weights}")

        print(f"Training completed in {iteration} epochs.")
        print(f"Final Weights: {self.weights}")

    def predict(self, X):
        """Predicts Pass/Fail for new data."""
        X_bias = np.c_[np.ones(X.shape[0]), X]
        scores = np.dot(X_bias, self.weights)
        return np.where(scores > 0, "Pass", "Fail")

    def plot_decision_boundary(self, X, y, title="Decision Boundary"):
        """Visualizes the data and the separation line."""
        plt.figure()
        
        # Plot Data Points
        sns.scatterplot(x=X['X1'], y=X['X2'], hue=y, palette={'Pass': 'green', 'Fail': 'red'}, s=100)
        
        # Calculate Line Coordinates based on Weights
        # 0 = w0*1 + w1*x1 + w2*x2  =>  x2 = -(w0 + w1*x1) / w2
        w0, w1, w2 = self.weights
        x_points = np.array([X['X1'].min(), X['X1'].max()])
        y_points = -(w0 + w1 * x_points) / w2
        
        plt.plot(x_points, y_points, color='blue', linewidth=2, label='Decision Boundary')
        plt.title(title)
        plt.xlim(X['X1'].min()-5, X['X1'].max()+5)
        plt.ylim(X['X2'].min()-5, X['X2'].max()+5)
        plt.legend()
        plt.show()

# --- HELPER: Generate Dummy Data (If CSV is missing) ---
def get_data():
    try:
        train = pd.read_csv('engineTest.csv')
        test = pd.read_csv('engineTestCheck.csv')
        return train, test
    except FileNotFoundError:
        print("Warning: CSV not found. Generating synthetic engine data...")
        # Create fake data that is linearly separable
        pass_data = pd.DataFrame({
            'X1': np.random.normal(50, 10, 50),
            'X2': np.random.normal(60, 10, 50),
            'Result': 'Pass'
        })
        fail_data = pd.DataFrame({
            'X1': np.random.normal(20, 10, 50),
            'X2': np.random.normal(20, 10, 50),
            'Result': 'Fail'
        })
        train = pd.concat([pass_data, fail_data]).reset_index(drop=True)
        # Create fake test data
        test = train.copy()
        return train, test

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 1. Load Data
    train_df, test_df = get_data()
    
    # 2. Prepare Features
    X_train = train_df[['X1', 'X2']]
    y_train = train_df['Result']
    
    X_test = test_df[['X1', 'X2']]
    
    # 3. Initialize and Train Model
    model = EngineClassifier(learning_rate=0.1)
    
    # Show status before training
    print("Initializing model...")
    
    model.fit(X_train, y_train)
    
    # 4. Visualize Training Result
    model.plot_decision_boundary(X_train, y_train, title="Training Data: Final Decision Boundary")
    
    # 5. Predict on Test Data
    test_df['Predictions'] = model.predict(X_test)
    
    # 6. Visualize Test Result
    print("\nTest Data Predictions (First 5):")
    print(test_df[['X1', 'X2', 'Predictions']].head())

        
    model.plot_decision_boundary(X_test, test_df['Predictions'], title="Test Data: Model Predictions")

