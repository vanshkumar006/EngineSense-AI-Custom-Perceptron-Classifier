EngineSense-AI-Custom-Perceptron-Classifier

🚀 Overview
EngineSense is a machine learning project that implements a Perceptron Linear Classifier from scratch using Python and NumPy. The model is designed to automate industrial quality control by classifying engine test results as "Pass" or "Fail" based on diagnostic sensor features (X1, X2).
Unlike projects that rely on high-level libraries like Scikit-Learn, this implementation focuses on the first principles of AI, manually handling weight optimization, bias integration, and the Perceptron Learning Rule.

🛠️ Key Technical Features
From-Scratch Implementation: Built without ML frameworks to demonstrate a deep understanding of weight update logic and algorithmic convergence.
Perceptron Learning Rule: Uses the iterative formula  W=W+(η⋅(target−prediction)⋅X) to adjust the decision boundary.

Bias Term Integration: Incorporates an X0 bias term to allow the decision boundary to shift flexibly, ensuring accurate separation even when data isn't centered at the origin.
Vectorized Operations: Built with NumPy for high-performance matrix multiplications (dot products).
Automated Visualization: Includes a custom plotting engine using Seaborn and Matplotlib to visualize the linear hyperplane (Decision Boundary) in real-time.

📊 Model Results
The model was trained and validated on engine diagnostic data with the following outcomes:
Training Speed: Successfully converged in only 2 epochs.
Optimized Weights:
Bias (w0): 9.2
Weight X1(w1): -5.236
Weight X2(w2): 5.562

Predictive Performance:
The model accurately separates the classes. For example:
Input (53.32, 65.00) →Pass ✅
Input (24.18, 10.85) →Fail ❌

🧠 Mathematical Logic
The classifier determines the result based on the linear equation:
f(x)=w0+w1x1+w2x2
If f(x)>0, the result is Pass. If f(x)≤0, the result is Fail. The boundary line is calculated by setting the equation to zero and solving for x2.

💻 Technologies Used
Python
NumPy: Mathematical operations and vectorization.
Pandas: Data management and CSV handling.
Matplotlib & Seaborn: Data visualization and boundary plotting.
