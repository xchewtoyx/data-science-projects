# Data Science Projects

The following are 8-10 ideas for end-of-semester projects designed to demonstrate understanding of the key pillars—Statistics, Calculus, Linear Algebra, and Practical Data Science (using Python)—across a 3-year undergraduate Data Science curriculum, structured into three semesters per year.

***

### Year 1: Building the Foundation

Year 1 focuses on establishing a **firm foundation** in Python programming, descriptive statistics, probability theory, and fundamental linear algebra and calculus concepts [1, 2]. Students should prioritize implementing core concepts *from scratch* where possible, using foundational libraries like base Python and NumPy/Pandas for data handling [3, 4].

#### Semester 1: Introduction to Data and Basic Calculations

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. Vector Algebra Toolkit** | LA, Python | Implement Python functions (using core Python lists or NumPy arrays) to perform vector addition, scalar multiplication, and calculate linear combinations ($\mathbf{c} \mathbf{v} + \mathbf{d} \mathbf{w}$) [5-8]. |
| **2. Simple Linear System Solver** | LA, Python | Write a Python program to solve a $3 \times 3$ system of linear equations ($\mathbf{A} \mathbf{x} = \mathbf{b}$) using an explicit implementation of Gaussian elimination (Elementary Row Operations) and back substitution, demonstrating the direct solution method [9-12]. |
| **3. Descriptive Statistics Report** | Stats, Python/DS | Choose a real dataset (like the `College` or `Auto` data set) and generate a numerical summary, including mean, standard deviation, and interquartile range (IQR) [13-16]. Use Python to produce histograms and boxplots to visualize the distribution shape [17, 18]. |
| **4. Probability Simulation Engine** | Stats, Python | Implement a Python simulation (e.g., coin flips or dice rolls) to estimate the probability of a compound event (e.g., getting heads 5 times out of 10), illustrating the frequentist interpretation of probability [19-21]. |
| **5. Core Python Data Wrangling** | Python/DS | Demonstrate fluent command of Python fundamentals by writing a program that processes nested lists or dictionaries, using control flow (if/elif/else), loops, and functions to transform data before exporting to a clean format [22, 23]. |
| **6. Basic NumPy Array Operations** | Python/DS, LA | Use NumPy's `ndarray` objects to perform basic element-wise arithmetic and apply Universal Functions (`ufuncs`) to large arrays, highlighting the performance advantages over standard Python lists [24-26]. |
| **7. Basic Calculus Function Analyzer** | Calculus, Python | Define a simple polynomial function $f(x)$ in Python. Calculate the average rate of change between two points, and find the instantaneous rate of change (derivative) at a single point, illustrating the concept of the derivative as the rate of change or slope [27, 28]. |
| **8. Database Query Project** | Python/DS | Use Python to connect to a simple database (e.g., relational SQL) and demonstrate foundational skills in data storage and management by running basic queries to retrieve, filter, and structure data [29]. |

#### Semester 2: Data Structures, Linear Independence, and Simple Inference

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. Pandas Data Cleaning and Munging** | Python/DS, Stats | Load a complex, unstructured dataset into a Pandas DataFrame, then apply methods for detecting and handling missing values (`isnull()`, `dropna()`, `fillna()`), removing duplicate records, and renaming columns, documenting the wrangling process [30-33]. |
| **2. Linear Independence Tester** | LA, Python | Write a program to test whether a given set of vectors is linearly independent by finding the nullspace solution to the equation $\mathbf{C}\mathbf{x} = \mathbf{0}$. If multiple solutions exist, the columns are dependent [34, 35]. |
| **3. Linear Regression Model (Statsmodels)** | Stats, Python/DS | Fit a multiple linear regression model using `statsmodels` (or `scikit-learn` for prediction focus) to predict a quantitative response variable. Report on the statistical significance (p-values) of the predictors and interpret the $\text{R}^2$ value for goodness of fit [36-39]. |
| **4. Differentiation Rule Implementation** | Calculus, Python | Implement the basic rules of differentiation (sum, product, and quotient rules) in a rudimentary Python program (or symbolic math library) to find the derivative of composed functions [40-42]. |
| **5. Sampling Distribution Demonstration** | Stats, Python | Write a program to simulate the sampling distribution of the sample mean ($\bar{x}$) for a non-normal parent population, demonstrating how the distribution approaches the Normal distribution as sample size increases, confirming the Central Limit Theorem [43-45]. |
| **6. Single-Variable Optimization** | Calculus, Python | Implement a numerical method (like a simple search algorithm or rudimentary gradient descent approach for univariate functions) to find the local minimum or maximum of a specified function, utilizing the concept that the derivative is zero at extrema [46, 47]. |
| **7. Matrix Inverse/Factorization using LU** | LA, Python | Use NumPy to find the LU decomposition of a matrix [48]. Use this decomposition to efficiently solve a system of linear equations, demonstrating the computational utility of matrix factorization [48, 49]. |
| **8. Data Visualization with Matplotlib/Seaborn** | Python/DS | Create informative visualizations (scatterplots, bar charts, boxplots) using Matplotlib and Seaborn for data exploration and reporting, applying appropriate labeling and annotation techniques [50-52]. |

#### Semester 3: Foundational ML, Subspaces, and Integration

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. Column Space and Solubility** | LA, Python | Analyze a matrix $A$ and a vector $\mathbf{b}$. Determine whether $\mathbf{b}$ lies in the column space $\mathbf{C}(A)$ by attempting to solve $\mathbf{A} \mathbf{x} = \mathbf{b}$. Conclude if the system is solvable based on whether $\mathbf{b}$ is a linear combination of the columns [53-55]. |
| **2. K-Nearest Neighbors (KNN) Classifier** | ML/DS, Python | Implement the KNN algorithm using `scikit-learn` to classify a set of data points (e.g., the Iris dataset). Evaluate the performance using a simple train/test split and report accuracy [56-59]. |
| **3. Confidence Sets Construction** | Stats, Python | Construct and interpret a $1-\alpha$ confidence interval for the population mean $\mu$ for a real-world dataset, demonstrating the concept of a confidence set as an interval that traps the true value with a given frequency [60-62]. |
| **4. Introduction to Integration** | Calculus, Python | Use numerical integration (e.g., trapezoidal rule, implemented in Python from scratch or via SciPy) to estimate the area under the curve of a non-elementary function over a given interval, reflecting the geometric definition of the integral [63-66]. |
| **5. Time-Series Resampling Analysis** | Python/DS | Load a simple time-series dataset. Use Pandas functionality (`pd.to_datetime`, `resample`) to change the frequency of the data (e.g., daily to weekly) and calculate rolling averages or moving window functions, addressing practical data constraints [67-69]. |
| **6. Simple Least Squares Fitting** | LA, Stats, Python | Given an overdetermined system ($\mathbf{A} \mathbf{x} \approx \mathbf{b}$), use the least squares method ($\mathbf{A}^T \mathbf{A} \mathbf{x} = \mathbf{A}^T \mathbf{b}$) to find the best approximate solution $\hat{\mathbf{x}}$. Use NumPy to solve the resulting matrix equation [70, 71]. |
| **7. Polynomial Regression Model Selection** | Stats, Python/DS | Fit polynomial regression models of different degrees (e.g., degree 1 to 5) to a simulated dataset. Use a cross-validation method (e.g., LOOCV or $k$-fold cross-validation) to select the optimal model complexity, analyzing the trade-off between bias and variance [72-75]. |
| **8. Non-Linear Curve Fitting** | Calculus, Python | Use derivatives to analyze a rational or complex function (e.g., $f(x) = \sin(x^2)$). Calculate the derivatives using the chain rule and find the extrema, visualizing the results alongside the original function [76, 77]. |

***

### Year 2: Intermediate Theory and Complex Modeling

Year 2 moves into more advanced mathematical concepts necessary for modern machine learning, focusing heavily on optimization, matrix structure, and inferential modeling [78, 79].

#### Semester 4: Optimization, Subspaces, and Foundational ML

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. Gradient Descent Implementation for LR** | Calculus, Python/DS | Implement Stochastic Gradient Descent (SGD) from scratch to train a simple Linear Regression or Logistic Regression model. Demonstrate how the gradient (the vector of partial derivatives) is used to iteratively update parameters and minimize the loss function [13, 42, 66, 80-82]. |
| **2. Fundamental Subspaces Analysis** | LA, Python | Analyze a rectangular matrix $A$. Find the dimension and bases for the four fundamental subspaces: $\mathbf{C}(A)$, $\mathbf{N}(A)$, $\mathbf{C}(A^T)$, and $\mathbf{N}(A^T)$. Discuss the orthogonality relationship between the row space and the nullspace [83-85]. |
| **3. Feature Engineering and Transformation** | Python/DS, Stats | Using Pandas, apply advanced data transformation techniques such as creating dummy variables (`pd.get_dummies`), binning quantitative data, and scaling/normalizing features, preparing a raw dataset for complex machine learning models [86, 87]. |
| **4. Multivariable Function Analysis** | Calculus, LA | Define a vector-scalar function $f(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}$. Calculate its partial derivatives and compute the gradient vector $\nabla f(\mathbf{x})$. Use this to determine the direction of the steepest ascent/descent from a specific point [88, 89]. |
| **5. Hypothesis Testing for Two Populations** | Stats, Python | Using two independent samples, perform a two-sample $t$-test to compare the means of two populations, ensuring that all assumptions (e.g., simple random sample, normality) are checked using Python tools (e.g., visualization/statistical tests) [90, 91]. |
| **6. Non-linear Feature Modeling** | ML/DS, Python | Apply non-linear modeling techniques (e.g., fitting cubic polynomials or step functions) to model a predictor-response relationship in a dataset, analyzing whether the non-linear approach improves prediction over a simple linear fit [92, 93]. |
| **7. Orthonormal Basis Construction** | LA, Python | Implement the Gram-Schmidt orthogonalization process in Python to transform a set of linearly independent vectors into a set of orthonormal vectors, forming an orthonormal basis [94, 95]. |
| **8. Data Aggregation via Split-Apply-Combine** | Python/DS | Utilize the Pandas `groupby()` facility to perform complex group operations (split-apply-combine) on a DataFrame, aggregating data by various criteria and applying custom functions to groups for summary statistics [96-99]. |

#### Semester 5: Matrix Factorization, MLE, and Advanced Optimization

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. Maximum Likelihood Estimation (MLE)** | Stats, Calculus, Python | Implement MLE to estimate the parameter $\theta$ for a simple parametric model (e.g., $p$ for Bernoulli/Binomial or $\mu, \sigma^2$ for Normal) based on observed data. Use optimization techniques or analytic solutions if available [100, 101]. |
| **2. Least-Squares Projection Implementation** | LA, Python | Write a Python program that explicitly solves the least-squares problem $\mathbf{A} \mathbf{x} \approx \mathbf{b}$ for an overdetermined system. Calculate the projection $\mathbf{p} = \mathbf{A}\hat{\mathbf{x}}$ and verify that the error vector $\mathbf{e} = \mathbf{b} - \mathbf{p}$ is orthogonal to the column space $\mathbf{C}(A)$ [84]. |
| **3. Multivariable Optimization using Gradient** | Calculus, Python | Implement the *vectorized* gradient descent algorithm to minimize a multivariable cost function $L(\mathbf{w})$ (e.g., logistic regression loss), utilizing NumPy for fast matrix operations [102, 103]. |
| **4. Simple Bayesian Inference** | Stats, Python | Implement a basic Bayesian inference problem (e.g., updating a prior distribution for a coin's probability $p$ given new observed flips). Calculate and visualize the resulting posterior distribution [104-106]. |
| **5. Multivariable Differentiation and Jacobian** | Calculus, LA | For a given vector-vector function $\mathbf{f}(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}^m$ (similar to a single layer of a neural network), calculate the Jacobian matrix $D\mathbf{f}(\mathbf{x})$ analytically, demonstrating the concept of total differentiation [107, 108]. |
| **6. Singular Value Decomposition (SVD) Basis** | LA, Python | Use NumPy to compute the SVD of a matrix $A$. Use the resulting matrices $U$ and $V$ to identify orthonormal bases for the four fundamental subspaces [109, 110]. |
| **7. Kernelized Classification (SVM/Python)** | ML/DS, Python | Apply a Support Vector Machine (SVM) using a non-linear kernel (e.g., polynomial or radial basis) to a classification task. Explore the effect of hyperparameters like degree and $C$ on the resulting decision boundary [59, 111, 112]. |
| **8. Data Merging and Relational Operations** | Python/DS | Integrate data from multiple sources (simulating multiple tables or files) using Pandas merge operations (`pd.merge()` or `pd.concat()`), utilizing different join types (inner, left, outer) and addressing index alignment issues [113-116]. |

#### Semester 6: Convexity, Algorithms, and Intermediate ML

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. Linear Programming (LP) Application** | LA, Optimization | Formulate a real-world resource allocation or scheduling problem (e.g., airline scheduling, portfolio optimization basics) as a Linear Program, defining the objective function and constraints. Use a Python solver to find the optimal solution [71, 117-119]. |
| **2. Principal Component Analysis (PCA)** | LA, ML, Python | Implement PCA using $\mathbf{scikit}$-learn on a multi-dimensional dataset to reduce its dimensionality. Relate the principal components found to the singular vectors obtained from the SVD or eigenvectors of the covariance matrix [59, 120-122]. |
| **3. Logistic Regression and ROC Analysis** | Stats, ML/DS, Python | Implement Logistic Regression for a binary classification task. Evaluate performance using metrics relevant to classification (e.g., confusion matrix, accuracy, precision, recall) and generate an ROC curve [34, 123]. |
| **4. Intro to Dynamic Programming** | Algorithms, Python | Solve a foundational problem (e.g., finding the longest common subsequence or the minimum cost path in a grid) using the dynamic programming paradigm, emphasizing the storage and reuse of solutions to subproblems [124, 125]. |
| **5. Nonparametric Distribution Estimation** | Stats, Python | Estimate the cumulative distribution function (CDF) or probability density function (PDF) of a dataset using nonparametric methods (e.g., empirical CDF or simple kernel density estimation), comparing the results to parametric assumptions [126, 127]. |
| **6. Quadratic Optimization Problem** | LA, Optimization | Formulate and solve a simple quadratic optimization problem (which is convex) subject to linear constraints, demonstrating the power of solving convex problems efficiently [71, 128]. |
| **7. Dimensionality Reduction Visualization** | LA, ML, Python | Apply PCA or manifold learning techniques to a complex dataset (e.g., image data or a feature set with many variables). Visualize the resulting lower-dimensional data structure, using Python for the implementation and visualization [120, 121]. |
| **8. Regression Model Regularization** | ML/DS, Stats, Python | Apply regularization techniques (Lasso or Ridge regression) to a linear model using $\mathbf{scikit}$-learn. Use cross-validation to tune the penalty parameter $\lambda$ and discuss how regularization combats overfitting [59, 129, 130]. |

***

### Year 3: Advanced Applications and Specialization

Year 3 focuses on modern, high-complexity models like Deep Learning, ensemble methods, unsupervised learning, and computationally intractable algorithms [56, 131, 132].

#### Semester 7: Unsupervised Learning, Ensemble Methods, and Complexity

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. Clustering Analysis and Interpretation** | ML/DS, Stats, Python | Apply and compare $K$-means clustering and Hierarchical Clustering to a multi-feature dataset. Use methods (like the elbow method) to justify the choice of $K$ and interpret the resulting clusters in terms of the original variables [59, 133-135]. |
| **2. Decision Trees and Random Forests** | ML/DS, Python | Use ensemble methods (Random Forests or Gradient Boosting) to solve a regression or classification task. Analyze the model results to determine feature importance, contrasting the performance with a single decision tree [136-139]. |
| **3. Complex Dynamic Programming Solution** | Algorithms, Python | Implement a full dynamic programming solution to a classic problem like the Knapsack Problem or RNA secondary structure prediction, demonstrating optimization over sequences or structures [125, 140]. |
| **4. Advanced Correlation Analysis** | Stats, Python | Investigate non-linear correlation in a dataset by modeling relationships using regression splines (e.g., natural splines) or Generalized Additive Models (GAMs), comparing this to linear correlation measures [92, 136]. |
| **5. Model Selection via Stepwise Regression** | Stats, Python | Use forward or backward stepwise selection to identify a satisfactory subset of predictors for a linear model on the training set, comparing the resulting test error with a full least squares model [141-143]. |
| **6. Nonparametric Hypothesis Testing** | Stats, Python | Implement a nonparametric test (e.g., sign test or Wilcoxon rank-sum test) appropriate for data that violates the assumptions of parametric tests (like normality), discussing when these alternatives are necessary [61, 91]. |
| **7. Large-Scale Data Wrangling Simulation** | Python/DS | Design a data pipeline simulating a large-scale process (e.g., word count using MapReduce concepts or large file processing), focusing on efficient memory management and optimized NumPy/Pandas techniques [144, 145]. |
| **8. Local Search Heuristics** | Algorithms, Python | Implement a local search heuristic (e.g., Metropolis algorithm or Simulated Annealing) to find an approximate solution to a known NP-hard problem like Maximum Cut, acknowledging that provable guarantees may not exist [140, 146]. |

#### Semester 8: Deep Learning and Network Analysis

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. Basic Neural Network for Classification** | DL, Calc, Python | Build a simple multilayer neural network (MLP) for classification. Implement the forward pass, and conceptually explain how the backpropagation algorithm uses the chain rule to compute gradients for weight updates during stochastic gradient descent [76, 102, 132, 147-149]. |
| **2. Convolutional Neural Network (CNN) Application** | DL, Python/DS | Use a modern Python library (e.g., PyTorch) to load and utilize a pretrained CNN model to classify images, demonstrating practical expertise in deep learning models for computer vision [150, 151]. |
| **3. Network Flow and Kirchhoff's Laws** | LA, Algorithms, Python | Construct the incidence matrix $A$ for a graph (e.g., electrical circuit or flow network). Use linear algebra (finding the nullspace $\mathbf{N}(A)$ and left nullspace $\mathbf{N}(A^T)$) to analyze the flow balance and potential differences in the network, demonstrating the link between matrices and graph theory [152-154]. |
| **4. Advanced Visualization (Interactive/Web)** | Python/DS | Use a library like Bokeh or D3.js (or its Python wrappers) to create an interactive, sophisticated data visualization for the web, demonstrating data storytelling skills [155-157]. |
| **5. Optimization Constraints and Convexity** | Optimization, LA | Research a non-convex problem (e.g., Maximum-Cut) and discuss how it can be approximated using convex optimization techniques like Semidefinite Programming (SDP) relaxations, citing the field's complexity [71, 158, 159]. |
| **6. Language Modeling with RNN/LSTM** | DL, Python | Use a recurrent neural network (RNN) or a Long Short-Term Memory (LSTM) network to model a simple sequential dataset (e.g., short text sequences or time series data like NYSE closing prices) [147, 148, 151, 160]. |
| **7. Approximation Algorithms** | Algorithms, Python | Analyze a known computationally intractable problem (like the Knapsack Problem). Implement an approximation algorithm (e.g., using Linear Programming approximations) and evaluate its performance relative to an optimal solution on small instances [146]. |
| **8. Numerical Linear Algebra Efficiency** | LA, Python | Write a benchmarking script to compare the computational efficiency (runtime) of solving $\mathbf{A} \mathbf{x} = \mathbf{b}$ using direct matrix inversion ($\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$), LU decomposition, and specialized iterative methods provided by NumPy/SciPy, validating that inversion is generally slower [25, 48, 161]. |

#### Semester 9: Capstone and Professional Practice

| Project Idea | Core Pillars Demonstrated | Description & Source Justification |
| :--- | :--- | :--- |
| **1. End-to-End Data Science Capstone** | All Pillars | Execute a complete, end-to-end data science project on a substantial real-world dataset (e.g., from marketing, finance, or biology [60]). This includes data acquisition, cleaning, feature engineering, comparative modeling (LR, SVM, RF), model validation, test error reporting, and a final inference/prediction report [162, 163]. |
| **2. Robust Statistics and Outlier Analysis** | Stats, Python | Conduct a study focusing on the impact of outliers or contamination on statistical estimates (e.g., mean and variance). Implement robust estimation techniques (like median or IQR-based methods) and compare their stability against classical estimators [164, 165]. |
| **3. Deep Learning Overfitting and Regularization** | DL, Stats, Python | Experiment with training a deep neural network until zero training error (interpolation) is achieved. Apply regularization techniques (e.g., ridge penalty or dropout) and early stopping using SGD to mitigate overfitting, comparing test set performance in each case [130, 166-168]. |
| **4. Statistical Learning Theory Exploration** | Stats, ML | Provide a theoretical review and practical exploration of the concepts of bias-variance tradeoff and the curse of dimensionality, illustrating these effects using appropriate simulated data and modeling examples [169-171]. |
| **5. Web Scraping and Data Acquisition Pipeline** | Python/DS | Build a functional Python script to scrape data from a structured source online, store the results efficiently (e.g., JSON or Parquet), and integrate the raw, acquired data into a Pandas DataFrame for analysis [31, 172-174]. |
| **6. Collaborative Filtering Recommender System** | ML/DS, LA, Python | Build a simple item-based or user-based collaborative filtering system using techniques like matrix factorization, demonstrating its use in suggesting relevant items or connections to users [135, 175]. |
| **7. Advanced Time Series Modeling (RNN/AR)** | DL, Stats, Python | Apply Recurrent Neural Networks (RNNs) or autoregressive models to forecast future values of a time series, comparing performance and complexity against simpler linear models [151]. |
| **8. Comparative Convex/Non-Convex Solver** | Optimization, LA | Compare the difficulty and efficiency of solving a typical convex problem (e.g., standard Least Squares) versus a known non-convex optimization problem (e.g., a simple global optimization case), discussing the theoretical guarantees available for each [71, 128]. |

References: