from __future__ import annotations

import hashlib
import json
import re
import requests

from config import OLLAMA_MODEL, OLLAMA_TAGS_URL, OLLAMA_URL, QUESTIONS_PER_VIVA
from utils.text import clean, concept_terms, format_experiment_title, parse_json_object

_EXPERIMENT_QUESTION_CACHE: dict[str, list[dict]] = {}

_EVAL_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "over", "after", "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "what", "which",
    "who", "whom", "this", "that", "these", "those", "how", "why", "where", "when",
    "you", "your", "we", "our", "i", "me", "my", "it", "its", "or", "and", "but",
    "if", "not", "no", "yes", "experiment", "question", "answer", "explain", "describe",
}

# Unrelated general computer/hardware/programming terms that must NEVER appear in viva questions
_BANNED_UNRELATED_TERMS = {
    "computer", "client", "server", "operating system", "hardware", "software",
    "cpu", "ram", "motherboard", "keyboard", "monitor", "programming language",
    "compiler", "interpreter", "syntax error", "import statement", "python library",
    "pip install", "operating systems", "hard drive", "source code",
}

# Rich, comprehensive, topic-specific conceptual questions strictly grounded in each experiment
_TOPIC_QUESTION_BANKS: dict[str, list[dict]] = {
    "Simple Linear Regression": [
        {
            "question": "In simple linear regression, what do the slope (m) and intercept (c) represent geometrically on the fitted line?",
            "expected_concepts": ["slope represents rate of change", "intercept represents y-value when x is zero", "line equation y=mx+c"],
            "difficulty": "easy",
            "topic_aspect": "mathematical_formula",
        },
        {
            "question": "Why is the Ordinary Least Squares (OLS) criterion used to fit the line, and what sum does it minimize?",
            "expected_concepts": ["ordinary least squares", "sum of squared residuals", "minimizing error", "best fit line"],
            "difficulty": "medium",
            "topic_aspect": "fitting_method",
        },
        {
            "question": "How do you interpret an R-squared value of 0.85 when evaluating the fitted regression model on your dataset?",
            "expected_concepts": ["r-squared", "85% variance explained", "goodness of fit", "dependent variable explained by independent variable"],
            "difficulty": "medium",
            "topic_aspect": "model_evaluation",
        },
        {
            "question": "What is the key difference between the independent variable (X) and the dependent variable (y) in this regression experiment?",
            "expected_concepts": ["independent variable is predictor", "dependent variable is response outcome", "features vs target"],
            "difficulty": "easy",
            "topic_aspect": "dataset_variables",
        },
        {
            "question": "How does an outlier data point with an extreme value affect the slope of the simple linear regression line?",
            "expected_concepts": ["outlier sensitivity", "leverage point distorts slope", "increases residual sum of squares"],
            "difficulty": "medium",
            "topic_aspect": "outlier_interpretation",
        },
        {
            "question": "In what practical scenario would a simple linear regression model be inadequate, requiring a non-linear or multi-feature approach?",
            "expected_concepts": ["non-linear relationships", "multiple influencing factors", "underfitting", "curved patterns"],
            "difficulty": "medium",
            "topic_aspect": "practical_application",
        },
    ],
    "Multiple Linear Regression": [
        {
            "question": "In multiple linear regression, how do multiple regression coefficients quantify the individual contribution of each feature?",
            "expected_concepts": ["partial regression coefficients", "feature weights", "holding other variables constant", "multivariate prediction"],
            "difficulty": "medium",
            "topic_aspect": "mathematical_formula",
        },
        {
            "question": "Why is multicollinearity between independent variables problematic, and how does it affect coefficient estimation?",
            "expected_concepts": ["multicollinearity", "correlated predictors", "unstable coefficients", "variance inflation"],
            "difficulty": "medium",
            "topic_aspect": "variable_correlation",
        },
        {
            "question": "Why is Adjusted R-squared preferred over regular R-squared when comparing multiple regression models with extra features?",
            "expected_concepts": ["adjusted r-squared", "penalty for redundant features", "degrees of freedom", "overfitting prevention"],
            "difficulty": "medium",
            "topic_aspect": "model_evaluation",
        },
        {
            "question": "What role does feature standardization or normalization play when fitting multiple linear regression via gradient descent?",
            "expected_concepts": ["feature scaling", "standardization", "gradient descent convergence", "different feature scales"],
            "difficulty": "medium",
            "topic_aspect": "data_preprocessing",
        },
        {
            "question": "How do Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) help quantify prediction error in this experiment?",
            "expected_concepts": ["mean squared error", "rmse", "residual magnitude", "penalizing large errors"],
            "difficulty": "easy",
            "topic_aspect": "error_metrics",
        },
        {
            "question": "What assumption is made regarding the distribution of residuals (errors) in multiple linear regression?",
            "expected_concepts": ["normal distribution of errors", "zero mean", "constant variance homoscedasticity", "independence of errors"],
            "difficulty": "medium",
            "topic_aspect": "model_assumptions",
        },
    ],
    "k-Nearest Neighbors (KNN)": [
        {
            "question": "In the KNN algorithm, how does selecting a very small k (like k=1) versus a large k influence the decision boundary?",
            "expected_concepts": ["hyperparameter k", "k=1 causes overfitting/noise sensitivity", "large k causes oversmoothing", "bias variance tradeoff"],
            "difficulty": "medium",
            "topic_aspect": "hyperparameter_tuning",
        },
        {
            "question": "Why is feature scaling strictly required before computing Euclidean distance between samples in KNN?",
            "expected_concepts": ["feature scaling", "euclidean distance", "dominance of larger scale features", "normalization"],
            "difficulty": "easy",
            "topic_aspect": "distance_computation",
        },
        {
            "question": "How does the KNN classifier resolve class assignment when querying an unseen test sample?",
            "expected_concepts": ["distance metric", "majority voting", "k nearest neighbors", "class frequency"],
            "difficulty": "easy",
            "topic_aspect": "classification_mechanism",
        },
        {
            "question": "Why is KNN classified as an instance-based lazy learning algorithm rather than an eager parametric model?",
            "expected_concepts": ["lazy learner", "no explicit training phase", "stores training instances", "computation during test inference"],
            "difficulty": "medium",
            "topic_aspect": "algorithm_type",
        },
        {
            "question": "In what scenario would Manhattan distance or Cosine similarity be preferred over standard Euclidean distance in KNN?",
            "expected_concepts": ["manhattan distance for grid-like or high dimensional data", "cosine similarity for text/angles", "sparse features"],
            "difficulty": "medium",
            "topic_aspect": "distance_metrics",
        },
        {
            "question": "How does the 'curse of dimensionality' degrade the classification accuracy and distance calculations in KNN?",
            "expected_concepts": ["curse of dimensionality", "equidistant points in high dimensions", "sparsity of data", "distance metric breakdown"],
            "difficulty": "medium",
            "topic_aspect": "dimensionality_effects",
        },
    ],
    "Decision Tree": [
        {
            "question": "In Decision Tree algorithms like ID3, how is Information Gain calculated from Entropy to choose the best split attribute?",
            "expected_concepts": ["information gain", "entropy reduction", "splitting attribute", "purity increase"],
            "difficulty": "medium",
            "topic_aspect": "splitting_criterion",
        },
        {
            "question": "What is the mathematical meaning of Entropy in this experiment, and when does a node reach an Entropy of zero?",
            "expected_concepts": ["entropy measures disorder/impurity", "zero entropy means pure node", "all samples belong to same class"],
            "difficulty": "medium",
            "topic_aspect": "entropy_theory",
        },
        {
            "question": "How does Gini Impurity used in CART differ conceptually from Entropy used in the ID3 algorithm?",
            "expected_concepts": ["gini impurity", "computational efficiency", "probability of misclassification", "information theory vs probability"],
            "difficulty": "medium",
            "topic_aspect": "impurity_metrics",
        },
        {
            "question": "Why do fully grown unconstrained decision trees tend to overfit, and how does maximum depth pruning mitigate this?",
            "expected_concepts": ["overfitting", "tree pruning", "max depth constraint", "leaf size constraint", "generalization"],
            "difficulty": "easy",
            "topic_aspect": "pruning_regularization",
        },
        {
            "question": "What is the functional difference between an internal decision node and a terminal leaf node in a Decision Tree?",
            "expected_concepts": ["decision node tests attribute", "leaf node assigns class label or regression value", "tree structure"],
            "difficulty": "easy",
            "topic_aspect": "tree_architecture",
        },
        {
            "question": "Why is a Decision Tree particularly advantageous for domains like healthcare or credit scoring where decision interpretability is required?",
            "expected_concepts": ["interpretability", "transparent if-then rules", "human verifiable decisions", "white box model"],
            "difficulty": "easy",
            "topic_aspect": "practical_application",
        },
    ],
    "Naive Bayes Classification": [
        {
            "question": "What is the core 'conditional independence' assumption in Naive Bayes, and why is it considered naive?",
            "expected_concepts": ["conditional independence", "features independent given class", "simplifies joint probability", "naive assumption"],
            "difficulty": "easy",
            "topic_aspect": "core_assumption",
        },
        {
            "question": "How does Bayes' theorem compute the posterior probability P(Class|Features) using prior probability and likelihood?",
            "expected_concepts": ["bayes theorem", "prior probability", "likelihood", "posterior probability", "evidence denominator"],
            "difficulty": "medium",
            "topic_aspect": "probability_formulation",
        },
        {
            "question": "Why is Laplace smoothing (adding alpha=1) necessary during probability computation in Naive Bayes?",
            "expected_concepts": ["laplace smoothing", "zero probability problem", "unseen feature value in training", "additive smoothing"],
            "difficulty": "medium",
            "topic_aspect": "smoothing_technique",
        },
        {
            "question": "How does Gaussian Naive Bayes model continuous numerical features compared to Multinomial Naive Bayes for discrete counts?",
            "expected_concepts": ["gaussian naive bayes", "normal distribution curve", "mean and variance", "multinomial for word counts"],
            "difficulty": "medium",
            "topic_aspect": "distribution_models",
        },
        {
            "question": "Why is Naive Bayes particularly effective for text classification applications like spam filtering and sentiment analysis?",
            "expected_concepts": ["text classification", "high dimensional word features", "fast linear computation", "bag of words representation"],
            "difficulty": "easy",
            "topic_aspect": "text_application",
        },
        {
            "question": "In the Play Tennis dataset, how does Naive Bayes handle categorical weather features such as Outlook, Temperature, and Humidity?",
            "expected_concepts": ["frequency counting", "conditional probability tables", "categorical feature multiplication", "prior and likelihood"],
            "difficulty": "medium",
            "topic_aspect": "dataset_features",
        },
    ],
    "K-Means Clustering": [
        {
            "question": "In K-Means clustering, how are cluster centroids initialized, and how are they recalculated in subsequent iterations?",
            "expected_concepts": ["centroid initialization", "mean of assigned data points", "iterative assignment", "convergence"],
            "difficulty": "easy",
            "topic_aspect": "centroid_algorithm",
        },
        {
            "question": "How does the Elbow Method use Within-Cluster Sum of Squares (WCSS / Inertia) to determine the optimal number of clusters K?",
            "expected_concepts": ["elbow method", "wcss inertia", "point of diminishing returns", "optimal k selection"],
            "difficulty": "medium",
            "topic_aspect": "optimal_k_selection",
        },
        {
            "question": "Why is standard K-Means sensitive to initial centroid placement, and how does K-Means++ solve this problem?",
            "expected_concepts": ["initial centroid sensitivity", "k-means++", "probabilistic distance-based spacing", "avoiding bad local minima"],
            "difficulty": "medium",
            "topic_aspect": "initialization_method",
        },
        {
            "question": "What are the primary limitations of K-Means when dealing with clusters of unequal variance, non-spherical shapes, or outliers?",
            "expected_concepts": ["spherical cluster assumption", "equal variance assumption", "outlier sensitivity shifts centroid", "density limitations"],
            "difficulty": "medium",
            "topic_aspect": "cluster_limitations",
        },
        {
            "question": "What is the fundamental difference between unsupervised clustering like K-Means and supervised classification?",
            "expected_concepts": ["unsupervised learning", "no ground truth target labels", "discovering latent groupings", "similarity metric"],
            "difficulty": "easy",
            "topic_aspect": "learning_paradigm",
        },
        {
            "question": "How do you evaluate clustering quality when ground truth labels are absent, using metrics like the Silhouette Score?",
            "expected_concepts": ["silhouette score", "intra cluster cohesion", "inter cluster separation", "range between -1 and 1"],
            "difficulty": "medium",
            "topic_aspect": "clustering_evaluation",
        },
    ],
    "K-Medoids Clustering": [
        {
            "question": "What is the primary difference between a centroid in K-Means and a medoid in the K-Medoids (PAM) algorithm?",
            "expected_concepts": ["medoid is an actual data point from dataset", "centroid is a computed mean", "pam algorithm", "interpretability"],
            "difficulty": "easy",
            "topic_aspect": "medoid_concept",
        },
        {
            "question": "Why is K-Medoids significantly more robust to noise and extreme outliers compared to standard K-Means?",
            "expected_concepts": ["robust to outliers", "actual central exemplar point", "minimizes absolute dissimilarity", "manhattan or custom distance"],
            "difficulty": "medium",
            "topic_aspect": "outlier_robustness",
        },
        {
            "question": "How does the Partitioning Around Medoids (PAM) algorithm decide whether swapping a medoid with a non-medoid point is beneficial?",
            "expected_concepts": ["swapping cost calculation", "total dissimilarity reduction", "iterative optimization", "objective function"],
            "difficulty": "medium",
            "topic_aspect": "pam_swapping_cost",
        },
        {
            "question": "In what practical situations or data formats is K-Medoids preferred over K-Means?",
            "expected_concepts": ["non-euclidean pairwise distance matrices", "categorical data", "datasets with severe outliers", "bioinformatics"],
            "difficulty": "medium",
            "topic_aspect": "practical_application",
        },
        {
            "question": "What is the computational complexity difference between the PAM algorithm in K-Medoids and standard K-Means for large datasets?",
            "expected_concepts": ["pam is computationally expensive O(k(n-k)^2)", "k-means is linear in samples O(nkt)", "scalability challenge"],
            "difficulty": "medium",
            "topic_aspect": "computational_complexity",
        },
        {
            "question": "How does K-Medoids compute cluster membership using total pairwise dissimilarity instead of squared Euclidean distance?",
            "expected_concepts": ["dissimilarity matrix", "absolute error minimization", "manhattan distance", "representative objects"],
            "difficulty": "medium",
            "topic_aspect": "dissimilarity_metric",
        },
    ],
    "DBSCAN Clustering": [
        {
            "question": "In the DBSCAN algorithm, what do the parameters 'eps' (epsilon) and 'min_samples' control in density estimation?",
            "expected_concepts": ["epsilon neighborhood radius", "min_samples threshold", "density requirement", "core point criterion"],
            "difficulty": "easy",
            "topic_aspect": "density_parameters",
        },
        {
            "question": "How does DBSCAN classify data points into core points, border points, and noise (outliers)?",
            "expected_concepts": ["core point has at least min_samples in eps", "border point within eps of core", "noise point has insufficient neighbors"],
            "difficulty": "medium",
            "topic_aspect": "point_classification",
        },
        {
            "question": "What major advantage does DBSCAN provide over K-Means when clustering irregularly shaped non-convex data?",
            "expected_concepts": ["arbitrary non-convex cluster shapes", "no predefined cluster count k", "automatic noise detection", "density connectivity"],
            "difficulty": "medium",
            "topic_aspect": "cluster_geometry",
        },
        {
            "question": "What does 'density-reachability' and 'density-connectivity' mean during cluster formation in DBSCAN?",
            "expected_concepts": ["density reachability", "chain of core points", "density connected", "cluster expansion"],
            "difficulty": "medium",
            "topic_aspect": "density_connectivity",
        },
        {
            "question": "How does varying the epsilon (eps) parameter too low or too high affect the number of clusters and noise points in DBSCAN?",
            "expected_concepts": ["small eps classifies most points as noise", "large eps merges distinct clusters into one", "parameter sensitivity"],
            "difficulty": "medium",
            "topic_aspect": "parameter_tuning",
        },
        {
            "question": "Why does DBSCAN struggle to identify clusters with significantly varying densities across the dataset?",
            "expected_concepts": ["varying cluster densities", "single global eps parameter limitation", "sparse vs dense clusters"],
            "difficulty": "medium",
            "topic_aspect": "density_variation",
        },
    ],
    "SVM Classification": [
        {
            "question": "In Support Vector Machines, what is the maximum margin hyperplane, and why is maximizing margin desirable?",
            "expected_concepts": ["maximum margin hyperplane", "decision boundary", "margin width", "generalization improvement"],
            "difficulty": "easy",
            "topic_aspect": "hyperplane_margin",
        },
        {
            "question": "What are support vectors in SVM, and why do only these specific data points influence the decision boundary?",
            "expected_concepts": ["support vectors", "closest training instances to hyperplane", "lie on margin boundaries", "determine hyperplane parameters"],
            "difficulty": "medium",
            "topic_aspect": "support_vectors",
        },
        {
            "question": "What is the 'kernel trick' in SVM, and how does it enable classification of non-linearly separable data?",
            "expected_concepts": ["kernel trick", "implicit mapping to higher dimension", "rbf radial basis function", "polynomial kernel", "inner product computation"],
            "difficulty": "medium",
            "topic_aspect": "kernel_trick",
        },
        {
            "question": "How does the regularization hyperparameter 'C' control the tradeoff between margin width and training classification errors?",
            "expected_concepts": ["regularization parameter c", "large c penalizes errors causing narrow margin", "small c allows soft margin", "bias variance tradeoff"],
            "difficulty": "medium",
            "topic_aspect": "regularization_parameter",
        },
        {
            "question": "What is the difference between Hard Margin SVM and Soft Margin SVM when handling non-separable noisy data?",
            "expected_concepts": ["hard margin strictly forbids errors", "soft margin uses slack variables", "handling overlapping classes"],
            "difficulty": "medium",
            "topic_aspect": "soft_margin",
        },
        {
            "question": "When using an RBF kernel in SVM, how does tuning the gamma parameter affect the influence of individual support vectors?",
            "expected_concepts": ["rbf kernel gamma parameter", "large gamma leads to high variance and tight radius", "small gamma creates broad smooth boundary"],
            "difficulty": "medium",
            "topic_aspect": "gamma_parameter",
        },
    ],
    "Principal Component Analysis (PCA)": [
        {
            "question": "What is the primary mathematical objective of Principal Component Analysis (PCA) in unsupervised dimensionality reduction?",
            "expected_concepts": ["dimensionality reduction", "variance maximization", "orthogonal projections", "removing redundant correlations"],
            "difficulty": "easy",
            "topic_aspect": "dimensionality_reduction",
        },
        {
            "question": "In PCA, what do the eigenvectors and eigenvalues of the data covariance matrix represent geometrically?",
            "expected_concepts": ["eigenvectors are directions of maximum variance", "eigenvalues represent magnitude of variance explained", "orthogonal axes"],
            "difficulty": "medium",
            "topic_aspect": "eigenvalues_eigenvectors",
        },
        {
            "question": "How do you interpret the Cumulative Explained Variance Ratio curve (scree plot) to select the number of principal components to keep?",
            "expected_concepts": ["cumulative explained variance", "scree plot elbow", "retaining 85-95% variance", "dimensional compression threshold"],
            "difficulty": "medium",
            "topic_aspect": "explained_variance",
        },
        {
            "question": "Why is mean-centering and standardizing features necessary before computing the covariance matrix in PCA?",
            "expected_concepts": ["mean centering", "feature scaling standardization", "preventing high scale features from dominating variance"],
            "difficulty": "medium",
            "topic_aspect": "data_centering",
        },
        {
            "question": "Why are principal components strictly orthogonal (perpendicular) to each other?",
            "expected_concepts": ["orthogonality", "zero covariance between components", "uncorrelated feature representation", "eigenvector properties"],
            "difficulty": "medium",
            "topic_aspect": "orthogonality",
        },
        {
            "question": "What is the tradeoff between reducing feature dimensions with PCA and preserving the original information for downstream models?",
            "expected_concepts": ["dimensionality reduction vs information loss", "variance retention", "eliminating noise", "interpretability tradeoff"],
            "difficulty": "medium",
            "topic_aspect": "reconstruction_tradeoff",
        },
    ],
    "Understanding the Perceptron": [
        {
            "question": "What mathematical operation does a single Perceptron perform on its input features (x), weights (w), and bias (b)?",
            "expected_concepts": ["linear combination", "dot product of weights and inputs", "sum w_i * x_i + bias", "weighted sum"],
            "difficulty": "easy",
            "topic_aspect": "mathematical_formula",
        },
        {
            "question": "How does the step activation function convert the Perceptron's weighted sum into a binary classification output?",
            "expected_concepts": ["step function", "threshold comparison", "output 1 if sum >= threshold else 0", "binary decision"],
            "difficulty": "easy",
            "topic_aspect": "activation_function",
        },
        {
            "question": "What is the role of the learning rate (eta) in updating weights during Perceptron training?",
            "expected_concepts": ["learning rate", "step size of weight update", "gradient scaling", "convergence stability"],
            "difficulty": "medium",
            "topic_aspect": "learning_rate",
        },
        {
            "question": "Why can a single Perceptron classify linearly separable problems like AND/OR logic gates but fail on XOR?",
            "expected_concepts": ["linear separability", "single linear decision boundary hyperplane", "xor is non-linear problem"],
            "difficulty": "medium",
            "topic_aspect": "linear_separability",
        },
        {
            "question": "In the Iris dataset classification with Perceptron, how are multi-dimensional flower features mapped to a decision boundary?",
            "expected_concepts": ["feature dimensions", "hyperplane separator", "sepal/petal measurements", "binary target split"],
            "difficulty": "medium",
            "topic_aspect": "dataset_application",
        },
        {
            "question": "What condition must be met for the Perceptron Learning Algorithm (PLA) to guarantee convergence?",
            "expected_concepts": ["linearly separable dataset", "perceptron convergence theorem", "finite number of weight updates"],
            "difficulty": "medium",
            "topic_aspect": "convergence_theory",
        },
    ],
    "Single Layer Perceptron": [
        {
            "question": "How does a Single Layer Perceptron update its weight vector when a training instance is misclassified?",
            "expected_concepts": ["weight update rule", "w_new = w_old + eta * (y_true - y_pred) * x", "error driven adjustment"],
            "difficulty": "medium",
            "topic_aspect": "weight_update",
        },
        {
            "question": "What is the purpose of the bias term in a Single Layer Perceptron, and what happens if bias is omitted?",
            "expected_concepts": ["bias allows decision boundary to shift away from origin", "intercept offset", "flexibility of hyperplane"],
            "difficulty": "easy",
            "topic_aspect": "bias_term",
        },
        {
            "question": "How does a Single Layer Perceptron differ functionally from a Multi-Layer Perceptron (MLP)?",
            "expected_concepts": ["no hidden layers", "restricted to linear decision boundaries", "mlp uses hidden layers and non-linear activations"],
            "difficulty": "easy",
            "topic_aspect": "architecture_difference",
        },
        {
            "question": "How does the choice of initial weights affect the training iterations of a Single Layer Perceptron?",
            "expected_concepts": ["weight initialization", "starting point on error surface", "number of convergence epochs"],
            "difficulty": "medium",
            "topic_aspect": "initialization",
        },
        {
            "question": "What metric or stopping criterion is used to terminate training in a Single Layer Perceptron experiment?",
            "expected_concepts": ["zero classification error on training set", "maximum epoch limit", "loss convergence"],
            "difficulty": "easy",
            "topic_aspect": "stopping_criterion",
        },
        {
            "question": "Why is a threshold or step activation function not differentiable at zero, and how does that affect backpropagation?",
            "expected_concepts": ["non-differentiable step function", "zero derivative almost everywhere", "need smooth activations for gradient descent"],
            "difficulty": "medium",
            "topic_aspect": "differentiability",
        },
    ],
    "Activation Functions": [
        {
            "question": "What is the primary conceptual reason for using non-linear activation functions in deep neural networks?",
            "expected_concepts": ["introducing non-linearity", "enabling learning of complex non-linear mappings", "preventing collapse into linear regression"],
            "difficulty": "easy",
            "topic_aspect": "non_linearity",
        },
        {
            "question": "How does the vanishing gradient problem occur with the Sigmoid activation function during backpropagation?",
            "expected_concepts": ["sigmoid derivative max is 0.25", "saturates at tails (0 and 1)", "gradients vanish through deep layers"],
            "difficulty": "medium",
            "topic_aspect": "vanishing_gradient",
        },
        {
            "question": "Why is ReLU (Rectified Linear Unit) computationally efficient, and what is the 'dying ReLU' issue?",
            "expected_concepts": ["relu is max(0, x)", "constant gradient of 1 for positive inputs", "dying relu when weights cause permanent negative inputs"],
            "difficulty": "medium",
            "topic_aspect": "relu_mechanics",
        },
        {
            "question": "How does Leaky ReLU or ELU (Exponential Linear Unit) solve the zero-gradient issue for negative inputs in ReLU?",
            "expected_concepts": ["small non-zero slope alpha for negative values", "prevents dead neurons", "elu smooth curve"],
            "difficulty": "medium",
            "topic_aspect": "leaky_relu_elu",
        },
        {
            "question": "Why is the Tanh activation function generally preferred over standard Sigmoid in hidden layers?",
            "expected_concepts": ["zero-centered output range (-1 to 1)", "stronger gradients", "faster convergence than standard sigmoid"],
            "difficulty": "medium",
            "topic_aspect": "tanh_zero_centered",
        },
        {
            "question": "In what specific layer and for what classification type is the Softmax activation function typically used?",
            "expected_concepts": ["output layer", "multi-class classification", "converts logits into normalized probability distribution summing to 1"],
            "difficulty": "easy",
            "topic_aspect": "softmax_application",
        },
    ],
    "Optimizers in Neural Network": [
        {
            "question": "How does Stochastic Gradient Descent (SGD) differ conceptually from standard Batch Gradient Descent in terms of update frequency?",
            "expected_concepts": ["sgd updates weights per individual sample or mini-batch", "batch gd uses entire dataset per epoch", "faster iterations vs noise"],
            "difficulty": "easy",
            "topic_aspect": "sgd_vs_batch",
        },
        {
            "question": "What role does the 'Momentum' hyperparameter play in accelerating gradient descent through ravines?",
            "expected_concepts": ["momentum adds fraction of past update vector", "dampens oscillations", "accelerates through flat surfaces"],
            "difficulty": "medium",
            "topic_aspect": "momentum",
        },
        {
            "question": "How does the Adam optimizer combine the advantages of Momentum and RMSprop?",
            "expected_concepts": ["adaptive learning rates", "first moment (mean of gradients)", "second moment (uncentered variance)", "bias correction"],
            "difficulty": "medium",
            "topic_aspect": "adam_optimizer",
        },
        {
            "question": "Why is choosing a learning rate that is too high or too low detrimental during neural network training?",
            "expected_concepts": ["too high causes divergence/oscillation", "too low causes very slow convergence or local minima entrapment"],
            "difficulty": "easy",
            "topic_aspect": "learning_rate_tuning",
        },
        {
            "question": "How does RMSprop prevent the learning rate from diminishing to zero compared to standard AdaGrad?",
            "expected_concepts": ["exponentially decaying average of squared gradients", "prevents continuous accumulation of historical gradients"],
            "difficulty": "medium",
            "topic_aspect": "rmsprop",
        },
        {
            "question": "How do you evaluate whether an optimizer has converged successfully by inspecting training and validation loss curves?",
            "expected_concepts": ["loss curve flattening", "minimal validation loss", "checking for divergence or overfitting gap"],
            "difficulty": "medium",
            "topic_aspect": "convergence_evaluation",
        },
    ],
    "VGG16 Transfer Learning": [
        {
            "question": "What is the concept of Transfer Learning, and why is a pre-trained VGG16 model effective for image classification?",
            "expected_concepts": ["transfer learning leverages features learned on ImageNet", "pre-trained weights", "reduces training time and data requirements"],
            "difficulty": "easy",
            "topic_aspect": "transfer_learning_concept",
        },
        {
            "question": "Why are the convolutional base layers frozen (trainable=False) when fine-tuning VGG16 on a new custom face dataset?",
            "expected_concepts": ["freezing prevents destroying pre-trained feature extractors", "retains generic edges/textures", "trains only new classifier head"],
            "difficulty": "medium",
            "topic_aspect": "layer_freezing",
        },
        {
            "question": "How do data augmentation techniques such as rotation and brightness adjustments improve classification accuracy and reduce overfitting?",
            "expected_concepts": ["data augmentation increases dataset diversity", "synthesizes variations", "enhances model generalization"],
            "difficulty": "easy",
            "topic_aspect": "data_augmentation",
        },
        {
            "question": "What is the role of the Flatten and Dense layers added on top of the VGG16 feature extractor?",
            "expected_concepts": ["flatten converts 2D feature maps to 1D vector", "dense layers perform high-level classification", "softmax output for classes"],
            "difficulty": "medium",
            "topic_aspect": "classification_head",
        },
        {
            "question": "Why are input images resized to 224x224 pixels before passing them into the VGG16 network?",
            "expected_concepts": ["vgg16 architecture fixed input resolution 224x224x3", "standard tensor shape for convolutional blocks"],
            "difficulty": "easy",
            "topic_aspect": "input_preprocessing",
        },
        {
            "question": "How does a custom callback like EarlyStopping or AccuracyThreshold help in training the transfer model efficiently?",
            "expected_concepts": ["callbacks monitor metrics during training", "stops training when target accuracy is reached", "prevents overfitting"],
            "difficulty": "medium",
            "topic_aspect": "callbacks_monitoring",
        },
    ],
    "ResNet-50 Transfer Learning": [
        {
            "question": "What is the fundamental architectural innovation of 'Residual Connections' (skip connections) in ResNet-50?",
            "expected_concepts": ["skip connections add input x to F(x)", "identity mapping", "mitigates vanishing gradients in very deep networks"],
            "difficulty": "medium",
            "topic_aspect": "residual_skip_connections",
        },
        {
            "question": "How do bottleneck blocks in ResNet-50 (1x1, 3x3, 1x1 convolutions) reduce computational complexity?",
            "expected_concepts": ["1x1 convolution reduces channel depth", "3x3 spatial convolution", "1x1 expands channels", "computational efficiency"],
            "difficulty": "medium",
            "topic_aspect": "bottleneck_architecture",
        },
        {
            "question": "Why does ResNet-50 perform significantly better on complex image datasets compared to shallower standard CNNs?",
            "expected_concepts": ["depth enables multi-scale hierarchical feature extraction", "50 layers deep without gradient degradation", "higher representational capacity"],
            "difficulty": "easy",
            "topic_aspect": "deep_representations",
        },
        {
            "question": "What is the difference between feature extraction (freezing all base layers) and fine-tuning (unfreezing top residual blocks) in ResNet-50?",
            "expected_concepts": ["feature extraction trains only top dense layers", "fine-tuning updates high-level residual weights with low learning rate"],
            "difficulty": "medium",
            "topic_aspect": "fine_tuning_strategy",
        },
        {
            "question": "What role does Global Average Pooling (GAP) play before the final classification layer in ResNet-50?",
            "expected_concepts": ["global average pooling replaces heavy flatten layer", "averages spatial feature maps", "drastically reduces parameter count"],
            "difficulty": "medium",
            "topic_aspect": "global_average_pooling",
        },
        {
            "question": "How do learning rate schedulers and weight decay regularizers prevent overfitting when training ResNet-50?",
            "expected_concepts": ["learning rate decay improves fine convergence", "weight decay penalizes large weights", "stabilizes gradient steps"],
            "difficulty": "medium",
            "topic_aspect": "regularization_optimization",
        },
    ],
    "Custom CNN Models": [
        {
            "question": "In a Convolutional Neural Network, what operation does a 2D convolution filter (kernel) perform on an input image?",
            "expected_concepts": ["element-wise multiplication and summation", "sliding dot product across spatial grid", "extracting feature maps"],
            "difficulty": "easy",
            "topic_aspect": "convolution_operation",
        },
        {
            "question": "What is the primary function of MaxPooling layers in a CNN, and how does pooling achieve translation invariance?",
            "expected_concepts": ["downsampling spatial dimensions", "reducing parameter count and computation", "extracting dominant features", "spatial invariance"],
            "difficulty": "easy",
            "topic_aspect": "pooling_layers",
        },
        {
            "question": "What do 'padding' (e.g. valid vs same) and 'stride' control in the spatial size of output feature maps?",
            "expected_concepts": ["padding preserves boundary information", "same padding maintains input dimension", "stride controls filter step size"],
            "difficulty": "medium",
            "topic_aspect": "padding_and_stride",
        },
        {
            "question": "How does the Dropout regularization layer prevent co-adaptation of neurons in CNN classification layers?",
            "expected_concepts": ["randomly deactivates fraction of neurons during training", "forces redundant representations", "reduces overfitting"],
            "difficulty": "medium",
            "topic_aspect": "dropout_regularization",
        },
        {
            "question": "Why do earlier convolutional layers capture low-level features (edges, textures) while deeper layers capture high-level semantic shapes?",
            "expected_concepts": ["hierarchical feature learning", "receptive field increases with depth", "composition of lower-level patterns"],
            "difficulty": "medium",
            "topic_aspect": "hierarchical_features",
        },
        {
            "question": "How do you interpret the training vs validation accuracy curve to detect overfitting in your custom CNN model?",
            "expected_concepts": ["widening gap between high training accuracy and low validation accuracy", "validation loss increasing while train loss drops"],
            "difficulty": "easy",
            "topic_aspect": "training_curves",
        },
    ],
    "Sequential Prediction (LSTM & GRU)": [
        {
            "question": "Why do standard Recurrent Neural Networks (RNNs) struggle with long-term temporal dependencies, and how does LSTM solve this?",
            "expected_concepts": ["exploding and vanishing gradient problem", "lstm introduces cell state highway and gating mechanisms", "preserves long-term memory"],
            "difficulty": "medium",
            "topic_aspect": "lstm_memory_cell",
        },
        {
            "question": "What are the three distinct gates in an LSTM cell (Forget, Input, Output), and what is the role of the Forget gate?",
            "expected_concepts": ["forget gate decides what information to discard from cell state", "input gate updates cell state", "output gate filters hidden state"],
            "difficulty": "easy",
            "topic_aspect": "lstm_gates",
        },
        {
            "question": "How does a Gated Recurrent Unit (GRU) simplify the LSTM architecture while retaining sequential modeling capabilities?",
            "expected_concepts": ["combines hidden state and cell state", "uses two gates: update gate and reset gate", "fewer parameters and faster training"],
            "difficulty": "medium",
            "topic_aspect": "gru_architecture",
        },
        {
            "question": "In sequential character prediction, how are discrete characters converted into numeric representations before passing to the LSTM?",
            "expected_concepts": ["one-hot encoding or token embedding", "mapping characters to integer vocabulary indices", "sequential tensor format"],
            "difficulty": "easy",
            "topic_aspect": "character_encoding",
        },
        {
            "question": "What role does the 'temperature' parameter play when sampling predictions from the Softmax probabilities of a character model?",
            "expected_concepts": ["temperature scales logit distribution", "low temperature produces deterministic/conservative text", "high temperature increases diversity/randomness"],
            "difficulty": "medium",
            "topic_aspect": "temperature_sampling",
        },
        {
            "question": "How does Teacher Forcing improve training convergence during sequence-to-sequence or sequential character generation?",
            "expected_concepts": ["passing ground truth previous token as input during training instead of model prediction", "stabilizes gradient updates"],
            "difficulty": "medium",
            "topic_aspect": "teacher_forcing",
        },
    ],
    "StyleGANs": [
        {
            "question": "In a Generative Adversarial Network (GAN), what are the competing objectives of the Generator and the Discriminator?",
            "expected_concepts": ["generator creates realistic synthetic samples to fool discriminator", "discriminator classifies real vs fake samples", "minimax game"],
            "difficulty": "easy",
            "topic_aspect": "gan_minimax",
        },
        {
            "question": "What is the key architectural difference between standard GANs and StyleGAN's mapping network and style-based generator?",
            "expected_concepts": ["mapping network maps latent z to intermediate space w", "disentangles feature representations", "adaptive instance normalization AdaIN"],
            "difficulty": "medium",
            "topic_aspect": "stylegan_architecture",
        },
        {
            "question": "How does Adaptive Instance Normalization (AdaIN) transfer style parameters to feature maps at each resolution level in StyleGAN?",
            "expected_concepts": ["adain normalizes feature map channels to zero mean and unit variance", "scales and shifts with learned style vectors (y_s, y_b)"],
            "difficulty": "medium",
            "topic_aspect": "adain_mechanism",
        },
        {
            "question": "What is 'disentangled latent space' in StyleGAN, and why is it beneficial for controlling visual attributes (pose, hair, expression)?",
            "expected_concepts": ["independent control of individual semantic attributes", "intermediate w space avoids correlation", "linear attribute interpolation"],
            "difficulty": "medium",
            "topic_aspect": "latent_space_disentanglement",
        },
        {
            "question": "How does Progressive Growing stabilize GAN training from low resolutions (4x4) to high resolutions (1024x1024)?",
            "expected_concepts": ["starts training on low-resolution images", "incrementally adds higher-resolution layers", "stabilizes adversarial dynamics"],
            "difficulty": "medium",
            "topic_aspect": "progressive_growing",
        },
        {
            "question": "What is 'Mode Collapse' in GAN training, and what symptom indicates that the generator has collapsed?",
            "expected_concepts": ["mode collapse occurs when generator produces limited identical samples", "loss of sample diversity", "discriminator fails to penalize repetition"],
            "difficulty": "medium",
            "topic_aspect": "mode_collapse",
        },
    ],
    "Audio Classification (LSTM & GRU)": [
        {
            "question": "In audio data classification, what are Mel-Frequency Cepstral Coefficients (MFCCs) and why are they used to represent audio signals?",
            "expected_concepts": ["mfccs capture spectral envelope of audio", "mel scale matches human auditory perception", "compact feature representation of speech/sound"],
            "difficulty": "easy",
            "topic_aspect": "mfcc_features",
        },
        {
            "question": "How does Short-Time Fourier Transform (STFT) convert a 1D raw audio waveform into a 2D Time-Frequency Spectrogram?",
            "expected_concepts": ["divides audio into overlapping short windowed frames", "computes fourier transform per window", "generates time-frequency magnitude matrix"],
            "difficulty": "medium",
            "topic_aspect": "stft_spectrogram",
        },
        {
            "question": "Why are sequential architectures like LSTM or GRU well-suited for processing extracted audio feature sequences?",
            "expected_concepts": ["audio is inherently sequential time-series data", "lstm models temporal transitions between consecutive frames", "captures acoustic dynamics"],
            "difficulty": "easy",
            "topic_aspect": "temporal_modeling",
        },
        {
            "question": "What role does audio sampling rate (e.g. 16kHz or 44.1kHz) play in preserving frequency information according to the Nyquist theorem?",
            "expected_concepts": ["nyquist rate requires sampling at least twice the highest frequency", "preserves audible frequencies without aliasing"],
            "difficulty": "medium",
            "topic_aspect": "sampling_rate",
        },
        {
            "question": "How does applying data augmentation like time stretching, pitch shifting, or background noise addition improve audio model robustness?",
            "expected_concepts": ["audio augmentation introduces realistic acoustic variations", "prevents overfitting to recording environment", "enhances generalization"],
            "difficulty": "medium",
            "topic_aspect": "audio_augmentation",
        },
        {
            "question": "What evaluation metrics are most appropriate for evaluating multi-class audio classification performance on imbalanced audio datasets?",
            "expected_concepts": ["f1-score", "precision", "recall", "confusion matrix", "unweighted average recall"],
            "difficulty": "medium",
            "topic_aspect": "audio_metrics",
        },
    ],
    "Human Detection in Thermal Images": [
        {
            "question": "How do thermal infrared images differ from standard RGB images in terms of pixel intensity representing heat radiation?",
            "expected_concepts": ["thermal intensity represents infrared surface temperature", "invariant to ambient lighting and darkness", "single-channel heat radiation contrast"],
            "difficulty": "easy",
            "topic_aspect": "thermal_imaging_physics",
        },
        {
            "question": "Why is human body temperature contrast particularly advantageous for object detection in low-light or foggy conditions?",
            "expected_concepts": ["human body emits distinct body temperature signature (around 37C)", "high contrast against cooler background", "unaffected by shadows"],
            "difficulty": "easy",
            "topic_aspect": "temperature_contrast",
        },
        {
            "question": "How do deep learning object detection networks predict bounding box coordinates and confidence scores for detected persons?",
            "expected_concepts": ["bounding box regression (x, y, width, height)", "objectness confidence score", "class probability distribution"],
            "difficulty": "medium",
            "topic_aspect": "bounding_box_regression",
        },
        {
            "question": "What is Non-Maximum Suppression (NMS), and how does it eliminate redundant overlapping bounding boxes around the same person?",
            "expected_concepts": ["intersection over union (IoU) thresholding", "retains box with highest confidence", "suppresses redundant overlapping boxes"],
            "difficulty": "medium",
            "topic_aspect": "non_maximum_suppression",
        },
        {
            "question": "What environmental challenges (e.g., warm background objects, heated machinery) cause false positive detections in thermal imaging?",
            "expected_concepts": ["thermal artifacts matching human body heat", "heated radiators/engines", "solar-heated walls creating thermal clutter"],
            "difficulty": "medium",
            "topic_aspect": "thermal_challenges",
        },
        {
            "question": "How is Mean Average Precision (mAP at IoU=0.50) calculated to measure the detection accuracy of human detection models?",
            "expected_concepts": ["area under precision-recall curve", "iou threshold 0.50 for true positives", "mean across classes"],
            "difficulty": "medium",
            "topic_aspect": "detection_evaluation",
        },
    ],
}


def _detect_topic_from_text(content: str, title: str = "") -> str:
    combined = (title + " " + content).lower()
    
    # Priority matching for specific algorithms and laboratory titles
    if re.search(r"thermal.*(?:image|human|detect)|human.*detect.*thermal", combined):
        return "Human Detection in Thermal Images"
    if re.search(r"audio.*(?:data|classif)|speech.*classif|mfcc", combined):
        return "Audio Classification (LSTM & GRU)"
    if re.search(r"stylegan|generative.*adversarial|latent.*space.*w", combined):
        return "StyleGANs"
    if re.search(r"sequential.*(?:character|predict)|lstm.*gru|gru.*lstm|character.*predict", combined):
        return "Sequential Prediction (LSTM & GRU)"
    if re.search(r"custom.*cnn|cnn.*model|convolutional.*neural.*network", combined):
        return "Custom CNN Models"
    if re.search(r"resnet|res.*net", combined):
        return "ResNet-50 Transfer Learning"
    if re.search(r"vgg16|vgg|face.*recognit.*transfer|transfer.*learning.*vgg", combined):
        return "VGG16 Transfer Learning"
    if re.search(r"optimizer|adam.*optimizer|sgd.*momentum|rmsprop", combined):
        return "Optimizers in Neural Network"
    if re.search(r"activation.*function|sigmoid.*relu|relu.*tanh", combined):
        return "Activation Functions"
    if re.search(r"single.*layer.*perceptron", combined):
        return "Single Layer Perceptron"
    if re.search(r"perceptron|understanding.*perceptron", combined):
        return "Understanding the Perceptron"
    if re.search(r"multiple.*linear.*regression|multi.*linear.*regress", combined):
        return "Multiple Linear Regression"
    if re.search(r"simple.*linear.*regression|linear.*regression", combined):
        return "Simple Linear Regression"
    if re.search(r"k.*medoid|pam.*cluster", combined):
        return "K-Medoids Clustering"
    if re.search(r"k.*means|kmeans", combined):
        return "K-Means Clustering"
    if re.search(r"k.*nearest|knn|nearest.*neighbor", combined):
        return "k-Nearest Neighbors (KNN)"
    if re.search(r"decision.*tree|id3|cart|entropy.*gain", combined):
        return "Decision Tree"
    if re.search(r"naive.*bayes|bayesian|bayes.*classif", combined):
        return "Naive Bayes Classification"
    if re.search(r"dbscan|density.*based.*cluster", combined):
        return "DBSCAN Clustering"
    if re.search(r"svm|support.*vector", combined):
        return "SVM Classification"
    if re.search(r"principal.*component|pca|eigenvalue", combined):
        return "Principal Component Analysis (PCA)"
    
    for topic in _TOPIC_QUESTION_BANKS.keys():
        clean_topic = format_experiment_title(topic, 0)
        if clean_topic.lower() in combined:
            return topic
            
    if title:
        return format_experiment_title(title, 0)
    return ""


def extract_experiment_topics_and_concepts(experiment_content: str, title: str = "") -> dict:
    """
    Extract specific technical topics, concepts, parameters, formulas, and metrics
    actually present in the selected experiment content.
    """
    material = clean(experiment_content)
    detected_topic = _detect_topic_from_text(material, title) or "Laboratory Experiment"

    # Extract all candidate domain terms from text
    raw_concepts = concept_terms(material, 35)
    filtered_terms = [
        c.replace("_", " ") for c in raw_concepts
        if len(c) > 3 and not c.isdigit() and len(c) < 25 and c.lower() not in _BANNED_UNRELATED_TERMS
    ]

    # Extract embedded manual viva questions if present in content
    embedded_viva_qs = []
    lines = experiment_content.splitlines()
    in_viva = False
    for line in lines:
        cleaned_line = clean(line)
        if re.search(r"(?i)\bviva\s*questions?\b", cleaned_line):
            in_viva = True
            continue
        if in_viva:
            # Check for question line like "1. Q1 What is ...?" or "Q1. What ...?"
            q_match = re.search(r"(?i)^(?:(?:\d+\.|\*|-)\s*)?q\d*[:.-]?\s*(.+\?)", cleaned_line)
            if q_match:
                candidate = q_match.group(1).strip()
                if len(candidate) > 15:
                    embedded_viva_qs.append(validate_and_clean_question(candidate))

    return {
        "topic": detected_topic,
        "key_terms": filtered_terms[:15],
        "embedded_viva_questions": embedded_viva_qs,
    }


def _extract_content_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_+#.-]{2,}", (text or "").lower())
    return {w for w in words if w not in _EVAL_STOP_WORDS}


def _is_repetition_or_keyword_only(question: str, answer: str) -> bool:
    q_words = _extract_content_words(question)
    a_words = _extract_content_words(answer)

    if not a_words:
        return True

    clean_ans = clean(answer).lower()
    non_answers = {
        "idk", "dont know", "don't know", "no idea", "pass", "skip", "none",
        "na", "n/a", "nothing", "dunno", "abc", "xyz", "asdf", "test", "whatever",
        "i don't know", "i dont know", "no answer", "nil"
    }
    if clean_ans in non_answers:
        return True

    if a_words.issubset(q_words) and len(a_words) <= len(q_words):
        return True

    return False


def ollama_available() -> bool:
    try:
        return requests.get(OLLAMA_TAGS_URL, timeout=3).ok
    except Exception:
        return False


def ask_ollama(prompt: str, temperature: float = 0.2, timeout: int = 50) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "num_predict": 450,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json().get("response") or ""
        if not text.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return text.strip()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama connection error: {exc}")


def is_generic_question(q_text: str) -> bool:
    """Detect if a viva question is generic, procedural, or template-like."""
    if not q_text or not isinstance(q_text, str):
        return True
    lower = q_text.lower().strip()
    generic_patterns = [
        r"\bwhat is the objective\b",
        r"\bwhat is the aim\b",
        r"\bexplain the procedure\b",
        r"\bdescribe the experiment\b",
        r"\bwhat are the steps\b",
        r"\bwhat is the title\b",
        r"\bwhat did you learn\b",
        r"\bwhat are the advantages\b",
        r"\bwhat are the disadvantages\b",
        r"\bwhat is the purpose of the experiment\b",
        r"\bexplain the experiment\b",
        r"\bwhat is the description\b",
        r"\bwhat is machine learning\b",
        r"\bwhat is deep learning\b",
        r"\bwhat is python\b",
        r"\bwhat is an experiment\b",
        r"\bwhat tools are used\b",
        r"\bhow does function within the algorithm\b",
        r"\bspecifically chosen and evaluated in this laboratory\b",
        r"\bwhat are the requirements\b",
        r"\bwhat software was used\b",
        r"\bwhat hardware was used\b",
    ]
    for pat in generic_patterns:
        if re.search(pat, lower):
            return True

    # Check for trivial single-clause definition questions like "What is linear regression?" or "What is CNN?"
    if re.match(r"^what is [a-z0-9 -]{2,20}\?$", lower):
        return True
    return False


def contains_unrelated_tech_noise(q_text: str, experiment_content: str) -> bool:
    """Check if question asks about computers/operating systems/hardware/software unless in experiment content."""
    lower_q = q_text.lower()
    exp_lower = experiment_content.lower()
    for term in _BANNED_UNRELATED_TERMS:
        if term in lower_q and term not in exp_lower:
            return True
    return False


def _are_questions_sufficiently_different(q1: str, q2: str) -> bool:
    """Ensure two questions in the same attempt test sufficiently different concepts."""
    w1 = _extract_content_words(q1)
    w2 = _extract_content_words(q2)
    if not w1 or not w2:
        return False
    overlap = len(w1.intersection(w2))
    jaccard = overlap / len(w1.union(w2))
    if jaccard > 0.55:
        return False
    return True


def validate_question(q_text: str, experiment_content: str, detected_topic: str) -> bool:
    """
    Validate that a generated viva question satisfies all strict requirements:
    1. Not empty, length between 25 and 250 chars.
    2. Capitalized and ends with a question mark.
    3. Not generic / procedural / template question.
    4. Does not contain banned unrelated computer/hardware/programming jargon.
    5. Directly relates to the experiment topic / content (key concepts present in experiment).
    6. Easy-to-moderate undergraduate laboratory difficulty.
    7. Clear, natural English suitable for an oral viva.
    """
    if not q_text or not isinstance(q_text, str):
        return False

    q_text = q_text.strip()
    if len(q_text) < 25 or len(q_text) > 250:
        return False

    if not q_text.endswith("?"):
        return False

    if not q_text[0].isupper():
        return False

    # Check for private unicode or corrupt glyphs
    if any(ord(ch) > 0xE000 and ord(ch) < 0xF8FF for ch in q_text) or "\ufffd" in q_text:
        return False

    # Check generic phrasing
    if is_generic_question(q_text):
        return False

    # Check unrelated computer jargon
    if contains_unrelated_tech_noise(q_text, experiment_content):
        return False

    # Validate relation to experiment: must contain at least one content word from topic or experiment
    topic_words = set(re.findall(r"[A-Za-z0-9_]{3,}", (detected_topic or "").lower())) - _EVAL_STOP_WORDS
    exp_words = _extract_content_words(experiment_content)
    q_words = _extract_content_words(q_text)

    if not q_words.intersection(topic_words) and not q_words.intersection(exp_words):
        return False

    return True


def validate_and_clean_question(raw_q: str) -> str:
    """Clean and repair question formatting, capitalization, and punctuation."""
    text = clean(raw_q)
    if not text:
        return ""
    # Strip leading bullet points, question numbers, or asterisks
    text = re.sub(r"^(?:q(?:uestion)?\s*\d*[:.-]?|\d+[:.-]?|\*|-)\s*", "", text, flags=re.I)
    text = clean(text)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if not text.endswith("?"):
        text = text.rstrip(".!:,;") + "?"
    return text


def generate_viva_questions(
    experiment_content: str,
    count: int = QUESTIONS_PER_VIVA,
    student_id: int | None = None,
    attempt_number: int = 1,
    title: str = "",
) -> list[dict]:
    """
    Generate exactly 3 easy-to-moderate undergraduate conceptual/application viva questions.
    Requirements:
    - Questions are strictly grounded in the selected experiment's actual content and topics.
    - Technical topics and concepts are extracted from the experiment beforehand.
    - Each question tests a different topic/concept aspect from that experiment.
    - Questions vary across different experiments and across multiple attempts for the same experiment.
    - Full relevant experiment content is passed to the LLM.
    - All generated questions undergo strict 7-point validation; invalid ones are discarded.
    """
    raw_material = clean(experiment_content)
    if len(raw_material) < 20:
        raise RuntimeError(
            "This experiment does not contain enough manual content to generate viva questions."
        )

    # 1. Extract specific technical topics and concepts from the experiment
    extracted_meta = extract_experiment_topics_and_concepts(raw_material, title)
    detected_topic = extracted_meta["topic"]
    key_terms = extracted_meta["key_terms"]

    target_count = count
    valid_questions: list[dict] = []
    seen_texts: set[str] = set()

    # Define technical aspect rotational foci across attempts
    aspect_sets = [
        [
            ("Core Mathematical Formulation", "mathematical equations, formulas, loss function, or objective criteria"),
            ("Algorithm Mechanics & Parameters", "model parameters, internal step mechanics, or hyperparameter roles"),
            ("Evaluation & Result Interpretation", "performance metrics, output interpretation, or error analysis"),
        ],
        [
            ("Dataset & Feature Preprocessing", "input features, data dimensions, scaling, or variable roles"),
            ("Hyperparameter Tuning & Behavior", "tuning parameters, convergence dynamics, or outlier handling"),
            ("Algorithm Structure & Architecture", "structural components, decision boundaries, or layer functions"),
        ],
        [
            ("Practical Edge Cases & Trade-offs", "overfitting, variance/bias tradeoffs, or practical limitations"),
            ("Comparative Mechanics & Formulation", "mathematical formulation compared to alternatives or baseline"),
            ("Output Metrics & Interpretation", "interpreting validation metrics, score thresholds, or classification results"),
        ],
    ]

    current_aspect_set = aspect_sets[(attempt_number - 1) % len(aspect_sets)]

    def _try_llm_generation() -> list[dict]:
        aspect_instructions = "\n".join(
            f"- Question {i+1} must test: {asp_name} ({asp_desc})."
            for i, (asp_name, asp_desc) in enumerate(current_aspect_set)
        )
        key_terms_str = ", ".join(key_terms[:10]) if key_terms else detected_topic

        # Full relevant experiment content passed directly to LLM (up to 7,500 chars)
        content_slice = raw_material[:7500]

        prompt = f"""You are an undergraduate university laboratory viva examiner.

EXPERIMENT TOPIC: {detected_topic}
EXTRACTED TECHNICAL CONCEPTS: {key_terms_str}

Generate EXACTLY {target_count + 1} clear, easy-to-moderate viva questions specifically testing the technical concepts, parameters, formulas, datasets, and results of THIS particular experiment.

EXAMINER GUIDELINES FOR THIS ATTEMPT:
{aspect_instructions}

STRICT EXAMINER RULES:
1. Every question MUST specifically test the actual subject matter of THIS experiment: {detected_topic}.
2. Each question MUST test a DIFFERENT technical concept from this experiment.
3. Easy-to-moderate undergraduate viva level: clear, natural English, grammatically correct.
4. STRICT PROHIBITIONS:
   - NEVER ask generic questions like "What is the objective?", "Explain the procedure", "What did you learn?", "What are the advantages?".
   - NEVER ask about generic computers, operating systems, software, coding syntax, or programming languages.

Return STRICT JSON ONLY:
{{"questions": [
  {{"question": "Viva question testing concept 1?", "expected_concepts": ["concept1", "concept2"], "difficulty": "easy"}},
  {{"question": "Viva question testing concept 2?", "expected_concepts": ["concept1", "concept2"], "difficulty": "medium"}},
  {{"question": "Viva question testing concept 3?", "expected_concepts": ["concept1", "concept2"], "difficulty": "medium"}},
  {{"question": "Viva question testing concept 4?", "expected_concepts": ["concept1", "concept2"], "difficulty": "medium"}}
]}}

EXPERIMENT CONTENT:
{content_slice}"""

        temp = 0.2 + (((attempt_number - 1) % 4) * 0.08)
        raw = ask_ollama(prompt, temperature=temp, timeout=40)
        payload = parse_json_object(raw)
        if not payload or not isinstance(payload.get("questions"), list):
            return []

        candidates = []
        for item in payload["questions"]:
            if not isinstance(item, dict):
                continue
            q_text = validate_and_clean_question(item.get("question") or "")
            if not q_text:
                continue

            exp_concepts = item.get("expected_concepts") or []
            if not isinstance(exp_concepts, list):
                exp_concepts = [str(exp_concepts)]
            exp_concepts = [clean(str(t)) for t in exp_concepts if clean(str(t))][:8]

            diff = str(item.get("difficulty") or "medium").lower()
            if diff not in {"easy", "medium", "hard"}:
                diff = "medium"

            candidates.append(
                {
                    "question": q_text,
                    "expected_concepts": exp_concepts if exp_concepts else key_terms[:3],
                    "difficulty": diff,
                }
            )
        return candidates

    # 1. Try LLM Generation if Ollama is available
    if ollama_available():
        try:
            llm_candidates = _try_llm_generation()
            for cand in llm_candidates:
                q_text = cand["question"]
                if not validate_question(q_text, raw_material, detected_topic):
                    continue
                key = q_text.lower()
                if key in seen_texts:
                    continue
                if any(not _are_questions_sufficiently_different(q_text, v["question"]) for v in valid_questions):
                    continue
                seen_texts.add(key)
                valid_questions.append(cand)
                if len(valid_questions) >= target_count:
                    break
        except Exception:
            pass

    # 2. If additional questions needed, check topic question bank (rotated by attempt & student ID)
    if len(valid_questions) < target_count and detected_topic in _TOPIC_QUESTION_BANKS:
        topic_bank = _TOPIC_QUESTION_BANKS[detected_topic]
        stride_offset = ((attempt_number - 1) * 3) + ((student_id or 0) % len(topic_bank))
        start_idx = stride_offset % len(topic_bank)

        ordered_candidates = []
        for i in range(len(topic_bank)):
            idx = (start_idx + i) % len(topic_bank)
            ordered_candidates.append(topic_bank[idx])

        for item in ordered_candidates:
            if len(valid_questions) >= target_count:
                break
            q_text = validate_and_clean_question(item["question"])
            if not validate_question(q_text, raw_material, detected_topic):
                continue
            key = q_text.lower()
            if key in seen_texts:
                continue
            if any(not _are_questions_sufficiently_different(q_text, v["question"]) for v in valid_questions):
                continue
            seen_texts.add(key)
            valid_questions.append({
                "question": q_text,
                "expected_concepts": item["expected_concepts"],
                "difficulty": item["difficulty"],
            })

    # 3. If still needed, check embedded manual viva questions
    if len(valid_questions) < target_count and extracted_meta.get("embedded_viva_questions"):
        for emb_q in extracted_meta["embedded_viva_questions"]:
            if len(valid_questions) >= target_count:
                break
            clean_emb = validate_and_clean_question(emb_q)
            if not validate_question(clean_emb, raw_material, detected_topic):
                continue
            key = clean_emb.lower()
            if key in seen_texts:
                continue
            if any(not _are_questions_sufficiently_different(clean_emb, v["question"]) for v in valid_questions):
                continue
            seen_texts.add(key)
            valid_questions.append({
                "question": clean_emb,
                "expected_concepts": key_terms[:4],
                "difficulty": "medium",
            })

    return valid_questions[:target_count]


def evaluate_answer(
    question: str,
    transcript: str,
    experiment_content: str,
    expected_concepts: list[str],
    max_marks: int = 10,
) -> dict:
    """
    Evaluate student's answer strictly against current question and selected experiment material.
    Evidence-based scoring:
    - Completely correct -> high/full marks (7.0 to max_marks)
    - Partially correct -> proportional partial marks (1.0 to 6.0)
    - Clearly incorrect / unrelated / off-topic / empty / repetition -> 0 marks
    """
    material = clean(experiment_content)[:4000]
    answer = clean(transcript)
    question_text = clean(question)
    concepts = [clean(str(c)) for c in (expected_concepts or []) if clean(str(c))]
    if not concepts:
        concepts = concept_terms(material, 10)

    # 1. Pre-check: Empty answer
    if not answer or len(answer) < 2:
        return {
            "score": 0.0,
            "matched_concepts": [],
            "missing_concepts": concepts,
            "incorrect_claims": [],
            "is_correct": False,
            "feedback": "No answer was provided.",
        }

    # 2. Pre-check: Avoidance phrases
    clean_ans = answer.lower().strip()
    non_answers = {
        "idk", "dont know", "don't know", "no idea", "pass", "skip", "none",
        "na", "n/a", "nothing", "dunno", "abc", "xyz", "asdf", "test", "whatever",
        "i don't know", "i dont know", "no answer", "nil", "not sure", "i have no idea",
        "i don't know the answer", "i dont know the answer", "dont know the answer",
    }
    if clean_ans in non_answers:
        return {
            "score": 0.0,
            "matched_concepts": [],
            "missing_concepts": concepts,
            "incorrect_claims": [],
            "is_correct": False,
            "feedback": "The answer contains no genuine explanation or factual content.",
        }

    # 3. Pre-check: Repetition of question words only
    if _is_repetition_or_keyword_only(question_text, answer):
        return {
            "score": 0.0,
            "matched_concepts": [],
            "missing_concepts": concepts,
            "incorrect_claims": [],
            "is_correct": False,
            "feedback": "The answer merely repeats words from the question or contains no genuine explanation.",
        }

    # 4. Pre-check: Domain word overlap
    a_words = _extract_content_words(answer)
    exp_words = (
        _extract_content_words(material)
        | _extract_content_words(question_text)
        | _extract_content_words(" ".join(concepts))
    )
    if not a_words.intersection(exp_words):
        return {
            "score": 0.0,
            "matched_concepts": [],
            "missing_concepts": concepts,
            "incorrect_claims": [],
            "is_correct": False,
            "feedback": "The answer is completely unrelated to the question or experiment material (0 marks).",
        }

    # 5. LLM Evaluation Prompt
    prompt = f"""You are a strict, fair, evidence-based university laboratory viva examiner.

EVALUATION RULES:
1. Compare the student's answer ONLY against the specific QUESTION, EXPECTED CONCEPTS, and EXPERIMENT MATERIAL below.
2. MARKING CRITERIA:
   - Completely correct answer demonstrating expected concepts -> high/full marks (7.0 to {max_marks}).
   - Partially correct answer demonstrating SOME expected concepts with genuine explanation supported by the experiment material -> proportional partial marks (1.0 to 6.0).
   - Clearly incorrect, wrong, or scientifically false answer -> score = 0.0, matched_concepts = [], is_correct = false.
   - Irrelevant, unrelated, or off-topic answer -> score = 0.0, matched_concepts = [], is_correct = false.
   - Mentioning keywords without genuine explanation -> score = 0.0, matched_concepts = [], is_correct = false.
3. MATCHED CONCEPTS:
   - Only list a concept in "matched_concepts" if the student's answer GENUINELY and CORRECTLY demonstrates understanding of that concept.
   - If no expected concepts are demonstrated, "matched_concepts" MUST be [].
4. HARD INVARIANT:
   - If "matched_concepts" is [] OR "is_correct" is false, "score" MUST BE 0.0.

QUESTION:
{question_text}

EXPECTED CONCEPTS:
{json.dumps(concepts)}

EXPERIMENT MATERIAL:
{material}

STUDENT ANSWER:
{answer}

Return STRICT JSON ONLY:
{{
  "score": 0.0,
  "matched_concepts": ["concept1"],
  "missing_concepts": ["concept2"],
  "incorrect_claims": [],
  "is_correct": true,
  "feedback": "Evidence-based explanation of marks awarded or why 0 marks were given."
}}"""

    payload = None
    if ollama_available():
        def _attempt_eval(attempt_idx: int) -> dict:
            temp = 0.0 if attempt_idx == 0 else 0.1
            raw = ask_ollama(prompt, temperature=temp, timeout=35)
            parsed = parse_json_object(raw)
            if not parsed or not isinstance(parsed, dict):
                raise ValueError("Invalid JSON response from Ollama evaluation.")
            return parsed

        try:
            payload = _attempt_eval(0)
        except Exception:
            try:
                payload = _attempt_eval(1)
            except Exception:
                payload = None

    if payload is None:
        matched_lexical = [c for c in concepts if any(w in answer.lower() for w in _extract_content_words(c))]
        if matched_lexical and len(a_words) >= 4:
            est_score = min(float(max_marks), round((len(matched_lexical) / max(1, len(concepts))) * float(max_marks), 1))
            payload = {
                "score": max(1.0, est_score),
                "matched_concepts": matched_lexical,
                "missing_concepts": [c for c in concepts if c not in matched_lexical],
                "incorrect_claims": [],
                "is_correct": True,
                "feedback": f"Evaluated based on core concept match: {', '.join(matched_lexical)}.",
            }
        else:
            payload = {
                "score": 0.0,
                "matched_concepts": [],
                "missing_concepts": concepts,
                "incorrect_claims": [],
                "is_correct": False,
                "feedback": "The answer does not contain concepts supported by the experiment material.",
            }

    # 6. Post-validation & Hard Rule Enforcements
    try:
        score_val = float(payload.get("score", 0))
    except (TypeError, ValueError):
        score_val = 0.0

    is_correct_val = bool(payload.get("is_correct", False))

    raw_matched = payload.get("matched_concepts") or []
    raw_missing = payload.get("missing_concepts") or []
    raw_incorrect = payload.get("incorrect_claims") or []

    if not isinstance(raw_matched, list):
        raw_matched = [str(raw_matched)]
    if not isinstance(raw_missing, list):
        raw_missing = [str(raw_missing)]
    if not isinstance(raw_incorrect, list):
        raw_incorrect = [str(raw_incorrect)]

    matched = [clean(str(item)) for item in raw_matched if clean(str(item))]
    missing = [clean(str(item)) for item in raw_missing if clean(str(item))]
    incorrect = [clean(str(item)) for item in raw_incorrect if clean(str(item))]
    feedback_str = clean(payload.get("feedback") or "")

    # Validate matched_concepts
    validated_matched = []
    ans_lower = answer.lower()
    for item in matched:
        item_words = _extract_content_words(item)
        if item_words and item_words.intersection(a_words):
            validated_matched.append(item)
        elif any(c.lower() in item.lower() or item.lower() in c.lower() for c in concepts) and any(w in ans_lower for w in item_words):
            validated_matched.append(item)

    matched = validated_matched

    # Check feedback sentiment
    rejection_keywords = (
        "irrelevant", "unrelated", "incorrect", "completely wrong",
        "does not address", "no relevant", "0 marks", "does not answer",
        "off-topic", "wrong answer", "fails to address", "not supported"
    )
    fb_lower = feedback_str.lower()
    if any(kw in fb_lower for kw in rejection_keywords) and not matched:
        is_correct_val = False

    # Enforce Hard Scoring Rules
    if not matched or not is_correct_val or score_val <= 0.0:
        score_val = 0.0
        matched = []
        is_correct_val = False
        if not feedback_str or feedback_str == "No additional evaluation feedback was provided.":
            feedback_str = "The answer does not correctly address the question or cover concepts supported by the experiment material (0 marks awarded)."

    score_val = max(0.0, min(float(max_marks), round(score_val, 1)))

    all_missing_set = set(missing)
    for c in concepts:
        if not any(c.lower() in m.lower() or m.lower() in c.lower() for m in matched):
            all_missing_set.add(c)
    missing = [clean(str(x)) for x in all_missing_set if clean(str(x))][:12]

    return {
        "score": score_val,
        "matched_concepts": matched[:12],
        "missing_concepts": missing,
        "incorrect_claims": incorrect[:12],
        "is_correct": is_correct_val,
        "feedback": feedback_str or "No additional evaluation feedback was provided.",
    }