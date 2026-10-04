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
}


def _detect_topic_from_text(content: str, title: str = "") -> str:
    combined = (title + " " + content).lower()
    
    # Priority matching for specific algorithms
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
    raw_concepts = concept_terms(material, 30)
    filtered_terms = [
        c.replace("_", " ") for c in raw_concepts
        if len(c) > 3 and not c.isdigit() and len(c) < 25 and c.lower() not in _BANNED_UNRELATED_TERMS
    ]

    # Categorize extracted elements based on content analysis
    math_terms = []
    param_terms = []
    eval_terms = []
    algo_terms = []

    lower_mat = material.lower()
    
    # Mathematical / Formulation checks
    for kw in ["slope", "intercept", "residual", "eigenvalue", "eigenvector", "entropy", "gain", "probability", "distance", "hyperplane", "margin", "dissimilarity", "least squares", "ols", "covariance"]:
        if kw in lower_mat:
            math_terms.append(kw)

    # Parameters / Hyperparameters
    for kw in ["epsilon", "eps", "min_samples", "k value", "max depth", "regularization c", "gamma", "alpha", "centroid", "medoid", "degrees of freedom"]:
        if kw in lower_mat:
            param_terms.append(kw)

    # Evaluation / Results
    for kw in ["r-squared", "adjusted r-squared", "mean squared error", "mse", "rmse", "accuracy", "silhouette score", "wcss", "inertia", "scree plot", "variance explained", "purity", "gini impurity"]:
        if kw in lower_mat:
            eval_terms.append(kw)

    # Core algorithms
    for kw in ["linear regression", "k-nearest neighbors", "decision tree", "naive bayes", "k-means", "k-medoids", "dbscan", "svm", "pca", "id3", "cart", "pam", "bayes theorem"]:
        if kw in lower_mat:
            algo_terms.append(kw)

    return {
        "topic": detected_topic,
        "key_terms": filtered_terms[:15],
        "math_concepts": math_terms,
        "parameters": param_terms,
        "evaluation_metrics": eval_terms,
        "algorithms": algo_terms,
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


def ask_ollama(prompt: str, temperature: float = 0.2, timeout: int = 35) -> str:
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
                    "num_predict": 400,
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


def validate_question(q_text: str, experiment_content: str, detected_topic: str) -> bool:
    """
    Validate that a generated viva question satisfies all strict requirements:
    1. Not empty, length between 20 and 250 chars.
    2. Capitalized and ends with a question mark.
    3. Not generic / procedural / template question.
    4. Does not contain banned unrelated computer/hardware/programming jargon.
    5. Directly relates to the experiment topic / content (key concepts present in experiment).
    6. Easy-to-moderate undergraduate laboratory difficulty.
    """
    if not q_text or not isinstance(q_text, str):
        return False

    q_text = q_text.strip()
    if len(q_text) < 20 or len(q_text) > 250:
        return False

    if not q_text.endswith("?"):
        return False

    if not q_text[0].isupper():
        return False

    lower_q = q_text.lower()

    # Banned generic heading / procedural phrases
    banned_generic = [
        "what is the objective",
        "what is the aim",
        "explain the procedure",
        "describe the experiment",
        "what are the steps",
        "what is the title",
        "what is the description",
        "what did you learn",
        "what are the advantages",
        "what are the disadvantages",
        "what is the purpose of the experiment",
        "explain the experiment",
        "what is linear regression?",
        "what is decision tree?",
        "what is knn?",
        "what is k-means?",
        "what is machine learning?",
        "what is python?",
    ]
    if any(phrase in lower_q for phrase in banned_generic):
        return False

    # Banned unrelated general computer/hardware/programming noise
    for term in _BANNED_UNRELATED_TERMS:
        if term in lower_q and term not in experiment_content.lower():
            return False

    # Validate relation to experiment: must contain at least one content word from topic or experiment
    topic_words = set(re.findall(r"\w+", (detected_topic or "").lower())) - _EVAL_STOP_WORDS
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


def is_generic_question(q_text: str) -> bool:
    lower_q = (q_text or "").lower()
    banned_phrases = [
        "what is the objective",
        "what is the aim",
        "explain the procedure",
        "describe the experiment",
        "what are the steps",
        "what is the title",
        "what is the description",
        "what did you learn",
        "what are the advantages",
        "what are the disadvantages",
        "what is the purpose of the experiment",
        "explain the experiment",
        "what is linear regression?",
        "what is decision tree?",
        "what is knn?",
        "what is k-means?",
    ]
    return any(phrase in lower_q for phrase in banned_phrases)


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
    - Full relevant experiment content is passed to the LLM when available.
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

    # Deterministic variation seed combining student_id and attempt_number
    seed_val = (student_id or 0) * 100 + (attempt_number or 1)

    # Topic and attempt-inclusive cache key
    content_hash = hashlib.md5(raw_material[:500].encode("utf-8")).hexdigest()
    cache_key = f"{detected_topic}_{count}_{seed_val}_{content_hash}"

    if cache_key in _EXPERIMENT_QUESTION_CACHE:
        cached = _EXPERIMENT_QUESTION_CACHE[cache_key]
        if len(cached) == count:
            return cached

    target_count = count

    def _request_llm_questions(num_needed: int, existing_questions: list[str]) -> list[dict]:
        avoid_str = ""
        if existing_questions:
            avoid_str = "\nDo NOT repeat these existing questions: " + "; ".join(existing_questions)

        key_terms_str = ", ".join(extracted_meta["key_terms"][:10])
        
        # Guide attempt focus to ensure variation between attempts
        attempt_focus_map = [
            "Focus on mathematical formulations, parameter definitions, and core principles.",
            "Focus on algorithm mechanics, data preprocessing, and evaluation metrics.",
            "Focus on hyperparameter tuning, outlier handling, and result interpretation.",
        ]
        focus_guide = attempt_focus_map[(attempt_number - 1) % len(attempt_focus_map)]

        prompt = f"""You are an undergraduate university laboratory viva examiner.

EXPERIMENT TOPIC: {detected_topic or 'Machine Learning Laboratory'}
EXTRACTED CONCEPTS: {key_terms_str}
ATTEMPT VARIATION FOCUS: {focus_guide}

Generate EXACTLY {num_needed} clear, easy-to-moderate viva questions specifically testing the technical concepts, algorithms, parameters, formulas, datasets, and results of THIS particular experiment.{avoid_str}

STRICT EXAMINER GUIDELINES:
1. Every question MUST specifically test the concepts, parameters, formulas, methods, or outputs of THIS experiment: {detected_topic}.
2. Questions MUST test DIFFERENT technical concepts from this experiment (e.g. one question on mathematical/formula components, one on algorithm mechanics/parameters, one on evaluation/interpretation).
3. Questions MUST be easy-to-moderate undergraduate viva level: clear, natural English, with flawless grammar and spelling.
4. STRICT PROHIBITIONS:
   - NEVER ask generic questions like "What is the objective?", "Explain the procedure", "What did you learn?", "What are the advantages?", "What are the steps?".
   - NEVER ask about general computers, operating systems, software, hardware, coding syntax, or programming languages unless explicitly part of the experiment.
5. Questions must be answerable from the experiment material below.

Return STRICT JSON ONLY:
{{"questions": [{{"question": "Technically focused viva question about this experiment?", "expected_concepts": ["concept1", "concept2"], "difficulty": "medium"}}]}}

FULL EXPERIMENT CONTENT:
{raw_material[:7000]}"""

        raw = ask_ollama(prompt, temperature=0.25 + ((attempt_number - 1) * 0.05), timeout=30)
        payload = parse_json_object(raw)
        if not payload or not isinstance(payload.get("questions"), list):
            return []

        results = []
        for item in payload["questions"]:
            if not isinstance(item, dict):
                continue
            q_text = validate_and_clean_question(item.get("question") or "")
            if not validate_question(q_text, raw_material, detected_topic):
                continue

            exp_concepts = item.get("expected_concepts") or []
            if not isinstance(exp_concepts, list):
                exp_concepts = [str(exp_concepts)]
            exp_concepts = [clean(str(t)) for t in exp_concepts if clean(str(t))][:8]

            diff = str(item.get("difficulty") or "medium").lower()
            if diff not in {"easy", "medium", "hard"}:
                diff = "medium"

            results.append(
                {
                    "question": q_text,
                    "expected_concepts": exp_concepts,
                    "difficulty": diff,
                }
            )
        return results

    valid_questions: list[dict] = []
    seen_texts: set[str] = set()
    seen_aspects: set[str] = set()

    # 1. Try LLM generation first if Ollama is available
    if ollama_available():
        try:
            llm_batch = _request_llm_questions(target_count, [])
            for q in llm_batch:
                key = q["question"].lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    valid_questions.append(q)
        except Exception:
            pass

    # 2. Fill from dedicated topic-specific question bank (guarantees 100% topic accuracy, no generic templates, and multi-attempt variety)
    if len(valid_questions) < target_count and detected_topic in _TOPIC_QUESTION_BANKS:
        topic_bank = _TOPIC_QUESTION_BANKS[detected_topic]
        
        # Stride rotation using attempt_number & student_id to sample distinct question subsets per attempt
        # Attempt 1 -> takes questions starting at offset 0 (e.g. idx 0, 1, 2)
        # Attempt 2 -> takes questions starting at offset 3 (e.g. idx 3, 4, 5)
        # Attempt 3 -> takes questions starting at offset 1 (e.g. idx 1, 3, 5)
        stride_offset = (attempt_number - 1) * 3 + (student_id or 0)
        start_idx = stride_offset % len(topic_bank)
        
        ordered_candidates = []
        for i in range(len(topic_bank)):
            idx = (start_idx + i) % len(topic_bank)
            ordered_candidates.append(topic_bank[idx])

        # First pass: pick distinct topic aspects
        for item in ordered_candidates:
            if len(valid_questions) >= target_count:
                break
            q_text = validate_and_clean_question(item["question"])
            if not validate_question(q_text, raw_material, detected_topic):
                continue
            key = q_text.lower()
            aspect = item.get("topic_aspect", "")
            
            if key not in seen_texts:
                if aspect and aspect in seen_aspects and len(valid_questions) < target_count - 1:
                    continue
                seen_texts.add(key)
                if aspect:
                    seen_aspects.add(aspect)
                valid_questions.append({
                    "question": q_text,
                    "expected_concepts": item["expected_concepts"],
                    "difficulty": item["difficulty"],
                })

        # Second pass: fill remaining if needed
        if len(valid_questions) < target_count:
            for item in ordered_candidates:
                if len(valid_questions) >= target_count:
                    break
                q_text = validate_and_clean_question(item["question"])
                if not validate_question(q_text, raw_material, detected_topic):
                    continue
                key = q_text.lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    valid_questions.append({
                        "question": q_text,
                        "expected_concepts": item["expected_concepts"],
                        "difficulty": item["difficulty"],
                    })

    # 3. For custom uploaded manuals, dynamically generate questions strictly grounded in the extracted concepts & parameters
    if len(valid_questions) < target_count:
        clean_terms = extracted_meta["key_terms"]
        if not clean_terms:
            clean_terms = ["model parameters", "evaluation metric", "dataset features", "training process"]

        shift = seed_val % len(clean_terms)
        rotated_terms = clean_terms[shift:] + clean_terms[:shift]

        custom_patterns = [
            ("In this experiment, how does {c} function within the algorithm, and what role does it play?", ["concept identification", "role"]),
            ("Why is {c} specifically chosen and evaluated in this laboratory experiment?", ["method rationale", "evaluation"]),
            ("How does tuning or optimizing {c} influence the final output and accuracy in this experiment?", ["parameter tuning", "accuracy"]),
            ("What key metric or observation regarding {c} indicates that the model has trained successfully?", ["result interpretation", "metric"]),
            ("In what practical engineering scenario would you apply the {c} technique demonstrated here?", ["practical application", "scenario"]),
        ]
        t_shift = seed_val % len(custom_patterns)
        rotated_patterns = custom_patterns[t_shift:] + custom_patterns[:t_shift]

        for idx, (tmpl, tmpl_concepts) in enumerate(rotated_patterns):
            if len(valid_questions) >= target_count:
                break
            term = rotated_terms[idx % len(rotated_terms)]
            formatted_q = validate_and_clean_question(tmpl.format(c=term))
            if validate_question(formatted_q, raw_material, detected_topic):
                key = formatted_q.lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    valid_questions.append({
                        "question": formatted_q,
                        "expected_concepts": tmpl_concepts + [term],
                        "difficulty": "medium",
                    })

    final_questions = valid_questions[:target_count]
    _EXPERIMENT_QUESTION_CACHE[cache_key] = final_questions
    return final_questions


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
            raw = ask_ollama(prompt, temperature=temp, timeout=30)
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