<p align="center">
  <img src="https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/Banners%20Diplodatos.png">
</p>

<h1 align='center'>

<a href="https://diplodatos.famaf.unc.edu.ar/">
    <p align="center"> Diplomatura en Ciencia de Datos, Aprendizaje Automatico y sus aplicaciones
    </h1>



<h1 align='center'>
Contenido.-
</h1>

## ➜ Linear Regression
[01_Linear_Regression.ipynb](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/01_Linear_Regression.ipynb)<br>
Dataset: [Diabetes de Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)<br>
`LinearRegression` `mean_squared_error` `r2_score` `plotly.express` `plotly.graph_objects` `datasets.load_diabetes()`
  
## ➜ Logistic Regression
Notebook:[02_Logistic_Regression.ipynb](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/01_Linear_Regression.ipynb)<br> 
Dataset: [Iris de Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)<br>
`LogisticRegression` `datasets.load_iris` `sns.pairplot` `plot_decision_boundary`

## ➜ Naive Bayes
Notebook: [03_Naive_Bayes.ipynb](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/03_Naive_Bayes.ipynb)<br>
Dataset: [Spam/Ham e-mails](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/spam_or_ham.txt)<br>
`MultinomialNB` `train_test_split` `CountVectorizer` `classification_report` `confusion_matrix`
  
## ➜ Perceptron
Notebook: [04_Perceptron.ipynb](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/04_Perceptron.ipynb)<br>
Dataset: Synthetic Dataset
```python
def load_dataset_up_down(size, seed=39):
    np.random.seed(seed)
    x = np.random.poisson(5, size) * (np.random.randint(0,2, size) * 2 - 1)
    y = (np.random.poisson(5, size) + 1) * (np.random.randint(0,2, size) * 2 - 1)
    X = np.array(list(zip(x,y)))
    Y = (X[:,1] > 0).astype(np.int8)
    return X, Y
```
`Perceptron`
  
  
## ➜ Support Vector Machines
Notebook: [05_Support_Vector_Machines.ipynb](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/05_Support_Vector_Machines.ipynb)<br>
Dataset: Synthetic Dataset
```python
def load_dataset_up_down(size, seed = 39):
    np.random.seed(seed)
    x = np.random.poisson(5, size) * (np.random.randint(0,2,size) * 2 - 1)
    y = (np.random.poisson(5, size) + 1) * (np.random.randint(0,2,size) * 2 - 1)
    X = np.array(list(zip(x,y)))
    Y = (X[:,1] > 0).astype(np.int8)
    return X, Y
```
`LinearSVC` `make_blobs` `DecisionBoundaryDisplay`
  
## ➜ Kernels
Notebook: [06_Kernels.ipynb](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/06_Kernels.ipynb)<br>
Dataset: Synthetic Dataset
```python
def load_dataset_disks(size, seed=39):    
    def get_sample_disk(from_, to, size):
        length = np.random.uniform(from_, to, size)
        angle = np.pi * np.random.uniform(0, 2, size)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        return np.array(list(zip(x,y)))

    np.random.seed(seed)
    size1 = int(size / 2)
    size2 = size - size1
    X = np.concatenate((get_sample_disk(0, 1, size1), get_sample_disk(1, 2, size2)))
    Y = np.concatenate((np.ones(size1), -np.ones(size2)))
    return X, Y
```
`SVC` `train_test_split` `classification_report` `kernel = 'sigmoid'` `kernel='rbf'` `kernel='poly'`
  
## ➜ Decision Trees
Notebook: [07_Decision_Trees.ipynb](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/07_Decision_Trees.ipynb)<br>
Dataset 1: [Iris de Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)<br>
Dataset 2: [Boston de Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)<br>
`DecisionTreeClassifier` `plot_tree` `load_iris` `load_boston`
  
## ➜  Random Forest
Notebook: [08_Random_Forest.ipynb](https://github.com/nicoambrosis/Aprendizaje-Supervisado/blob/main/08_Random_Forest.ipynb)<br>
Dataset : [breast_cancer de Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)<br>
`load_breast_cancer` `train_test_split` `classification_report` `confusion_matrix` `ConfusionMatrixDisplay` `DecisionTreeClassifier` `plot_tree` `plot_tree`
