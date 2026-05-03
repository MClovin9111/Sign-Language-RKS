# Esto lo vamos a usar para representar imagenes como matrices o vectores
# Igual permite hacer operaciones con vectores
import numpy as np

# Esto lo usamos para guardar el modelo entrenado y los parametros de RKS
import joblib

# Vamos a usar regresion logistica para clasificar las señas
from sklearn.linear_model import LogisticRegression

from config import LABEL_MAP

class RandomKitchenSinks:
    def __init__(self, n_funciones=1000, gamma=1.0):
        # numero de funciones aleatorias para la proyección
        self.n_features = n_funciones
        # gama es el parametro de la RBF, controla la escala de las funciones aleatorias
        self.gamma = gamma
        # Pesos y bias aleatorios para la proyección
        self.W = None
        self.b = None

    def fit(self, X):
        # X es el dataset de entrada
        # n_samples es el numero de ejemplos
        # n_dim es la dimension de cada ejemplo (numero de caracteristicas)
        n_samples, n_dim = X.shape

        # Inicializamos los pesos y bias aleatorios para la proyección 
        self.W = np.random.normal(0, np.sqrt(2*self.gamma), (n_dim, self.n_features))
        self.b = np.random.uniform(0, 2*np.pi, self.n_features)

        # Llamamos a transform que es donde se aplica la formula de RKS para transformar los datos
        return self._transform(X)
    
    def _transform(self, X):
        # Proyectamos los datos usando la formula de RKS
        Z = np.dot(X, self.W) + self.b
        return np.sqrt(2.0 / self.n_features) * np.cos(Z)
    
    def transform(self, X):
        # Solo transformamos los datos usando los pesos y bias ya definidos
        return self._transform(X)
    

# Entranamos el modelo
def train(X_train, y_train, path="sign_model.pkl"):
    # Creamos una instancia de RKS de la clase que definimos antes
    rks = RandomKitchenSinks(n_funciones=1000, gamma=0.5)
    X_train_rks = rks.fit(X_train)

    # Creamos un modelo de regresion logistica para clasificar las señas
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_rks, y_train)

    # Guardamos el modelo y el RKS juntos usando joblib
    joblib.dump((rks, model), path)

#Cargamos el modelo para hacer predicciones
def load_model(path="sign_model.pkl"):
    return joblib.load(path)

# Vamos con la funcion de prediccion
def predict(image: np.ndarray, model_path="sign_model.pkl") -> str:
    # Cargamos el modelo y el RKS
    rks, model = load_model(model_path)
    
    # Vectorizamos la imagen 
    X = image.reshape(1, -1)  # Convertimos la imagen a un vector de una sola fila

    # Transformamos la imagen usando RKS
    X_rks = rks.transform(X)

    # Hacemos la prediccion con el modelo de regresion logistica
    pred = model.predict(X_rks)[0]

    # Mapeamos la prediccion a la letra correspondiente usando LABEL_MAP
    return LABEL_MAP[pred]
