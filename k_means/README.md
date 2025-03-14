# Implementación de K-means desde cero

En esta práctica se implementa el algoritmo de K-means desde cero y se realizan una serie de pruebas sobre un dataset sintético para comprobar su correcto funcionamiento.

> [!NOTE]
> Usuario de GitHub para la corrección: **usbt0p**
>
> Si quieres ver el documento renderizado correctamente, puedes hacerlo en [este enlace](https://github.com/usbt0p/apau_II) o utilizando un visor de Markdown como el de Visual Studio Code.

## Explicación del código

El código está documetado, pero esta es una explicación general de su funcionamiento.
En primer lugar, k-means se ha implementado como una clase, `KMeansFromScratch`. Esta es su estructura:

```python
class KMeansFromScratch():
    '''K-means clustering algorithm from scratch.'''

    def __init__(self, data: np.ndarray, k: int, n_iter: int, points_as_centroids: bool = True):

    def select_initial_centroids(self, points_as_centroids) -> np.ndarray:

    def get_new_clusters(self, points, centroids) -> np.ndarray:

    def update_centroids(self, cluster_idxs):

    def run(self, debug=False):

    def __str__(self):
```

- El constructor `__init__` de la clase recibe los datos, el número de clusters a formar, el número de iteraciones y un booleano para la inicialización de los centroides. Esta inicialización se realiza automáticamente durante la instanciación de la clase.

- `select_initial_centroids` selecciona los centroides iniciales. El booleano `points_as_centroids` es un flag que indica si los centroides Written without AI assisted generation.iniciales se elegirán de entre los puntos del dataset, o si se seleccionarán aleatoriamente dentro del rango de los datos.

- `get_new_clusters` asigna cada punto al cluster correspondiente al centroide más cercano. En caso de  que varios centroides estén a la misma distancia, se asigna a aquel que tenga menos puntos asignados. Este proceso se realiza con la ayuda de un mapa `ppc`que almacena el número de puntos asignados a cada cluster.
El cálculo de la distancia se realiza con la función `np.linalg.norm`, que calcula la norma euclídea entre dos puntos.
El resultado es un array de índices que indican a qué cluster pertenece cada punto.

- `update_centroids` recalcula los centroides, tomando como su nueva posición la media de las coordenadas de los puntos asignados a su cluster. La selección de puntos del cluster y media se realizan usando la indexación booleana de numpy y la función `np.mean` respectivamente.

- `run` se encarga de ejecutar el algoritmo de k-means. En cada iteración, se asignan los puntos a los clusters, se recalculan los centroides y se comprueba si se ha alcanzado el número de iteraciones. Antes de actualizar los centroides, se debe comprobar si todos ellos tienen al menos un punto asignado. En caso contrario, se selecciona un nuevo punto aleatorio del dataset como centroide, y no se cuenta la iteración actual. 

    El flag `debug` permite mostrar información adicional sobre el proceso de clustering, mostrando plots 2D de los clusters en cada iteración.
    El pseudocódigo utilizado para el algoritmo es el siguiente:

    ```python
    initialize centroids
    while iter != n_iter:
        assign each point to the cluster of the closest centroid
        if all centroids have points:
            update centroids to be the mean of their elements
            n_iter += 1
        else: 
            randomly reinitialize empty centroids
    ```

- `__str__` devuelve una representación en string de la clase, mostrando los centroides y los clusters, y los datos del algoritmo.

## Resultados gráficos

A continuación se muestran los resultados obtenidos con el dataset sintético. Se ha utilizado un dataset de 300 puntos con 4 clusters, y se ha ejecutado el algoritmo durante 4 iteraciones. 

En las imágenes se puede observar cómo los clusters se van formando a medida que se ejecutan las iteraciones.

<img src="imgs/kmeans_0.png" width="400">
</img>

<img src="imgs/kmeans_1.png" width="400">
</img>

<img src="imgs/kmeans_2.png" width="400">
</img>

<img src="imgs/kmeans_3.png" width="400">
</img>

<img src="imgs/kmeans_4.png" width="400">
</img>

En la primera iteración, los centroides se han seleccionado aleatoriamente, y los puntos se han asignado a los clusters más cercanos. En las siguientes, los centroides se actualizan, y los puntos se reasignan a los clusters. Este proceso se repite hasta que se alcanza el número de iteraciones especificado.


