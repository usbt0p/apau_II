# APAU II: Aprendizaje Automático No Supervisado

## Proyecto 1: Implementación de K-means desde cero
[El primer proyecto](k_means/) consiste en implementar el algoritmo de K-means desde cero y realizar una serie de pruebas sobre un dataset sintético para comprobar su correcto funcionamiento.

### Features:
- Para la inicialización de centroides, debería ser posible elegir entre:
    - Inicialización aleatoria de puntos no necesariamente existentes, dentro de los límites de los datos
    - Inicialización aleatoria de centroides a partir de puntos del dataset
- Medida de distancia: Euclidiana
- Si un punto es equidistante de dos centroides, se aplica el balance de tamaño 
    (asignación al centroide con menos puntos)
- Criterio de convergencia: debería detenerse después de max_iter iteraciones (a especificar como un parámetro)