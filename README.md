# Solver 2D de la Ecuación de Schrödinger: Potencial de Pullen-Edmonds 

Este repositorio contiene el desarrollo del proyecto semestral para la asignatura **Física Computacional**. El objetivo principal del proyecto es construir un solver numérico eficiente para calcular los niveles de energía y reconstruir las funciones de onda espaciales de una partícula atrapada en un potencial bidimensional acoplado.

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

* Informe Avance de Proyecto: Documento fuente en LaTeX que contiene el informe académico del proyecto correspondiente al avance.
* Avance_Proyecto_Enzo_Ocaranza_70%.py: Script de Python totalmente funcional que ejecuta el motor de cálculo, realiza la diagonalización, reconstruye la amplitud espacial de la función de onda y genera los gráficos dinámicos.

## Métodos Numéricos Implementados

A la fecha, el motor computacional cubre de forma rigurosa los siguientes aspectos físicos y matemáticos:

1. Representación Matricial Discreta: Construcción de los operadores de posición y momento en 1D utilizando las matrices de los operadores escalera sobre la base truncada del oscilador armónico.
2. Extensión al Espacio Bidimensional: Uso del producto de Kronecker (scipy.sparse.kron) para proyectar los operadores unidimensionales al espacio algebraico 2D.
3. Diagonalización con el Algoritmo de Lanczos: Implementación de la función scipy.sparse.linalg.eigsh para resolver de forma óptima el problema de autovalores, extrayendo únicamente los niveles de menor energía.
4. Reconstrucción Espacial Cuántica: Evaluación de los polinomios de Hermite mediante scipy.special.eval_hermite junto con una envolvente Gaussiana y su factor de normalización físico exacto para prevenir divergencias numéricas. Se mapea la amplitud real usando mapas de colores divergentes (RdBu_r), donde el rojo representa fases positivas y el azul fases negativas.

## Próximos Pasos

Para completar la entrega final del proyecto a finales de semestre, el trabajo restante se enfocará en:
* Implementar el Clasificador Automático de Simetrías basado en las propiedades del grupo de simetría C4v. El algoritmo evaluará de forma lógica la paridad de la matriz de coeficientes respecto a su transpuesta para etiquetar autónomamente cada estado como Simétrico o Antisimétrico.

## Autor
* Enzo Ocaranza - Departamento de Física, Universidad Técnica Federico Santa María
