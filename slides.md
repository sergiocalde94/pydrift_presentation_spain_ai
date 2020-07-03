---
title: Spain AI - pydrift
---

### Mi modelo ha dejado de funcionar y ~~no~~ sé por qué

Sergio Calderón Pérez-Lozao

Junio 2020

<a class="icon-github social-button color" href="http://github.com/sergiocalde94" target="_blank"></a>
<a class="icon-twitter social-button color" href="http://twitter.com/sergiocalde94" target="_blank"></a>
<a class="icon-linkedin social-button color" href="https://es.linkedin.com/in/sergiocalde94" target="_blank"></a>

---

## Sobre mí

- Data Scientist en Orange 👨‍🔬
- Modelos reproducibles y código limpio 🖥️
- No reinvento la rueda ⚙️
- Cerveza y pasarlo bien 🍺

----

<img src="https://github.com/sergiocalde94/pydrift_presentation_spain_ai/blob/master/images/datafreaks.jpeg?raw=true" width="85%">

---

## ¿Qué es un Data Scientist?

> Person who is better at statistics than any software engineer and better at software engineering than any statistician. — Josh Wills, Director of Data Engineering at Slack

----

## ¿Qué es un Data Scientist?

> En el principio fui estadístico, luego me llamaron minero de datos, ahora me conocen como Data Scientist, ¿que nos deparará el futuro? — José Luis Cañadas Reche, Senior Data Scientist at Orange

----

## ¿Qué es un Data Scientist?

> A veces hay un poco de estadística en mi código. — Tomás Borrella Martín, ~~Product Owner~~ Data Scientist at Orange

----

## Y para mi... Resuelve problemas

- Cuestionarse todo y dar soluciones a problemas, ~~normalmente~~, con datos
- Las soluciones a estos problemas tienen que ser reproducibles
- Las tecnologías no son lo importante
    - El "cómo" vs el "porqué"

---

### ¿Que problema intenta resolver `pydrift`?

- Los modelos se degradan 📉
- Las fuentes de datos no son tan homogéneas como nos gustaría 🗃️
- En muchas ocasiones los comportamientos son estacionales 🍂
- Hay que mimar mucho nuestra target 🎯
    - Además de alinearla con el problema que queremos resolver, pero eso es otro tema

----

#### ¿Por qué `pydrift`?

- Escrita en **py**thon
- Mide el **drift**
- Hay alguna solución escrita
    - [azureml.datadrift](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets) implementada en python (requiere esta plataforma)
    - [drifter](https://modeloriented.github.io/drifter/) escrita en R

---

#### ¿Cuánto tiempo está mi modelo en producción sin degradarse?

<img src="https://glassboxmedicine.files.wordpress.com/2019/02/roc-curve-v2.png" width="45%">

----

#### Causas de mal comportamiento

- Nuestro modelo es muy complejo y no generaliza bien: *overfitting* ‼️
- Nuestro modelo es muy simple y no aprende patrones: *underfitting* ‼️

----

#### Causas de degradación

- Las `X` no tienen las mismas distribuciones que las que se aprendió el modelo (covariate shift) ‼️
- La `y` no tiene la misma relación con las `X` que tenía a la hora de entrenar (concept shift) ‼️
- La `y` cambia de distribución (prior probability shift) ‼️

###### [Covariate shift, the hidden problem of real world data science](https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/)️

----

> Definiremos como **DataDrift**  esos cambios en las **X** y como **ModelDrift** los que se refieren a la **y**

----

<img src="https://i.redd.it/kvvgv6zzhtp11.png" width="45%">

---

#### Pero entonces, qué son *DataDrift* y *ModelDrift*

<img src="https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif" width="45%">

----

#### Concepto

- Modelo discriminatorio
- Comparación de distribuciones

----

#### Dataframes a izquierda y derecha

```python
from pydrift import DriftChecher

# Read left and right data with pandas
...

DriftChecher(df_left, df_right)
```

- Modelo discriminatorio con target ficticio **is_left** con valores **1** o **0**
- **Kolmogorov** y **chisquare** para comparar las variables de ambos conjuntos

----

#### Python 🐍

Trabaja por debajo con **scipy** y **shap**, basándose en las estructuras de **pandas**, modelos de **sklearn** y apoyándose en las visualizaciones de **plotly**

----

#### ¿Y como lo calculamos?

```python
import pandas as pd

from pydrift import DataDriftChecker

# Read train and test data with pandas
...

data_drift_checker = DataDriftChecker(X_train, X_test)
data_drift_checker.ml_model_can_discriminate()
```

----

- Todos los métodos se acompañan de visualizaciones 
- En un entorno productivo podrían ser desactivados fácilmente, así como los mensajes que se le imprimen al usuario 

---

#### Tipos de visualizaciones

- **InterpretableDrift**: Tanto para **DataDrift** como **ModelDrift** (incluso para cualquier modelo de **sklearn**) 👀
- Se puede aplicar al modelo discriminador o a tu modelo 👬

----

#### Variables más discriminadoras

- A través de la librería [shap](https://shap.readthedocs.io/en/latest/)
- Nos muestra las variables que discriminan mejor

```python
data_drift_checker.ml_model_can_discriminate()
```

<img src="https://sergiocalde94.github.io/pydrift/docs/images/most_discriminative_features.png" width="45%">

----

#### Comparación de histogramas

- A través de la librería [plotly](https://plotly.com/python/)
- Compara los histogramas en train y test de las variables seleccionadas

```python
(model_drift_checker
 .interpretable_drift_classifier_model
 .both_histogram_plot('Sex'))
```

<img src="https://sergiocalde94.github.io/pydrift/docs/images/histogram_comparison.png" width="45%">

----

#### Importancia escalada versus coeficiente de drift

- A través de la librería [plotly](https://plotly.com/python/)
- Pinta un mapa en el que tenemos la posibilidad de ver si las variables donde tenemos drift son críticas o no

```python
(model_drift_checker
 .show_feature_importance_vs_drift_map_plot(top=5))
```

<img src="https://sergiocalde94.github.io/pydrift/docs/images/feature_importance_scaled_vs_drift_coefficient.png" width="45%">

----

#### Pesos para reentrenar

- A través de la librería [plotly](https://plotly.com/python/)
- Idea cogida de [este](https://towardsdatascience.com/how-dis-similar-are-my-train-and-test-data-56af3923de9b) post de Medium
- Muestra la distribución de los pesos para reentrenar

```python
# Then you can apply `weights` to retrain your model
weights = model_drift_checker.sample_weight_for_retrain()
```

<img src="https://sergiocalde94.github.io/pydrift/docs/images/weights.png" width="45%">

----

#### Comparación de *Partial Dependence  Plot*

- A través de la librería [plotly](https://plotly.com/python/)
- Nos muestra el partial depencence plot para la variable seleccionada en cada conjunto

```python
(model_drift_checker
 .interpretable_drift_classifier_model
 .partial_dependence_comparison_plot('Age'))
```

<img src="https://sergiocalde94.github.io/pydrift/docs/images/partial_dependence_comparison.png" width="45%">

----

#### Drift por bins ordenados

- A través de la librería [plotly](https://plotly.com/python/)
- Idea cogida de la librería de R [drifter](https://modeloriented.github.io/drifter/)
- Ordena, corta en `n` bins y cuenta cuantos registros de cada conjunto hay en cada bin

```python
(model_drift_checker
 .interpretable_drift_classifier_model
 .drift_by_sorted_bins_plot('Embarked'))
```

<img src="https://sergiocalde94.github.io/pydrift/docs/images/shift_plot_by_bin.png" width="45%">

---

## Casos de uso

----

#### Comparación de muestras ✔️

- Dividimos siempre en conjuntos de train y validación
    - Aleatoriamente (80/20)
    - Out of time
- Muchas veces trabajamos con dos origenes
    - Por ejemplo campañas de marketing vs nuestros datos

----

#### Cambios drásticos (COVID) ✔️

- En Orange, al igual que en casi todas las industrias, nuestros datos han cambiado mucho
    - ¿Las tendencias **pre-covid** son iguales que las **covid** y que las **post-covid**?
- Puede ser también un cambio en las tarifas que se ofrecen
    - ¿Comprarán igual los clientes si en vez de 10€ les cobramos 20€?

----

#### Explicar comportamientos pasados ✔️

- ¿Por qué a **mismo score hay grupos que se comportan distinto**?
- **Grupos en los que el modelo funcionó peor** -> Estudiar drift en estos grupos

----

#### No es la biblia ❌

- **DataDrift** no siempre implica **ModelDrift**
- **ModelDrift** no siempre implica **DataDrift**
- **La figura del data scientist sigue siendo igual de importante**, simplemente facilita el trabajo

---

#### Demo time

- [DriftChecker](https://nbviewer.jupyter.org/github/sergiocalde94/pydrift/blob/master/notebooks/1-Titanic-Drift-Demo.ipynb)
- [DataDrift](https://nbviewer.jupyter.org/github/sergiocalde94/pydrift/blob/master/notebooks/1-Titanic-Data-Drift-Demo.ipynb)
- [ModelDrift](https://nbviewer.jupyter.org/github/sergiocalde94/pydrift/blob/master/notebooks/1-Titanic-Model-Drift-Demo.ipynb)
- [DriftCheckerEstimator](https://nbviewer.jupyter.org/github/sergiocalde94/pydrift/blob/master/notebooks/1-Titanic-DriftCheckerEstimator-Demo.ipynb)
- [Comparación de muestras](https://nbviewer.jupyter.org/github/sergiocalde94/pydrift_presentation_spain_ai/blob/master/notebooks/Sample-Comparation.ipynb)

---

## y, ¿Cómo lo arreglo?

<img src="https://media.giphy.com/media/SphdK3sFxFYZy/giphy.gif" width="45%">

----

#### No existe una formula mágica, pero...

- Muchas veces los drifts son **fallos en las fuentes de datos**
- A veces ocurre en variables que **no son importantes** en la decisión de nuestro modelo
- Puede servir para **comunicar** que tu modelo no se va a comportar como negocio espera
- También para **cambiar las variables** de tu modelo para un determinado momento

----

#### Garbage In Garbage Out

<img src="https://imgs.xkcd.com/comics/machine_learning_2x.png" width="45%">

---

#### Limitaciones ~de momento😈~

- Solo válido si predecimos con todos los datos 👎
- Al comparar distribuciones y fijarnos en el pvalue dependemos del tamaño muestral 👎
- Solo es compatible con dataframes de pandas y modelos que sigan el API de sklearn 👎
- Seguramente habrá más no conocidas en este momento 👎

---

#### ~¿Puedo contribuir?~ ¡Puedes contribuir!

- Cualquier idea es bienvenida 💡
- Reporte de bugs es necesario 🐛
    - Con el mayor uso habrá gente que descubrirá fallos
- Actualmente no hay guía para contribuir pero pronto la subiré 🙏
    - Si no sabes como déjame un mensaje en cualquier red social

---

#### ¡Gracias! ¿Alguna pregunta?

<img src="https://media.giphy.com/media/xUPGcJ1EHTMBkRHxnO/giphy.gif" width="45%">
