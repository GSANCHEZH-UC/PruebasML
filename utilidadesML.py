import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analisis_variable_barras(pdata, xname, yname, title):
    conteo = pdata.groupby([xname, yname])[yname].count().unstack()
    
    # Calcula los porcentajes
    porcentajes = conteo.div(conteo.sum(axis=1), axis=0) * 100

    # Crea el gráfico de barras
    ax = porcentajes.plot(kind='bar', stacked=True)
    plt.title(title, fontsize=10)
    plt.xlabel(xname, fontsize=10)
    plt.ylabel('Porcentaje', fontsize=10)
    plt.xticks(rotation=0)
    plt.legend(title=yname, loc='upper right', fontsize=8)

    # Agregar valores a las barras
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x+width/2, y+height/2, '{:.1f}%'.format(height), horizontalalignment='center', verticalalignment='center', fontsize=8)

    plt.tight_layout()
    plt.show()

def analisis_variable_violin(pdata, xname, yname, title):
    sns.violinplot(data=pdata, x=xname, y=yname)
    plt.xlabel(xname, fontsize=10)
    plt.ylabel(yname, fontsize=10)
    plt.title(title, fontsize=10)
    plt.show()

def prueba_t(data, dependiente, independiente, g1, g2, afect):
    grupo_1 = data[data[dependiente] == g1][independiente]
    grupo_2 = data[data[dependiente] == g2][independiente]

    # Realizar la prueba t
    t_stat, p_value = stats.ttest_ind(grupo_1, grupo_2)

    print(f"*** Prueba t para {independiente} ***")
    print(f'Estadístico t: {t_stat:.5f}')
    print(f'Valor p: {p_value:.5f}')
    if p_value < 0.05:
        print("Se rechaza la hipótesis nula. Las medias son significativamente diferentes.")
    else:
        print("No se puede rechazar la hipótesis nula. Las medias son similares.")
        afect.drop(independiente, axis=1, inplace=True)
    print(data.groupby(dependiente)[independiente].mean())
    print("\n")

def normalizar(data, columna):
    media = data[columna].mean()
    desviacion_estandar = data[columna].std()
    data[columna] = (data[columna] - media) / desviacion_estandar