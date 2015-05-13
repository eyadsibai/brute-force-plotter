import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from utils import ignore_if_exist_or_save
import math

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        width = rect.get_width()

        if math.isnan(height):
            height = 0.0
        if (height, width) == (1.0,1.0):
            continue
        plt.text(rect.get_x()+width/2., 1.0+height, '%d'%int(height),
                ha='center', va='bottom')

@ignore_if_exist_or_save
def histogram_violin_plots(data, axes, file_name=None):
    # histogram
    sns.distplot(data, ax=axes[0], axlabel='')
    sns.violinplot(data, ax=axes[1])
    sns.despine(left=True)


@ignore_if_exist_or_save
def bar_plot(data, col, hue=None, file_name=None):
    sns.countplot(col, hue=hue, data=data.sort(col))
    sns.despine(left=True)

    subplots = [x for x in plt.gcf().get_children() if isinstance(x, matplotlib.axes.Subplot)]
    for plot in subplots:
        rectangles = [x for x in plot.get_children() if isinstance(x, matplotlib.patches.Rectangle)]
    autolabel(rectangles)



@ignore_if_exist_or_save
def scatter_plot(data, col1, col2, file_name=None):
    sns.regplot(x=col1, y=col2, data=data, fit_reg=False)
    sns.despine(left=True)


@ignore_if_exist_or_save
def bar_box_violin_dot_plots(data, category_col, numeric_col, axes,
                             file_name=None):
    sns.barplot(category_col, numeric_col, data=data, ax=axes[0])
    sns.boxplot(category_col, numeric_col,
                data=data[data[numeric_col].notnull()], ax=axes[2])
    sns.violinplot(category_col, numeric_col, data=data, kind='violin', inner="quartile", scale='count',
                   ax=axes[3])
    sns.stripplot(category_col, numeric_col, data=data, jitter=True, ax=axes[1])
    sns.despine(left=True)


@ignore_if_exist_or_save
def heatmap(data, file_name=None):
    cmap = "BuGn" if (data.values >= 0).all() else "coolwarm"
    sns.heatmap(data=data, annot=True, fmt="d", cmap=cmap)
    sns.despine(left=True)

