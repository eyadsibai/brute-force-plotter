import seaborn as sns

from utils import ignore_if_exist_or_save


@ignore_if_exist_or_save
def histogram_violin_plots(data, axes, file_name=None):
    # histogram
    sns.distplot(data, ax=axes[0], axlabel='')
    sns.violinplot(data, ax=axes[1])


@ignore_if_exist_or_save
def bar_plot(data, col, hue=None, file_name=None):
    sns.countplot(col, hue=hue, data=data)


@ignore_if_exist_or_save
def scatter_plot(data, col1, col2, file_name=None):
    sns.regplot(x=col1, y=col2, data=data, fit_reg=False)


@ignore_if_exist_or_save
def bar_box_violin_dot_plots(data, category_col, numeric_col, axes,
                             file_name=None):
    sns.barplot(category_col, numeric_col, data=data, ax=axes[0])
    sns.boxplot(category_col, numeric_col,
                data=data[data[numeric_col].notnull()], ax=axes[2])
    sns.violinplot(category_col, numeric_col, data=data, kind='violin',
                   ax=axes[3])
    sns.stripplot(category_col, numeric_col, data=data, jitter=True, ax=axes[1])


@ignore_if_exist_or_save
def heatmap(data, file_name=None):
    sns.heatmap(data=data, annot=True)
