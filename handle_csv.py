import pandas as pd
import matplotlib.pyplot as plt

def show_chart(csv):
    ax=pd.value_counts(csv['type'],ascending=True).plot(kind='barh',
                                                       fontsize="20",
                                                       title="Class Distribution",
                                                       figsize=(40,100))
    ax.set(xlabel="Images per emotion", ylabel="Emotions")
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.title.set_size(40)
    plt.show()