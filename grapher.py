from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter,HourLocator
import seaborn as sns
import argparse
import probscale
import pandas as pd
import numpy as np
from pandas import DataFrame
plt.style.use('ggplot')


class stat_grapher:
    '''
    input:
        data: pandas df
        symbol_col: symbol column name
        value_col :value column name

    total methods:
        get_hist
        get_boxplot
        get_probplot
        get_trend
        get_summary
    '''

    def __init__(self,data,symbol_col,value_col):
        self.data = data
        self.symbol_col = symbol_col
        self.value_col = value_col


    def get_hist(self):
        data = self.data
        symbol_col = self.symbol_col
        value_col = self.value_col

        cat = list(set(data[symbol_col].tolist()))
        fig = plt.figure(figsize=(8,6))
        for k in range(len(cat)):
            try:
                sns.distplot( data[data[symbol_col]==cat[k]][value_col] ,label=data[data[symbol_col]==cat[k]][symbol_col],kde = True)
            except:
                sns.distplot( data[data[symbol_col]==cat[k]][value_col] ,label=data[data[symbol_col]==cat[k]][symbol_col],kde = False)
        fig.legend(fontsize = 'large',loc='upper right',bbox_to_anchor=(0.84, 0.85))

        # kw-test median

        from scipy import stats
        cat = list(set(data[symbol_col].tolist()))
        sub_list = []
        for i in range(len(cat)):
            sub_list.append(list(data[data[symbol_col]==cat[i]][value_col]))
        
        if len(cat)==2:
            s1 = [i for i in sub_list[0] if not isinstance(i, str)]
            s2 = [i for i in sub_list[1] if not isinstance(i, str)]
            #from scipy.stats import median_test
            #stat, p, med, tbl = median_test(s1,s2,nan_policy='omit')
            kw = stats.kruskal(s1,s2,nan_policy='omit')
            plt.title('p-value for KW-test: %.4f'%kw.pvalue)

        plt.show()



    def get_probplot(self):
        data = self.data
        symbol_col = self.symbol_col
        value_col = self.value_col

        fg = (
                sns.FacetGrid(data=data, hue=symbol_col, aspect=1.5,size = 5,legend_out =True)
                .map(probscale.probplot, value_col,probax='y',plottype='prob',scatter_kws=dict(marker='o', linestyle='-'))
                .set_axis_labels(x_var=value_col, y_var='Probability')
                .set(ylim=(0.001,99.999))
                .add_legend()
            )

        
        #kw-test
        from scipy import stats
        cat = list(set(data[symbol_col].tolist()))
        sub_list = []
        for i in range(len(cat)):
            sub_list.append(list(data[data[symbol_col]==cat[i]][value_col]))
        
        if len(cat) == 2:
            s1 = [i for i in sub_list[0] if not isinstance(i, str)]
            s2 = [i for i in sub_list[1] if not isinstance(i, str)]
            #from scipy.stats import median_test
            #stat, p, med, tbl = median_test(s1,s2,nan_policy='omit')
            kw = stats.kruskal(s1,s2,nan_policy='omit')
            plt.title('p-palue for KW-test: %.4f'%kw.pvalue)

        plt.show()
        return fg



    def get_boxplot(self):
        plt.figure(figsize=(10, 7))
        sns.boxplot(x=self.symbol_col, y=self.value_col, data=self.data)
        plt.show()



    def get_summary(self):
        # summary by each group
        cat = list(set(self.data[self.symbol_col].tolist()))
        summary_df = DataFrame()
        for k in range(len(cat)):
            tmp = self.data[self.data[self.symbol_col]==cat[k]][self.value_col].describe()
            summary_df.insert(k,cat[k],tmp)

        summary_df.insert(k+1,'total',self.data[self.value_col].describe())
        print(summary_df)
        summary_df.to_csv('summary_df.csv',index=True)
        return summary_df



    def get_trend(self,time_col):
        cat = list(set(self.data[self.symbol_col].tolist()))

        fig, ax = plt.subplots(figsize=(13, 6))
        for i in range(len(cat)):
            sub_df = self.data[self.data[self.symbol_col]==cat[i]]
            ax.plot(sub_df[time_col],sub_df[self.value_col],'.-',markersize=12,label=cat[i])

        ax.xaxis.set_major_formatter(DateFormatter("%m/%d %H:%M"))
        ax.xaxis.set_major_locator(HourLocator(byhour=range(0,24,1)))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(rotation=60)
        plt.ylabel(self.value_col)
        plt.xlabel(time_col)
        plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='grapher')
    parser.add_argument('--data_dir', default='./pure_data.csv', help='path to data')
    parser.add_argument('--y_col', default='y',help='column name of predict target')
    parser.add_argument('--cat_col', default=None,help='column name of predict target')
    parser.add_argument('--graph',help='what kind of graph you want? support get_hist/get_probplot/get_boxplot/get_trend')

    args = parser.parse_args()
    
    data = pd.read_csv(args.data_dir)

    if not args.cat_col:
        data['cat_col'] = np.ones(len(data))
        args.cat_col = 'cat_col'

    stat_graph = stat_grapher(data,args.cat_col,args.y_col)
    graph_type = args.graph

    
    if graph_type == 'boxplot': 
        stat_graph.get_boxplot()
    elif graph_type =='hist':
        stat_graph.get_hist()
    elif graph_type =='probplot':
        stat_graph.get_probplot()
    elif graph_type == 'summary':
        stat_graph.get_summary()

    else:
        print('not support this graph type yet.')
    
