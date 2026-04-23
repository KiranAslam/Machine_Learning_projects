import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


class DataVisualizer:
    def __init__(self,df):
        self.df=df
        pio.templates.default = "plotly_dark"

    def plot_univariate_distribution(self,column:str):
   
        if self.df[column].dtype in ['int64', 'float64']:
            fig = px.histogram(
                 self.df, x=column, marginal="box", 
                 title=f"Distribution of {column}",
                 color_discrete_sequence=['#00d4ff']
               )
        else:
            df_counts = self.df[column].value_counts().reset_index()
            df_counts.columns = [column, 'count']
            fig = px.bar(
                df_counts, 
                x=column,     
                y='count',   
                title=f"Frequency of {column}",
                color_discrete_sequence=['#00d4ff']
                )
        return fig

    def plot_bivariate_relationship(self, x_column:str, y_column:str):
        is_x_num = self.df[x_column].dtype in ['int64', 'float64']
        is_y_num = self.df[y_column].dtype in ['int64', 'float64']

        if is_x_num and is_y_num:
            return px.scatter(self.df, x=x_column, y=y_column, trendline="ols", title=f"{y_column} vs {x_column} Correlation",render_mode='webgl')
        elif not is_x_num and is_y_num:
            return px.box(self.df, x=x_column, y=y_column, title=f"{y_column} Distribution by {x_column}")
        else:
            return px.density_heatmap(self.df, x=x_column, y=y_column, title=f"Heatmap of {x_column} and {y_column}",color_continuous_scale='Viridis')

    def plot_correlation_matrix(self , numeric_cols:list):
        corr = self.df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Matrix",color_continuous_scale='RdBu_r')
        return fig
