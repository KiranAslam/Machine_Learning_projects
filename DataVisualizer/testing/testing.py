import numpy as np
import pandas as pd
import plotly.express as px


df = pd.read_csv("Sample - Superstore.csv", encoding="cp1252")
#print(df.describe())
print(df.head(10))
#print(df.info())
#print(df.isnull())

fig1 = px.histogram(df, x ="Sales",log_y=True,marginal="box",title="Sales Distribution with Outliers")
fig1.show()
fig2 = px.pie(df, names='Category', title="Good Pie Chart: Low Cardinality")
fig2.show()
fig3 = px.pie(df, names='Sub-Category', title="Bad Pie Chart: High Cardinality Trap")
fig3.show()
df["Order Date"] = pd.to_datetime(df['Order Date'])
monthly_data = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
monthly_data["Order Date"] = monthly_data["Order Date"].dt.to_timestamp()
fig4 = px.line(monthly_data, x = "Order Date", y="Sales",title="Monthly Sales Trend", markers=True)
fig4.show()

fig5= px.scatter(df,x="Sales",y="Profit",color="Category", opacity = 0.3,title="Sales vs Profit Correlation")
fig5.show()
fig6 = px.treemap(df,path =['Region','Category','Sub-Category'], values='Sales',title="Regional Sales Hierarchy Breakdown")
fig6.show()