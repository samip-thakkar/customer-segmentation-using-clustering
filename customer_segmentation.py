# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.io as pio
pio.templates.default = "none"
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans

#Get the data
df = pd.read_csv('Orders - Analysis Task.txt')

#Check for null values
print(df.info())

#Check the descriptive statistics of the data
print(df.describe())

#Exclude the data where the order was not sold/ returned
df = df[df['ordered_item_quantity'] > 0]

"""Check for the products ordered"""
def encode_column(column):
    if column > 0:
        return 1
    else:
        return 0

#Create function to identify customers with multiple orders
"""Steps: 1. aggregates a given dataframe by column list, 
    as a result creates a aggregated dataframe by counting the ordered item quantities

    2. adds number_of_X ordered where X is the second element in the column_list 
    to the aggregated dataframe by encoding ordered items into 1

    3. creates final dataframe containing information about 
    how many of X are ordered, based on the first element passed in the column list"""
    
def aggregate_by_ordered_quantity(df, column_list):
    aggregated_dataframe = df.groupby(column_list).ordered_item_quantity.count().reset_index()
    aggregated_dataframe['products_ordered'] = aggregated_dataframe.ordered_item_quantity.apply(encode_column)
    final_dataframe = aggregated_dataframe.groupby(column_list[0]).products_ordered.sum().reset_index()
    return final_dataframe

customers = aggregate_by_ordered_quantity(df, ['customer_id', 'product_type'])


""" Average Return Rate (ratio of returned item quantity and ordered item quantity)."""
ordered_sum_by_customer_order = (df.groupby(["customer_id", "order_id"]).ordered_item_quantity.sum().reset_index())
returned_sum_by_customer_order = (df.groupby(["customer_id", "order_id"]).returned_item_quantity.sum().reset_index())

#Merge two dataframes to be able to calculate unit return rate
ordered_returned_sums = pd.merge(ordered_sum_by_customer_order, returned_sum_by_customer_order)
ordered_returned_sums["average_return_rate"] = (-1 * ordered_returned_sums["returned_item_quantity"] / ordered_returned_sums["ordered_item_quantity"])

#Take average of the unit return rate for all orders of a customer
customer_return_rate = (ordered_returned_sums.groupby("customer_id").average_return_rate.mean().reset_index())
return_rates = pd.DataFrame(customer_return_rate["average_return_rate"].value_counts().reset_index())
return_rates.rename(columns= {"index": "average return rate","average_return_rate": "count of unit return rate"},inplace=True)
return_rates.sort_values(by="average return rate")

#Add average_return_rate to customers dataframe
customers = pd.merge(customers,customer_return_rate,on="customer_id")

"""Total spending sum of total sales value which is the amount after the taxes and returns"""
customer_total_spending = df.groupby('customer_id').total_sales.sum().reset_index()
customer_total_spending.rename(columns = {'total_sales' : 'total_spending'}, inplace = True)

#Add total sales to customer datafeame
customers = customers.merge(customer_total_spending, on = 'customer_id')
print("The number of customers from the existing customer base:", customers.shape[0])


"""Feature Modeling"""
#Drop id column as is not a feature
customers = customers.drop(columns = 'customer_id', axis = 1)


#Visualize features
fig = make_subplots(rows=3, cols=1, subplot_titles=("Products Ordered", "Average Return Rate", "Total Spending"))
fig.append_trace(go.Histogram(x=customers.products_ordered), row=1, col=1)
fig.append_trace(go.Histogram(x=customers.average_return_rate), row=2, col=1)
fig.append_trace(go.Histogram(x=customers.total_spending), row=3, col=1)
fig.update_layout(height=800, width=800, title_text="Distribution of the Features")
plot(fig)


"""Features Scaling"""
#Convert the values of the dataframe to log form
def apply_log1p_transformation(df, column):
    df["log_" + column] = np.log1p(df[column])
    return df["log_" + column]

#Apply on products ordered
apply_log1p_transformation(customers, "products_ordered")
#Appylt on average return
apply_log1p_transformation(customers, "average_return_rate")
#Apply on total spending
apply_log1p_transformation(customers, "total_spending")

#Visualize log transformation applied features¶
fig = make_subplots(rows=3, cols=1,
                   subplot_titles=("Products Ordered", "Average Return Rate", "Total Spending"))
fig.append_trace(go.Histogram(x=customers.log_products_ordered), row=1, col=1)
fig.append_trace(go.Histogram(x=customers.log_average_return_rate), row=2, col=1)
fig.append_trace(go.Histogram(x=customers.log_total_spending), row=3, col=1)

fig.update_layout(height=800, width=800,
                  title_text="Distribution of the Features after Logarithm Transformation")

plot(fig)

customers.iloc[:, 3:]

"""Create K-means Model"""
model = KMeans(init='k-means++', max_iter=500, random_state=42)
model.fit(customers.iloc[:,3:])
# print the sum of distances from all examples to the center of the cluster
print("within-cluster sum-of-squares (inertia) of the model is:", model.inertia_)

#Hyperparameter tuning for k
def make_list_of_k(k, df):
    cluster_values = list(range(1, k + 1))
    inertia_values = []
    
    for c in cluster_values: 
        model = KMeans(n_clusters = c, init = 'k-means++', max_iter = 500, random_state = 42)
        model.fit(df)
        inertia_values.append(model.inertia_)
    
    return inertia_values

#Visualize the results for different k
results = make_list_of_k(15, customers.iloc[:, 3:])

k_values_distances = pd.DataFrame({"clusters": list(range(1, 16)), "within cluster sum of squared distances": results})

# visualization for the selection of number of segments
fig = go.Figure()
fig.add_trace(go.Scatter(x=k_values_distances["clusters"], y=k_values_distances["within cluster sum of squared distances"], mode='lines+markers'))
fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 1, dtick = 1), title_text="Within Cluster Sum of Squared Distances VS K Values", xaxis_title="K values", yaxis_title="Cluster sum of squared distances")
plot(fig)

#We will take k = 4
model = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 500, random_state = 42)
model.fit_predict(customers.iloc[:, 3:])

#Visualization of Kmeans
cluster_centers = model.cluster_centers_
actual_data = np.expm1(cluster_centers)
add_points = np.append(actual_data, cluster_centers, axis=1)

# add labels to customers dataframe and add_points array
add_points = np.append(add_points, [[0], [1], [2], [3]], axis=1)
customers["clusters"] = model.labels_

centers_df = pd.DataFrame(data=add_points, columns=["products_ordered", "average_return_rate", "total_spending", "log_products_ordered", "log_average_return_rate", "log_total_spending", "clusters"])

#Convert the label as integer
centers_df['clusters'] = centers_df['clusters'].astype('int')

# differentiate between data points and cluster centers
customers["is_center"] = 0
centers_df["is_center"] = 1
# add dataframes together
customers = customers.append(centers_df, ignore_index=True)

"""Visualize the customer segmentation"""

# add clusters to the dataframe
customers["cluster_name"] = customers["clusters"].astype(str)
# visualize log_transformation customer segments with a 3D plot
fig = px.scatter_3d(customers, x="log_products_ordered",
y="log_average_return_rate", z="log_total_spending",
color='cluster_name', hover_data=["products_ordered", "average_return_rate", "total_spending"], category_orders = {"cluster_name": ["0", "1", "2", "3"]},
symbol = "is_center")

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plot(fig)

"""Check the clusters result"""
# values for log_transformation
cardinality_df = pd.DataFrame(customers.cluster_name.value_counts().reset_index())

cardinality_df.rename(columns={"index": "Customer Groups", "cluster_name": "Customer Group Magnitude"}, inplace=True)

fig = px.bar(cardinality_df, x="Customer Groups", y="Customer Group Magnitude", color = "Customer Groups", category_orders = {"Customer Groups": ["0", "1", "2", "3"]})

fig.update_layout(xaxis = dict( tickmode = 'linear', tick0 = 1, dtick = 1), yaxis = dict( tickmode = 'linear',tick0 = 1000, dtick = 1000))
plot(fig)