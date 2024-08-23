# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
!pip install pyarrow

# %%
all_data = pd.read_feather(r"D:\e-Commerce_Project\Sales_data.ftr")

# %%
all_data.head()

# %%
all_data.isnull().sum()

# %%
all_data = all_data.dropna(how= "all")

# %%
all_data.isnull().sum()

# %%
all_data.duplicated()

# %%
duplicate_rows = all_data[all_data.duplicated()]
duplicate_rows


# %%
all_data = all_data.drop_duplicates()

# %%
all_data.head()

# %%
all_data['Order Date'][0]

# %%
'04/19/19 08:46'.split(' ')[0]

# %%
'04/19/19 08:46'.split(' ')[0].split('/')[0]

# %%
def return_month(x):
    return x.split('/')[0]

# %%
all_data['Order Date'].apply(return_month)

# %%
all_data['Order Date'].apply(return_month).unique()

# %%
all_data['Month'] = all_data['Order Date'].apply(return_month)

# %%
filter1 = all_data['Month'] == 'Order Date'

# %%
all_data[filter1]

# %%
all_data = all_data[~filter1]

# %%
import calendar

# Define the function to convert month number (as a string or integer) to month name
def month_number_to_name(month):
    month = int(month)  # Convert the month to an integer
    return calendar.month_name[month]  # Use the integer to get the month name

# Apply the function to the 'Month' column
all_data['Month Name'] = all_data['Month'].apply(month_number_to_name)


# %%
all_data['Quantity Ordered'] = all_data['Quantity Ordered'].astype(int)
all_data['Price Each'] = all_data['Price Each'].astype(float)  #Price could be 10.99 so use float

# %%
all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']

# %%
import matplotlib.pyplot as plt


monthly_sales = all_data.groupby('Month Name')['Sales'].sum()  #Grouped the data by month names and used summation for sales


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_sales = monthly_sales.reindex(month_order)


colors = plt.cm.viridis(np.linspace(0, 1, len(monthly_sales)))
plt.figure(figsize=(10, 6))
plt.bar(monthly_sales.index, monthly_sales.values, color=colors)


plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Sales in USD')


plt.xticks(rotation=45)
plt.show()


# %%
#Inference From the above graph December has the maximum sales, it could be due to Christmas season

# %%
#Now seeing which city has the maximum number of orders

# %%
all_data['Purchase Address'].str.split(',').str.get(1)

# %%
all_data['City'] = all_data['Purchase Address'].str.split(',').str.get(1)

# %%
all_data.head()

# %%
all_data['City'].value_counts()

# %%
city_order_counts = all_data['City'].value_counts()


# %%
plt.figure(figsize=(10, 8))
plt.pie(city_order_counts, labels=city_order_counts.index, autopct='%1.0f%%', startangle=140) #autopct used to show percentage
plt.title('Distribution of Orders by City')
plt.axis('equal')
plt.show()


# %%
#Now seeing what product sold the most

# %%
all_data.columns

# %%
all_data.groupby(['Product']).agg({'Quantity Ordered':'sum', 'Price Each':'mean'})

# %%
count_df = all_data.groupby(['Product']).agg({'Quantity Ordered':'sum', 'Price Each':'mean'}).reset_index()

# %%
count_df

# %%

sns.set(style="whitegrid")


products = count_df['Product'].values


fig, ax1 = plt.subplots(figsize=(12, 6)) 

ax2 = ax1.twinx()


ax1.bar(count_df['Product'], count_df['Quantity Ordered'], color='skyblue', alpha=0.7, label='Order Count')
ax2.plot(count_df['Product'], count_df['Price Each'], color='orange', marker='o', label='Avg Price')


ax1.set_xticklabels(products, rotation= 'vertical', fontsize=10)

ax1.set_xlabel('Product')
ax1.set_ylabel('Order Count', color='skyblue')
ax2.set_ylabel('Avg Price of Product', color='orange')


ax1.grid(True, which='major', axis='y', linestyle='--')


plt.title('Product Order Count and Average Price Comparison')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')



plt.tight_layout()  
plt.show()


# %%
#Cheaper The price higher the quantity ordered

# %%
#We can build pivot table with crosstab

all_data['Product'].value_counts()[0:5].index

# %%
most_sold_product = all_data['Product'].value_counts()[0:5].index

# %%
all_data['Product'].isin(most_sold_product)

# %%
all_data[all_data['Product'].isin(most_sold_product)]

# %%
most_sold_product_df = all_data[all_data['Product'].isin(most_sold_product)]

# %%
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
most_sold_product_df['Month Name'] = pd.Categorical(most_sold_product_df['Month Name'], 
                                                    categories=month_order, 
                                                    ordered=True)
most_sold_product_df = most_sold_product_df.sort_values('Month Name')


# %%
pivot_table = most_sold_product_df.groupby(['Month Name', 'Product']).size().unstack()

# %%
plt.figure(figsize= (16,8))
pivot_table.plot(kind='line', marker='s', ax=plt.gca())
plt.title('Product Sales by Month', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Products Sold', fontsize=12)


# %%
#What products are most sold together?
#This can be used to make a recommendation system

# %%
all_data['Order ID']

# %%
# We will keep all duplicate Order IDs
df_duplicated = all_data[all_data['Order ID'].duplicated(keep= False)] #marks all occurence of duplicates as true

# %%
df_duplicated

# %%
dup_products = df_duplicated.groupby(['Order ID'])['Product'].apply(lambda x: ','.join(x)).reset_index().rename(columns ={'Product':'Grouped_Products'})

# %%
dup_products['Grouped_Products'].value_counts()[0:5]

# %%
grouped_products_count = dup_products['Grouped_Products'].value_counts()[0:5]


# %%
plt.figure(figsize=(10, 8))
plt.pie(grouped_products_count, labels=grouped_products_count.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Grouped Products')


# %%
#At what time ads should be displayed

all_data.head()

# %%
all_data['Time'] = all_data['Order Date'].str.split(' ').str.get(1)

# %%
all_data['Hour'] = all_data['Time'].str.split(':').str.get(0)

# %%
all_data.head()

# %%
Hourly_Purchases = all_data.groupby('Hour')['Product'].count()
plt.figure(figsize=(10, 6))
Hourly_Purchases.plot(kind='bar', color='skyblue')


plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Number of Products Sold', fontsize=12)
plt.title('Number of Products Sold by Hour', fontsize=16)




# %%



