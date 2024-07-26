from collections import defaultdict
import streamlit as st
import requests

import pandas as pd
import plotly.express as px
from datetime import datetime

@st.cache_data(ttl=600)
def get_all_orders(cookie):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
        "X-Requested-With": "OnlineShopping.WebApp",
        "Cookie": cookie
    }
    url = "https://www.woolworths.co.nz/api/v1/shoppers/my/past-orders"
    response = requests.get(url, headers=headers, timeout=10, data={})
    data = response.json()
    orders = data["items"]
    return orders

@st.cache_data(ttl=600)
def get_one_order_items(cookie, order_id):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
        "X-Requested-With": "OnlineShopping.WebApp",
        "Cookie": cookie
    }
    url = f"https://www.woolworths.co.nz/api/v1/shoppers/my/past-orders/{order_id}/items"
    response = requests.get(url, headers=headers, timeout=10, data={})
    data = response.json()
    order = data["products"]["items"]
    return order

def get_item_price(item):
    if item["price"].get("averagePricePerSingleUnit"):
        return item["price"]["averagePricePerSingleUnit"]
    return item["price"]["salePrice"]

def process_orders(orders, all_order_items):
    processed_data = []
    unique_items = defaultdict(lambda: {"total_cost": 0, "total_quantity": 0})

    for order in orders:
        order_date = datetime.strptime(order["orderDate"], "%Y-%m-%dT%H:%M:%S.%f")
        order_items = all_order_items[order["orderId"]]

        category_costs = defaultdict(float)
        for item in order_items:
            category = item["departments"][0]["name"] if item["departments"] else "Uncategorized"
            quantity = item["quantity"].get("value", 1) or 1  # Use 1 if value is None or 0
            price = get_item_price(item)
            category_costs[category] += quantity * price

            # Aggregate unique items
            item_key = (item["name"], item["sku"])
            unique_items[item_key]["total_cost"] += quantity * price
            unique_items[item_key]["total_quantity"] += quantity
            unique_items[item_key]["unit_price"] = price
            unique_items[item_key]["category"] = category

        for category, cost in category_costs.items():
            processed_data.append({
                "Order ID": order["orderId"],
                "Date": order_date,
                "Total": order["total"],
                "Category": category,
                "Category Cost": cost
            })

    df = pd.DataFrame(processed_data)
    unique_items_df = pd.DataFrame.from_dict(unique_items, orient='index',
                                             columns=["total_cost", "total_quantity", "unit_price", "category"])
    unique_items_df.index.names = ["name", "sku"]
    unique_items_df.reset_index(inplace=True)

    return df, unique_items_df

def main():
    st.title("Woolworths Order Analysis")
    st.write("Welcome to the Woolworths order analyser app!")

    st.text_input("Put your cookie in here 🍪", key="cookie", type="password")
    cookie = st.session_state.cookie

    if cookie:
        with st.spinner("Fetching and processing orders..."):
            orders = get_all_orders(cookie)
            all_order_items = {order["orderId"]: get_one_order_items(cookie, order["orderId"]) for order in orders}
            df, unique_items_df = process_orders(orders, all_order_items)

        st.success("Data processed successfully!")

        # Total spend over time
        st.subheader("Total Spend Over Time")
        fig_total_spend = px.line(df.groupby("Date")["Total"].first().reset_index(),
                                  x="Date", y="Total", title="Total Spend Over Time")
        st.plotly_chart(fig_total_spend)

        # Breakdown of total costs per category
        st.subheader("Total Costs by Category")
        fig_category_total = px.pie(df.groupby("Category")["Category Cost"].sum().reset_index(),
                                    values="Category Cost", names="Category", title="Total Costs by Category")
        st.plotly_chart(fig_category_total)

        # Category costs over time
        st.subheader("Category Costs Over Time")
        fig_category_time = px.line(df.groupby(["Date", "Category"])["Category Cost"].sum().reset_index(),
                                    x="Date", y="Category Cost", color="Category", title="Category Costs Over Time")
        st.plotly_chart(fig_category_time)

        # Top 10 most expensive items
        st.subheader("Top 10 Most Expensive Unique Items")
        top_items = unique_items_df.nlargest(10, "total_cost")
        fig_top_items = px.bar(top_items, x="name", y="total_cost",
                               title="Top 10 Most Expensive Unique Items",
                               hover_data=["total_quantity", "unit_price", "category"])
        fig_top_items.update_xaxes(tickangle=45)
        st.plotly_chart(fig_top_items)

        # Interactive order details
        with st.container(border=True):
            st.subheader("Order Details")
            selected_order = st.selectbox("Select an order", options=df["Order ID"].unique())
            order_details = df[df["Order ID"] == selected_order]
            st.write(f"Order Date: {order_details['Date'].iloc[0]}")
            st.write(f"Total Amount: ${order_details['Total'].iloc[0]:.2f}")
            st.write("Category Breakdown:")
            st.dataframe(order_details[["Category", "Category Cost"]])

        # Spending insights
        st.subheader("Spending Insights")
        avg_total = df.groupby("Date")["Total"].first().mean()
        max_total = df.groupby("Date")["Total"].first().max()
        min_total = df.groupby("Date")["Total"].first().min()
        top_category = df.groupby("Category")["Category Cost"].sum().idxmax()

        st.write(f"Average order total: ${avg_total:.2f}")
        st.write(f"Highest order total: ${max_total:.2f}")
        st.write(f"Lowest order total: ${min_total:.2f}")
        st.write(f"Category with highest total spend: {top_category}")


if __name__ == "__main__":
    main()
