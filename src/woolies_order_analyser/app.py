import base64
import calendar
import hashlib
import json
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import dotenv
import pandas as pd
import plotly.express as px
import pymupdf
import requests
import streamlit as st
from openai import OpenAI
from plotly.graph_objs import Figure

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionContentPartImageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
    )

dotenv.load_dotenv()

# Constants
CACHE_DIR = ".pdf_cache"
Path.mkdir(Path(CACHE_DIR), exist_ok=True, parents=True)
MAX_ALLOWED_DIFFERENCE = 1.50  # Maximum allowed difference for order total validation


# Type aliases
OrderItem = dict[str, Any]


class Order(TypedDict):
    orderId: str
    orderDate: str
    total: str


def get_cache_key(order_id: str, pdf_content: bytes) -> str:
    return hashlib.md5(f"{order_id}:{pdf_content}".encode()).hexdigest()  # noqa: S324


def save_to_disk_cache(key: str, data: Any) -> None:
    with (Path(CACHE_DIR) / f"{key}.pkl").open("wb") as f:
        pickle.dump(data, f)


def load_from_disk_cache(key: str) -> Any:
    try:
        with (Path(CACHE_DIR) / f"{key}.pkl").open("rb") as f:
            return pickle.load(f)  # noqa: S301
    except FileNotFoundError:
        return None


@st.cache_resource(ttl=600)
def get_openai_client() -> OpenAI:
    return OpenAI()


@lru_cache(maxsize=100)
def extract_data_from_pdf_cached(cache_key: str) -> list[OrderItem]:
    return load_from_disk_cache(cache_key)


@st.cache_data(ttl=600)
def get_all_orders(cookie: str) -> list[Order]:
    all_orders = []
    page = 1
    total_items = None

    try:
        while total_items is None or len(all_orders) < total_items:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
                "X-Requested-With": "OnlineShopping.WebApp",
                "Cookie": cookie,
            }
            url = "https://www.woolworths.co.nz/api/v1/shoppers/my/past-orders"
            params = {"dateFilter": "year-2024", "page": page}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if total_items is None:
                total_items = data["totalItems"]

            all_orders.extend(data["items"])
            page += 1
    except requests.RequestException as e:
        st.error(f"Failed to fetch orders: {e!s}")
        return all_orders
    else:
        return all_orders


@st.cache_data(ttl=600)
def get_order_invoice(cookie: str, order_id: str) -> bytes:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
            "X-Requested-With": "OnlineShopping.WebApp",
            "Cookie": cookie,
        }
        url = f"https://www.woolworths.co.nz/api/v1/shoppers/my/past-orders/{order_id}/invoice"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch invoice for order {order_id}: {e!s}")
        return b""
    else:
        return response.content


def image_to_base64(image: bytes) -> str:
    return base64.b64encode(image).decode()


def extract_data_from_pdf_uncached(pdf_content: bytes) -> list[OrderItem]:
    pdf = pymupdf.Document(stream=pdf_content)
    matrix = pymupdf.Matrix(90 / 72, 90 / 72)

    images = [pymupdf.utils.get_pixmap(page, matrix=matrix).tobytes(output="png") for page in pdf]
    images_base64 = [image_to_base64(pix) for pix in images]

    image_messages: list[ChatCompletionContentPartImageParam] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
        }
        for image in images_base64
    ]

    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": """Please extract the items from this invoice. Reply with a list of JSON objects with the following keys:
        - description: the description of the item
        - ordered: the quantity ordered of the item (prefer 'ea' units if multiple units are present)
        - ordered_unit: the unit of the quantity ordered of the item (e.g. kg, ea)
        - supplied: the quantity supplied of the item (prefer 'ea' units if multiple units are present)
        - supplied_unit: the unit of the quantity supplied of the item (e.g. kg, ea)
        - unit_price: the unit price of the item (prefer 'ea' units if multiple units are present)
        - unit_price_unit: the unit of the unit price of the item (e.g. kg, ea)
        - amount: the total cost of the item
        - category: the category of the item

        For example:
        {
            'items': [
                {
                    "description": "Gopala paneer cheese 300g block",
                    "ordered": 1,
                    "ordered_unit": "ea",
                    "supplied": 1,
                    "supplied_unit": "ea",
                    "unit_price": 6.4,
                    "unit_price_unit": "ea",
                    "amount": 6.4,
                    "category": "Deli & Chilled Foods"
                },
                {
                    "description": "Fresh fruit bananas yellow per kg loose",
                    "ordered": 5,
                    "ordered_unit": "ea",
                    "supplied": 5,
                    "supplied_unit": "ea",
                    "unit_price": 3.7,
                    "unit_price_unit": "kg",
                    "amount": 3.95,
                    "category": "Food"
                }
            ]
        }
        """,  # noqa: E501
    }

    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here are the images of the invoice:"},
            *image_messages,
        ],
    }

    client = get_openai_client()

    completion: ChatCompletion = client.chat.completions.create(
        messages=[system_message, user_message],
        model="gpt-4o",
        response_format={"type": "json_object"},
        max_tokens=4000,
    )

    if completion.choices and completion.choices[0].message.content:
        content = completion.choices[0].message.content
        try:
            return cast(list[OrderItem], json.loads(content)["items"])
        except (json.JSONDecodeError, KeyError):
            msg = "Failed to parse the API response"
            raise ValueError(msg) from None
    else:
        msg = "No content in the API response"
        raise ValueError(msg)


def process_single_order(order: Order, invoice_content: bytes) -> list[OrderItem]:
    cache_key = get_cache_key(order["orderId"], invoice_content)
    invoice_items = extract_data_from_pdf_cached(cache_key)

    if invoice_items is None:
        invoice_items = extract_data_from_pdf_uncached(invoice_content)
        save_to_disk_cache(cache_key, invoice_items)

    return [
        {
            "order_id": order["orderId"],
            "date": order["orderDate"],
            "category": item["category"],
            "name": item["description"],
            "total_price": float(item["amount"]) if item["amount"] else 0.0,
            "quantity": float(item["supplied"]) if item["supplied"] else 0.0,
            "unit_price": float(item["unit_price"]) if item["unit_price"] else 0.0,
        }
        for item in invoice_items
    ]


def validate_order_total(order: Order, processed_items: list[OrderItem]) -> tuple[bool, float, float]:
    """Validate the total price of an order.

    Comparing the sum of line items with the order total from get_all_orders, allowing for a difference of up to $1.50
    to account for bag charges.
    """
    calculated_total = sum(item["total_price"] for item in processed_items)
    original_total = float(order["total"])
    difference = abs(calculated_total - original_total)
    is_valid = difference <= MAX_ALLOWED_DIFFERENCE
    return is_valid, calculated_total, original_total


def process_orders(
    orders: list[Order],
    all_order_invoices: dict[str, bytes],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    all_items = []
    unique_items = defaultdict(lambda: {"total_cost": 0, "total_quantity": 0})
    validation_results = []

    progress_bar = st.progress(0)
    status_message = st.empty()

    total_orders = len(orders)

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_order = {
            executor.submit(process_single_order, order, all_order_invoices[order["orderId"]]): order
            for order in orders
        }

        for i, future in enumerate(as_completed(future_to_order)):
            order = future_to_order[future]
            order_items = future.result()

            # Validate order total
            is_valid, calculated_total, original_total = validate_order_total(order, order_items)
            validation_results.append(
                {
                    "order_id": order["orderId"],
                    "is_valid": is_valid,
                    "calculated_total": calculated_total,
                    "original_total": original_total,
                },
            )

            all_items.extend(order_items)

            for item in order_items:
                item_key = item["name"]
                unique_items[item_key]["total_cost"] += item["total_price"]
                unique_items[item_key]["total_quantity"] += item["quantity"]
                unique_items[item_key]["unit_price"] = item["unit_price"]
                unique_items[item_key]["category"] = item["category"]

            progress = (i + 1) / total_orders
            progress_bar.progress(progress)
            status_message.text(f"Processed {i + 1} out of {total_orders} invoices")

    progress_bar.empty()
    status_message.empty()

    df = pd.DataFrame(all_items)
    df["date"] = pd.to_datetime(df["date"])

    unique_items_df = pd.DataFrame.from_dict(
        unique_items,
        orient="index",
        columns=["total_cost", "total_quantity", "unit_price", "category"],
    )
    unique_items_df.index.name = "name"
    unique_items_df.reset_index(inplace=True)

    return df, unique_items_df, validation_results


def create_total_spend_chart(df: pd.DataFrame) -> Figure:
    total_spend_df = df.groupby("date")["total_price"].sum().reset_index()
    total_spend_df["moving_average"] = total_spend_df["total_price"].rolling(window=3).mean()
    fig = px.line(
        total_spend_df,
        x="date",
        y=["total_price", "moving_average"],
        title="Total Spend Over Time with 3-Order Moving Average",
        labels={"value": "Amount ($)", "variable": "Metric"},
        hover_data={"date": "|%B %d, %Y"},
    )
    fig.update_traces(name="Total Spend", selector={"name": "total_price"})
    fig.update_traces(name="3-Order Moving Average", selector={"name": "moving_average"})
    return fig


def create_monthly_spend_heatmap(df: pd.DataFrame) -> Figure:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly_spend = df.groupby(["year", "month"])["total_price"].sum().reset_index()

    year_range = range(monthly_spend["year"].min(), monthly_spend["year"].max() + 1)
    month_range = range(1, 13)
    complete_date_range = pd.MultiIndex.from_product([year_range, month_range], names=["year", "month"])

    monthly_spend = monthly_spend.set_index(["year", "month"]).reindex(complete_date_range).fillna(0).reset_index()

    monthly_spend["month_name"] = monthly_spend["month"].apply(lambda x: calendar.month_abbr[x])
    pivot_data = monthly_spend.pivot(index="year", columns="month", values="total_price")

    fig = px.imshow(
        pivot_data,
        labels={"x": "Month", "y": "Year", "color": "Spend"},
        x=[calendar.month_abbr[i] for i in range(1, 13)],
        y=pivot_data.index,
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(title="Monthly Spend Heatmap")
    return fig


def create_category_costs_chart(df: pd.DataFrame) -> Figure:
    category_time_df = df.groupby(["date", "category"])["total_price"].sum().reset_index()
    fig = px.area(
        category_time_df,
        x="date",
        y="total_price",
        color="category",
        title="Category Costs Over Time (Stacked)",
        labels={"total_price": "Amount ($)", "date": "Date"},
        hover_data={"date": "|%B %d, %Y"},
    )
    return fig


def create_top_items_chart(unique_items_df: pd.DataFrame) -> Figure:
    top_items = unique_items_df.nlargest(10, "total_cost")
    fig = px.bar(
        top_items,
        x="total_cost",
        y="name",
        orientation="h",
        title="Top 10 Most Expensive Unique Items",
        labels={"total_cost": "Total Spend ($)", "name": "Item"},
        hover_data=["total_quantity", "unit_price", "category"],
        color="category",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def create_category_distribution_chart(df: pd.DataFrame) -> Figure:
    category_total_df = df.groupby("category")["total_price"].sum().reset_index()
    fig = px.treemap(
        category_total_df,
        path=["category"],
        values="total_price",
        title="Spending Distribution by Category",
        hover_data=["total_price"],
    )
    return fig


def create_item_frequency_chart(df: pd.DataFrame) -> Figure:
    item_frequency = df.groupby("name")["order_id"].nunique().sort_values(ascending=False).head(10)
    fig = px.bar(
        x=item_frequency.index,
        y=item_frequency.values,
        labels={"x": "Item", "y": "Number of Orders"},
        title="Top 10 Most Frequently Purchased Items",
    )
    return fig


def create_spending_trend_chart(df: pd.DataFrame) -> Figure:
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_trend = df.groupby("year_month")["total_price"].sum().reset_index()
    monthly_trend["year_month"] = monthly_trend["year_month"].astype(str)
    fig = px.line(
        monthly_trend,
        x="year_month",
        y="total_price",
        title="Monthly Spending Trend",
        labels={"year_month": "Month", "total_price": "Total Spend ($)"},
    )
    fig.update_xaxes(tickangle=45)
    return fig


def create_order_breakdown_chart(order_details: pd.DataFrame) -> Figure:
    category_breakdown = order_details.groupby("category")["total_price"].sum().reset_index()
    fig = px.pie(
        category_breakdown,
        values="total_price",
        names="category",
        title=f"Category Breakdown for Order {order_details['order_id'].iloc[0]}",
    )
    return fig


def display_validation_results(validation_results: list[dict[str, Any]]):
    with st.expander("View Validation Results"):
        valid_orders = sum(1 for result in validation_results if result["is_valid"])
        invalid_orders = len(validation_results) - valid_orders

        st.write(f"Total orders processed: {len(validation_results)}")
        st.write(f"Valid order totals: {valid_orders}")
        st.write(f"Invalid order totals: {invalid_orders}")

        if invalid_orders > 0:
            st.warning("Some order totals do not match.")
            for result in validation_results:
                if not result["is_valid"]:
                    st.write(f"Order ID: {result['order_id']}")
                    st.write(f"Calculated Total: ${result['calculated_total']:.2f}")
                    st.write(f"Original Total: ${result['original_total']:.2f}")
                    st.write(f"Difference: ${abs(result['calculated_total'] - result['original_total']):.2f}")
                    st.write("---")


def display_order_details(df: pd.DataFrame):
    st.subheader("Order Details")

    # Ensure order_id is integer type
    df["order_id"] = df["order_id"].astype(int)

    # Create a new column that combines order_id and date, sorted descending by order_id
    df_sorted = df.sort_values("order_id", ascending=False)
    df_sorted["order_id_date"] = df_sorted.apply(
        lambda row: f"{row['order_id']} - {row['date'].strftime('%Y-%m-%d %H:%M')}",
        axis=1,
    )

    # Get unique order_id_date combinations
    unique_orders = df_sorted["order_id_date"].unique()

    # Create the selectbox with the combined order_id and date
    selected_order_id_date = st.selectbox("Select an order", options=unique_orders, index=None)

    if selected_order_id_date is not None:
        # Extract the order_id from the selected value and convert to integer
        selected_order_id = int(selected_order_id_date.split(" - ")[0])

        # Filter the DataFrame based on the selected order_id
        order_details = df[df["order_id"] == selected_order_id]

        if not order_details.empty:
            st.write(f"Order Date: {order_details['date'].iloc[0]}")
            st.write(f"Total Amount: ${order_details['total_price'].sum():.2f}")

            chart_col, df_col = st.columns(2)

            with chart_col:
                st.plotly_chart(create_order_breakdown_chart(order_details))
            with df_col:
                st.dataframe(
                    order_details[["name", "quantity", "unit_price", "total_price", "category"]],
                    hide_index=True,
                )
        else:
            st.warning(f"No details found for order {selected_order_id}")
    else:
        st.info("Please select an order to view its details.")


def load_and_process_data(
    selected_orders: list[Order],
    all_order_invoices: dict[str, bytes],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    st.info("Processing selected orders and invoices...")
    df, unique_items_df, validation_results = process_orders(selected_orders, all_order_invoices)
    return df, unique_items_df, validation_results


def main():  # noqa: PLR0915
    st.set_page_config(page_title="Woolies Order Analyser", layout="wide", page_icon="🛒")
    st.title("🛒 Woolies Order Analyser")
    st.write("Welcome to the Woolies Order Analyser!")

    cookie = st.text_input(
        "Put your cookie in here 🍪",
        help="Get this from the headers of a network request in your browser dev tools, after you have signed in to the Woolies website",  # noqa: E501
        key="cookie",
        type="password",
    )

    if not cookie:
        return

    if "data_processed" not in st.session_state:
        st.session_state.data_processed = False

    if not st.session_state.data_processed:
        with st.spinner("Fetching orders..."):
            all_orders = get_all_orders(cookie)
            if not all_orders:
                st.error("No orders found. Please check your cookie and try again.")
                return
            st.success(f"Found {len(all_orders)} orders.")

        order_df = pd.DataFrame(all_orders)
        order_df["orderDate"] = pd.to_datetime(order_df["orderDate"])
        order_df = order_df.sort_values("orderDate", ascending=False)
        order_df["formatted_date"] = order_df["orderDate"].dt.strftime("%Y-%m-%d %H:%M")
        order_df["select"] = True

        st.subheader("Select orders to analyse:")
        selected_df = st.data_editor(
            order_df[["select", "orderId", "formatted_date", "total"]],
            column_config={
                "select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select orders for analysis",
                    default=True,
                    width="small",
                ),
                "orderId": st.column_config.TextColumn(
                    "Order ID",
                    help="Unique identifier for the order",
                ),
                "formatted_date": st.column_config.TextColumn(
                    "Order Date",
                    help="Date and time of the order",
                ),
                "total": st.column_config.NumberColumn(
                    "Total ($)",
                    help="Total amount of the order",
                    format="$%.2f",
                ),
            },
            disabled=["orderId", "formatted_date", "total"],
            hide_index=True,
            use_container_width=True,
        )

        selected_orders = order_df[selected_df["select"]].to_dict("records")

        if not selected_orders:
            st.warning("No orders selected. Please select at least one order to proceed.")
            return

        if st.button(
            f"Proceed with {len(selected_orders)} selected orders",
            help="When you click this, we'll retrieve details of all the selected orders and extract data from their invoices. This can take a few minutes.",  # noqa: E501
        ):
            st.info(f"Fetching invoices for {len(selected_orders)} selected orders...")
            invoice_progress = st.progress(0)
            invoice_status = st.empty()
            all_order_invoices = {}
            for i, order in enumerate(selected_orders):
                all_order_invoices[order["orderId"]] = get_order_invoice(cookie, order["orderId"])
                progress = (i + 1) / len(selected_orders)
                invoice_progress.progress(progress)
                invoice_status.text(f"Fetched {i + 1} out of {len(selected_orders)} invoices")

            invoice_progress.empty()
            invoice_status.empty()
            st.success("All selected invoices fetched successfully.")

            typed_selected_orders = cast(list[Order], selected_orders)
            df, unique_items_df, validation_results = load_and_process_data(typed_selected_orders, all_order_invoices)

            st.session_state.df = df
            st.session_state.unique_items_df = unique_items_df
            st.session_state.validation_results = validation_results
            st.session_state.data_processed = True

            st.success("Data processed successfully!")
            st.rerun()

    if st.session_state.data_processed:
        df = st.session_state.df
        unique_items_df = st.session_state.unique_items_df
        validation_results = st.session_state.validation_results

        # Display validation results
        display_validation_results(validation_results)

        # Create two columns for the first row of charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Total Spend Over Time")
            st.plotly_chart(create_total_spend_chart(df), use_container_width=True)

        with col2:
            st.subheader("Monthly Spend Heatmap")
            st.plotly_chart(create_monthly_spend_heatmap(df), use_container_width=True)

        # Create two columns for the second row of charts
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Category Costs Over Time")
            st.plotly_chart(create_category_costs_chart(df), use_container_width=True)

        with col4:
            st.subheader("Top 10 Most Expensive Unique Items")
            st.plotly_chart(create_top_items_chart(unique_items_df), use_container_width=True)

        # Create two columns for the third row of charts
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Spending Distribution by Category")
            st.plotly_chart(create_category_distribution_chart(df), use_container_width=True)

        with col6:
            st.subheader("Most Frequently Purchased Items")
            st.plotly_chart(create_item_frequency_chart(df), use_container_width=True)

        col_insights, col_trend = st.columns(2)

        with col_insights:
            st.subheader("Spending Insights")
            order_totals = df.groupby("order_id")["total_price"].sum()
            avg_total = order_totals.mean()
            max_total = order_totals.max()
            min_total = order_totals.min()
            top_category = df.groupby("category")["total_price"].sum().idxmax()
            total_spend = df["total_price"].sum()

            # Create three columns for metrics within the Spending Insights column
            col7, col8, col9 = st.columns(3)
            with col7:
                st.metric("Total Spend", f"${total_spend:.2f}")
                st.metric("Average Order Total", f"${avg_total:.2f}")
            with col8:
                st.metric("Highest Order Total", f"${max_total:.2f}")
                st.metric("Lowest Order Total", f"${min_total:.2f}")
            with col9:
                st.metric("Top Spending Category", top_category)
                st.metric("Number of Orders", len(df["order_id"].unique()))

        with col_trend:
            st.subheader("Spending Trend Analysis")
            st.plotly_chart(create_spending_trend_chart(df), use_container_width=True)

        with st.container(border=True):
            display_order_details(df)


if __name__ == "__main__":
    main()
