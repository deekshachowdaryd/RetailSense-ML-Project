
import streamlit as st
import pandas as pd
import joblib
import json
import os

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="RetailSense",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------------
# LOAD PROJECT FILES
# -----------------------------------

@st.cache_resource
def load_models():
    svm = joblib.load("retailsense_assets/cancellation_svm.pkl")
    kmeans = joblib.load("retailsense_assets/kmeans_model.pkl")
    scaler = joblib.load("retailsense_assets/kmeans_scaler.pkl")

    with open("retailsense_assets/metadata.json") as f:
        metadata = json.load(f)

    return svm, kmeans, scaler, metadata


@st.cache_data
def load_data():
    folder = "retailsense_assets"

    invoice = pd.read_csv(f"{folder}/invoice_data.csv")
    customers = pd.read_csv(f"{folder}/customer_features.csv")
    clusters = pd.read_csv(f"{folder}/cluster_summary.csv", index_col=0)
    countries = pd.read_csv(f"{folder}/country_probability.csv")
    hours = pd.read_csv(f"{folder}/hour_probability.csv")
    days = pd.read_csv(f"{folder}/day_probability.csv")

    return invoice, customers, clusters, countries, hours, days


try:
    svm_model, kmeans_model, kmeans_scaler, metadata = load_models()

    (
        invoice_data,
        customer_features,
        cluster_summary,
        country_probability,
        hour_probability,
        day_probability
    ) = load_data()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# -----------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------

st.sidebar.title("RetailSense")
st.sidebar.caption("Retail Transaction Analytics")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "SVM Prediction",
        "Probability Analysis",
        "Customer Segmentation"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("Online Retail Dataset")


# -----------------------------------
# PAGE 1: DASHBOARD
# -----------------------------------

if page == "Dashboard":

    st.title("Retail Analytics Dashboard")
    st.write(
        "Explore transaction behaviour, cancellation patterns "
        "and customer purchasing trends."
    )

    total = len(invoice_data)
    cancelled = int(invoice_data["Cancelled"].sum())
    normal = total - cancelled
    cancellation_rate = cancelled / total * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Invoices", f"{total:,}")
    col2.metric("Cancelled Invoices", f"{cancelled:,}")
    col3.metric("Completed Invoices", f"{normal:,}")
    col4.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Invoice Distribution")

        invoice_chart = pd.DataFrame({
            "Invoice Type": ["Non-cancelled", "Cancelled"],
            "Count": [normal, cancelled]
        })

        st.bar_chart(
            invoice_chart.set_index("Invoice Type")
        )

    with right:
        st.subheader("Cancellation by Country")

        top_countries = country_probability[
            country_probability["Total_Invoices"] >= 100
        ].sort_values(
            "Cancellation_Percentage",
            ascending=False
        ).head(10)

        st.bar_chart(
            top_countries.set_index("Country")[
                "Cancellation_Percentage"
            ]
        )

    st.divider()

    st.subheader("Cancellation by Transaction Hour")

    st.line_chart(
        hour_probability.set_index("Hour")[
            "Cancellation_Percentage"
        ]
    )

    st.caption(
        "All dashboard statistics describe historical records "
        "in the Online Retail dataset."
    )


# -----------------------------------
# PAGE 2: SVM PREDICTION
# -----------------------------------

elif page == "SVM Prediction":

    st.title("Transaction Cancellation Classification")

    st.write(
        "Enter invoice characteristics to classify an invoice "
        "using our trained Support Vector Machine model."
    )

    st.info(
        "This model classifies invoice characteristics using "
        "patterns learned from historical records. It does not "
        "establish whether a new purchase will be cancelled in the future."
    )

    st.subheader("Transaction Information")

    countries = sorted(invoice_data["Country"].dropna().unique())

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            line_count = st.number_input(
                "Number of product entries",
                min_value=1,
                value=3
            )

            unique_products = st.number_input(
                "Number of unique products",
                min_value=1,
                value=3
            )

            total_quantity = st.number_input(
                "Total quantity",
                min_value=1,
                value=10
            )

            average_price = st.number_input(
                "Average unit price",
                min_value=0.01,
                value=5.0,
                step=0.5
            )

            total_value = st.number_input(
                "Total invoice value",
                min_value=0.01,
                value=50.0,
                step=1.0
            )

        with col2:
            country = st.selectbox(
                "Country",
                countries
            )

            hour = st.slider(
                "Transaction hour",
                min_value=0,
                max_value=23,
                value=12
            )

            day = st.selectbox(
                "Day of the week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ][x]
            )

            month = st.selectbox(
                "Month",
                options=list(range(1, 13))
            )

        submitted = st.form_submit_button(
            "Classify Invoice",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "LineCount": line_count,
            "UniqueProducts": unique_products,
            "TotalQuantity": total_quantity,
            "AveragePrice": average_price,
            "TotalValue": total_value,
            "Country": country,
            "Hour": hour,
            "DayOfWeek": day,
            "Month": month
        }])

        # Match the exact training feature order
        input_data = input_data[metadata["svm_features"]]

        prediction = svm_model.predict(input_data)[0]

        st.divider()
        st.subheader("Classification Result")

        if prediction == 1:
            st.error(
                "Classification: Cancelled invoice"
            )
        else:
            st.success(
                "Classification: Non-cancelled invoice"
            )

        st.caption(
            "This is a model classification, not a confirmed "
            "real-world transaction status."
        )


# -----------------------------------
# PAGE 3: PROBABILITY ANALYSIS
# -----------------------------------

elif page == "Probability Analysis":

    st.title("Probability Analysis")

    st.write(
        "Explore historical cancellation probabilities "
        "across different transaction characteristics."
    )

    total = len(invoice_data)
    cancelled = int(invoice_data["Cancelled"].sum())

    probability = cancelled / total

    col1, col2 = st.columns(2)

    col1.metric(
        "Overall Cancellation Probability",
        f"{probability * 100:.2f}%"
    )

    col2.metric(
        "Overall Non-cancellation Probability",
        f"{(1 - probability) * 100:.2f}%"
    )

    st.divider()

    st.subheader("Cancellation Probability by Country")

    countries = country_probability[
        country_probability["Total_Invoices"] >= 100
    ].sort_values(
        "Cancellation_Percentage",
        ascending=False
    )

    st.bar_chart(
        countries.set_index("Country")[
            "Cancellation_Percentage"
        ]
    )

    st.dataframe(
        countries[
            [
                "Country",
                "Total_Invoices",
                "Cancelled_Invoices",
                "Cancellation_Percentage"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Cancellation Probability by Hour")

    st.line_chart(
        hour_probability.set_index("Hour")[
            "Cancellation_Percentage"
        ]
    )

    st.dataframe(
        hour_probability,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Cancellation Probability by Day")

    st.bar_chart(
        day_probability.set_index("Day")[
            "Cancellation_Percentage"
        ]
    )

    st.dataframe(
        day_probability[
            [
                "Day",
                "Total_Invoices",
                "Cancelled_Invoices",
                "Cancellation_Percentage"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Country and time-based probabilities are historical "
        "conditional proportions. Small sample sizes can make "
        "some percentages unstable."
    )


# -----------------------------------
# PAGE 4: CUSTOMER SEGMENTATION
# -----------------------------------

elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    st.write(
        "Explore customer groups identified using "
        "K-Means clustering."
    )

    cluster_counts = customer_features["Cluster"].value_counts().sort_index()

    cols = st.columns(len(cluster_counts))

    for i, (cluster, count) in enumerate(cluster_counts.items()):
        cols[i].metric(
            f"Cluster {cluster} Customers",
            f"{count:,}"
        )

    st.divider()

    st.subheader("Cluster Comparison")

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    st.divider()

    st.subheader("Customer Segmentation Visualisation")

    st.scatter_chart(
        customer_features,
        x="Completed_Orders",
        y="Total_Spend",
        color="Cluster"
    )

    st.divider()

    st.subheader("Explore Individual Customers")

    selected_cluster = st.selectbox(
        "Select a customer cluster",
        sorted(customer_features["Cluster"].unique())
    )

    selected_customers = customer_features[
        customer_features["Cluster"] == selected_cluster
    ]

    st.dataframe(
        selected_customers.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Customer groups are based on purchasing and cancellation "
        "features used during K-Means training."
    )
