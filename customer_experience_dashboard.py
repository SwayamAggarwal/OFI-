# customer_experience_dashboard.py
"""
Customer Experience Dashboard (robust fallback + interventions download)
- Auto-maps columns (Order_ID -> order_id, Order_Date -> order_date, Order_Value_INR -> amount, etc.)
- If customer_id missing: computes risk at order-level and provides interventions per order
- Adds download button for at-risk entities (customers or orders)
Run:
    python -m pip install --upgrade pip
    python -m pip install streamlit pandas plotly numpy
    python -m streamlit run customer_experience_dashboard.py
"""
import os
from pathlib import Path
import io
import datetime
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer Experience Dashboard (Fallback + Interventions)", layout="wide")

DEFAULT_DATA_DIR = r"C:\Users\hp\Desktop\OFI\Case study internship data"
EXPECTED_FILES = {
    "orders": ["orders.csv", "orders_data.csv"],
    "delivery": ["delivery_performance.csv", "delivery.csv"],
    "feedback": ["customer_feedback.csv", "feedback.csv"],
    "vehicles": ["vehicle_fleet.csv", "vehicles.csv"],
    "routes": ["routes_distance.csv", "routes.csv"],
    "warehouse": ["warehouse_inventory.csv", "inventory.csv"],
    "costs": ["cost_breakdown.csv", "costs.csv"],
}

# ---------------- Utilities ----------------
@st.cache_data(show_spinner=False)
def find_files_in_dir(folder: str, expected: Dict[str, List[str]]) -> Dict[str, str]:
    p = Path(folder)
    found_map = {k: "" for k in expected}
    if not p.exists() or not p.is_dir():
        return found_map
    csvs = {f.name.lower(): str(f) for f in p.glob("*.csv")}
    for key, names in expected.items():
        for name in names:
            if name.lower() in csvs:
                found_map[key] = csvs[name.lower()]
                break
    return found_map

def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        st.warning(f"Could not read CSV {path}: {e}")
        return pd.DataFrame()

def try_parse_dates(df: pd.DataFrame, candidates: List[str]) -> Tuple[pd.DataFrame, str]:
    for c in candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                return df, c
            except Exception:
                continue
    return df, ""

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df

# ---------------- Load ----------------
@st.cache_data(show_spinner=False)
def load_datasets(folder: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    paths = find_files_in_dir(folder, EXPECTED_FILES)
    dfs = {k: safe_read_csv(v) for k, v in paths.items()}
    for k, df in dfs.items():
        if not df.empty:
            dfs[k] = normalize_colnames(df)
    meta = {}
    # initial try to detect dates with common names (will be updated after adapt step)
    orders_date_candidates = ["order_date", "Order_Date", "date", "created_at"]
    delivery_date_candidates = ["delivery_date", "Delivery_Date", "delivered_at", "date"]
    feedback_date_candidates = ["feedback_date", "Feedback_Date", "date"]
    dfs["orders"], meta["orders_date_col"] = try_parse_dates(dfs["orders"], orders_date_candidates)
    dfs["delivery"], meta["delivery_date_col"] = try_parse_dates(dfs["delivery"], delivery_date_candidates)
    dfs["feedback"], meta["feedback_date_col"] = try_parse_dates(dfs["feedback"], feedback_date_candidates)
    return dfs, meta

# ---------------- Auto-map & adapt ----------------
def adapt_columns(dfs: dict) -> Tuple[dict, dict]:
    dfs = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in dfs.items()}

    # ORDERS
    if not dfs.get("orders", pd.DataFrame()).empty:
        o = dfs["orders"]
        rename_o = {}
        if "Order_ID" in o.columns: rename_o["Order_ID"] = "order_id"
        if "Order_Date" in o.columns: rename_o["Order_Date"] = "order_date"
        if "Order_Value_INR" in o.columns: rename_o["Order_Value_INR"] = "amount"
        if "Customer_ID" in o.columns: rename_o["Customer_ID"] = "customer_id"
        if rename_o:
            o = o.rename(columns=rename_o)
        if "order_date" in o.columns:
            try:
                o["order_date"] = pd.to_datetime(o["order_date"], errors="coerce")
            except Exception:
                pass
        dfs["orders"] = o

    # DELIVERY
    if not dfs.get("delivery", pd.DataFrame()).empty:
        d = dfs["delivery"]
        rename_d = {}
        if "Order_ID" in d.columns: rename_d["Order_ID"] = "order_id"
        if "Actual_Delivery_Days" in d.columns: rename_d["Actual_Delivery_Days"] = "actual_delivery_days"
        if "Promised_Delivery_Days" in d.columns: rename_d["Promised_Delivery_Days"] = "promised_delivery_days"
        if "Customer_Rating" in d.columns: rename_d["Customer_Rating"] = "rating"
        if "Delivery_Cost_INR" in d.columns: rename_d["Delivery_Cost_INR"] = "delivery_cost"
        if rename_d:
            d = d.rename(columns=rename_d)
        if {"actual_delivery_days", "promised_delivery_days"}.issubset(set(d.columns)):
            try:
                d["delivery_delay_days"] = pd.to_numeric(d["actual_delivery_days"], errors="coerce") - pd.to_numeric(d["promised_delivery_days"], errors="coerce")
                d["delay_minutes"] = d["delivery_delay_days"] * 24 * 60
            except Exception:
                d["delay_minutes"] = np.nan
        dfs["delivery"] = d

    # FEEDBACK
    if not dfs.get("feedback", pd.DataFrame()).empty:
        f = dfs["feedback"]
        rename_f = {}
        if "Order_ID" in f.columns: rename_f["Order_ID"] = "order_id"
        if "Feedback_Date" in f.columns: rename_f["Feedback_Date"] = "feedback_date"
        if "Rating" in f.columns: rename_f["Rating"] = "rating"
        if rename_f:
            f = f.rename(columns=rename_f)
        if "feedback_date" in f.columns:
            try:
                f["feedback_date"] = pd.to_datetime(f["feedback_date"], errors="coerce")
            except Exception:
                pass
        dfs["feedback"] = f

    # ROUTES
    if not dfs.get("routes", pd.DataFrame()).empty:
        r = dfs["routes"]
        rename_r = {}
        if "Order_ID" in r.columns: rename_r["Order_ID"] = "order_id"
        if "Distance_KM" in r.columns: rename_r["Distance_KM"] = "distance_km"
        if "Traffic_Delay_Minutes" in r.columns: rename_r["Traffic_Delay_Minutes"] = "traffic_delay_minutes"
        if rename_r:
            r = r.rename(columns=rename_r)
        dfs["routes"] = r

    # VEHICLES
    if not dfs.get("vehicles", pd.DataFrame()).empty:
        v = dfs["vehicles"]
        rename_v = {}
        if "Vehicle_ID" in v.columns: rename_v["Vehicle_ID"] = "vehicle_id"
        if "Vehicle_Type" in v.columns: rename_v["Vehicle_Type"] = "vehicle_type"
        if rename_v:
            v = v.rename(columns=rename_v)
        dfs["vehicles"] = v

    # WAREHOUSE
    if not dfs.get("warehouse", pd.DataFrame()).empty:
        w = dfs["warehouse"]
        rename_w = {}
        if "Warehouse_ID" in w.columns: rename_w["Warehouse_ID"] = "warehouse_id"
        if "Current_Stock_Units" in w.columns: rename_w["Current_Stock_Units"] = "quantity"
        if "Reorder_Level" in w.columns: rename_w["Reorder_Level"] = "reorder_level"
        if rename_w:
            w = w.rename(columns=rename_w)
        if "sku" not in w.columns and "Product_Category" in w.columns:
            w["sku"] = w["Product_Category"]
        dfs["warehouse"] = w

    # COSTS
    if not dfs.get("costs", pd.DataFrame()).empty:
        c = dfs["costs"]
        rename_c = {}
        if "Order_ID" in c.columns: rename_c["Order_ID"] = "order_id"
        if rename_c:
            c = c.rename(columns=rename_c)
        cost_cols = [col for col in c.columns if any(k in col.lower() for k in ["cost", "fee", "overhead", "charges", "maintenance"]) and col != "order_id"]
        if cost_cols:
            try:
                c["total_cost"] = c[cost_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            except Exception:
                c["total_cost"] = np.nan
        dfs["costs"] = c

    # Rebuild metadata for orders date column to canonical 'order_date' if present
    meta = {}
    if "order_date" in dfs.get("orders", pd.DataFrame()).columns:
        meta["orders_date_col"] = "order_date"
    else:
        meta["orders_date_col"] = ""
    return dfs, meta

# ---------------- Derived metrics & fallback ----------------
def prepare_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "order_id" not in df.columns:
        for alt in ["id", "orderid", "OrderID"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "order_id"})
                break
    if "customer_id" not in df.columns:
        for alt in ["cust_id", "client_id", "CustomerID"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "customer_id"})
                break
    if "amount" not in df.columns:
        for alt in ["total", "Order_Value_INR", "order_value", "price"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "amount"})
                break
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df

def merge_main(orders: pd.DataFrame, delivery: pd.DataFrame, feedback: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    o = prepare_orders(orders)
    df = o.copy()
    if not delivery.empty:
        try:
            if "order_id" in delivery.columns:
                df = df.merge(delivery, on="order_id", how="left", suffixes=("", "_delivery"))
            elif "customer_id" in delivery.columns and "customer_id" in df.columns:
                df = df.merge(delivery, on="customer_id", how="left", suffixes=("", "_delivery"))
        except Exception as e:
            st.warning(f"Delivery merge failed: {e}")
    if not feedback.empty:
        try:
            if "order_id" in feedback.columns and "order_id" in df.columns:
                df = df.merge(feedback, on="order_id", how="left", suffixes=("", "_feedback"))
            elif "customer_id" in feedback.columns and "customer_id" in df.columns:
                df = df.merge(feedback, on="customer_id", how="left", suffixes=("", "_feedback"))
        except Exception as e:
            st.warning(f"Feedback merge failed: {e}")
    if not costs.empty:
        try:
            if "order_id" in costs.columns and "total_cost" in costs.columns:
                costs_small = costs[["order_id", "total_cost"]].drop_duplicates(subset=["order_id"])
                df = df.merge(costs_small, on="order_id", how="left")
        except Exception as e:
            st.warning(f"Costs merge failed: {e}")
    return df

def compute_customer_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    If customer_id exists -> compute per-customer KPIs.
    If not -> compute per-order KPIs and create synthetic customer_id (order-level) fallback.
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    # ensure order_id
    if "order_id" not in df.columns:
        df["order_id"] = df.index.astype(str)

    # If there is no customer_id, fallback to order-level grouping and create synthetic customer_id
    fallback_to_orders = False
    if "customer_id" not in df.columns:
        fallback_to_orders = True
        # create synthetic customer_id = ORDER::<order_id> so grouping produces per-order rows
        df["customer_id"] = df["order_id"].apply(lambda x: f"ORDER::{x}")

    grp = df.groupby("customer_id")
    summary = grp.agg(
        orders_count=("order_id", lambda x: x.nunique()),
        total_spent=("amount", lambda x: x.sum(skipna=True) if "amount" in df.columns else np.nan)
    ).reset_index()

    # last order date
    date_fields = [c for c in ["order_date", "Order_Date", "date", "created_at", "delivered_at"] if c in df.columns]
    if date_fields:
        last = df.groupby("customer_id")[date_fields[0]].max().reset_index().rename(columns={date_fields[0]: "last_order_date"})
        summary = summary.merge(last, on="customer_id", how="left")
    else:
        summary["last_order_date"] = pd.NaT

    # rating
    if "rating" in df.columns:
        ratings = grp["rating"].mean().reset_index().rename(columns={"rating": "avg_rating"})
        summary = summary.merge(ratings, on="customer_id", how="left")
    else:
        summary["avg_rating"] = np.nan

    # complaints
    complaint_col = None
    for c in ["complaint", "complaints", "num_complaints", "complaint_count"]:
        if c in df.columns:
            complaint_col = c
            break
    if complaint_col:
        comp = grp[complaint_col].sum(min_count=1).reset_index().rename(columns={complaint_col: "complaints_count"})
        summary = summary.merge(comp, on="customer_id", how="left")
    else:
        summary["complaints_count"] = 0

    # delay
    delay_col = None
    for c in ["delay_minutes", "delay", "delivery_delay_days"]:
        if c in df.columns:
            delay_col = c
            break
    if delay_col:
        delays = grp[delay_col].mean().reset_index().rename(columns={delay_col: "avg_delay_minutes"})
        summary = summary.merge(delays, on="customer_id", how="left")
    else:
        summary["avg_delay_minutes"] = np.nan

    summary["last_order_date"] = pd.to_datetime(summary["last_order_date"], errors="coerce")
    today = pd.to_datetime(datetime.datetime.now())
    summary["recency_days"] = (today - summary["last_order_date"]).dt.days

    # scoring heuristic
    def score_row(r):
        s = 0
        if not pd.isna(r.get("avg_delay_minutes", np.nan)):
            if r["avg_delay_minutes"] > 60:
                s += 2
            elif r["avg_delay_minutes"] > 15:
                s += 1
        if not pd.isna(r.get("avg_rating", np.nan)):
            if r["avg_rating"] < 3:
                s += 2
            elif r["avg_rating"] < 4:
                s += 1
        if r.get("complaints_count", 0) >= 1:
            s += 1
        if not pd.isna(r.get("recency_days", np.nan)) and r["recency_days"] > 90:
            s += 1
        return s

    summary["risk_score"] = summary.apply(score_row, axis=1)
    summary["risk_level"] = pd.cut(summary["risk_score"], bins=[-1, 0, 2, 99], labels=["Low", "Medium", "High"]).astype(str)
    summary["CLV"] = summary["total_spent"]
    # include flag whether this was fallback
    summary["is_order_fallback"] = fallback_to_orders
    return summary

# ---------------- Interventions helper ----------------
def recommend_intervention(row: pd.Series) -> str:
    """Return a suggested intervention string based on risk and driver columns."""
    # high-level suggestions
    if row.get("risk_level") == "High":
        # If high delay & low rating & complaints -> personal outreach + discount
        if not pd.isna(row.get("avg_delay_minutes")) and row.get("avg_delay_minutes") > 60:
            return "High priority: Personal call + 30% discount coupon; investigate logistics."
        if not pd.isna(row.get("avg_rating")) and row.get("avg_rating") < 3:
            return "High priority: Refund/Replacement + personal follow-up; quality check."
        return "High priority: Personal outreach; retention offer."
    elif row.get("risk_level") == "Medium":
        # medium -> feedback form / small incentive
        if not pd.isna(row.get("avg_delay_minutes")) and row.get("avg_delay_minutes") > 15:
            return "Send apology + 10% discount; offer faster shipping on next order."
        return "Send feedback form & 5% promo; monitor next order."
    else:
        return "No action required (Low risk)."

# ---------------- Streamlit UI ----------------
def main():
    st.title("Customer Experience Dashboard — Fallback & Interventions")
    st.markdown("If `customer_id` is not present in your data, the dashboard will compute risk per order and provide interventions per order.")

    st.sidebar.header("Data folder")
    data_dir = st.sidebar.text_input("Enter full path to your data folder", value=DEFAULT_DATA_DIR)
    st.sidebar.caption("Expected CSVs: orders, delivery_performance, customer_feedback, vehicle_fleet, routes_distance, warehouse_inventory, cost_breakdown")

    with st.spinner("Loading datasets..."):
        dfs, meta = load_datasets(data_dir)
        # adapt columns to canonical names; get updated meta
        try:
            dfs, new_meta = adapt_columns(dfs)
            # prefer new_meta orders_date_col if present
            if new_meta.get("orders_date_col"):
                meta["orders_date_col"] = new_meta["orders_date_col"]
        except Exception as e:
            st.warning(f"Auto-mapping failed: {e}")

    # Show what's loaded
    st.sidebar.header("Files found")
    found = find_files_in_dir(data_dir, EXPECTED_FILES)
    for key, path in found.items():
        st.sidebar.write(f"{key}: {Path(path).name if path else 'NOT FOUND'}")

    with st.expander("Datasets loaded (samples & columns)"):
        for name, df in dfs.items():
            st.write(f"**{name}** — rows: {len(df)} columns: {len(df.columns)}")
            if len(df) > 0:
                st.write("Columns:", list(df.columns))
                st.dataframe(df.head(3))

    # Merge and compute
    merged = merge_main(dfs.get("orders", pd.DataFrame()), dfs.get("delivery", pd.DataFrame()), dfs.get("feedback", pd.DataFrame()), dfs.get("costs", pd.DataFrame()))
    customer_summary = compute_customer_kpis(merged)

    # Filters
    st.sidebar.header("Filters")
    orders_date_col = meta.get("orders_date_col", "")
    if orders_date_col and orders_date_col in dfs.get("orders", pd.DataFrame()).columns:
        try:
            min_date = pd.to_datetime(dfs["orders"][orders_date_col].min()).date()
            max_date = pd.to_datetime(dfs["orders"][orders_date_col].max()).date()
            date_range = st.sidebar.date_input("Order date range", [min_date, max_date])
        except Exception:
            date_range = None
    else:
        date_range = None

    risk_choices = ["Low", "Medium", "High"]
    selected_risks = st.sidebar.multiselect("Risk levels", options=risk_choices, default=risk_choices)

    # Apply date filter to merged
    filtered = merged.copy()
    if date_range and orders_date_col and orders_date_col in filtered.columns:
        try:
            start, end = date_range
            filtered = filtered[(filtered[orders_date_col].dt.date >= start) & (filtered[orders_date_col].dt.date <= end)]
        except Exception:
            pass

    # recompute summary after filtering
    filtered_summary = compute_customer_kpis(filtered)
    if not filtered_summary.empty:
        filtered_summary = filtered_summary[filtered_summary["risk_level"].isin(selected_risks)]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_entities = int(customer_summary["customer_id"].nunique()) if not customer_summary.empty else 0
        st.metric("Total Customers / Entities", total_entities)
    with col2:
        at_risk_count = int(customer_summary[customer_summary["risk_level"].isin(["Medium", "High"])].shape[0]) if not customer_summary.empty else 0
        pct = f"{round(100*at_risk_count/total_entities,1)}%" if total_entities else "N/A"
        st.metric("At-Risk (Medium+High)", at_risk_count, delta=pct)
    with col3:
        avg_rating = round(customer_summary["avg_rating"].mean(), 2) if "avg_rating" in customer_summary.columns and not customer_summary["avg_rating"].isna().all() else "N/A"
        st.metric("Avg Rating", avg_rating)
    with col4:
        avg_delay = round(customer_summary["avg_delay_minutes"].mean(), 1) if "avg_delay_minutes" in customer_summary.columns and not customer_summary["avg_delay_minutes"].isna().all() else "N/A"
        st.metric("Avg Delay (min)", avg_delay)

    st.markdown("---")

    # Visuals
    st.subheader("Risk Distribution")
    if not filtered_summary.empty:
        fig_pie = px.pie(filtered_summary, names="risk_level", title="Risk Level Distribution", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No customer/order summary available for current filters to show risk distribution.")

    st.subheader("Orders Over Time")
    if not dfs.get("orders", pd.DataFrame()).empty and orders_date_col and orders_date_col in dfs["orders"].columns:
        try:
            od = dfs["orders"].copy()
            od[orders_date_col] = pd.to_datetime(od[orders_date_col], errors="coerce")
            od = od.dropna(subset=[orders_date_col])
            od["order_date_only"] = od[orders_date_col].dt.date
            if "order_id" in od.columns:
                ts = od.groupby("order_date_only").agg(orders_count=("order_id", "nunique"), revenue=("amount", "sum")).reset_index()
            else:
                ts = od.groupby("order_date_only").agg(orders_count=("order_date_only", "count"), revenue=("amount", "sum")).reset_index()
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Bar(x=ts["order_date_only"], y=ts["orders_count"], name="Orders"))
            if "revenue" in ts.columns:
                fig_ts.add_trace(go.Scatter(x=ts["order_date_only"], y=ts["revenue"], name="Revenue", yaxis="y2"))
                fig_ts.update_layout(yaxis2=dict(overlaying="y", side="right", title="Revenue"))
            fig_ts.update_layout(title="Orders and Revenue Over Time", xaxis_title="Date")
            st.plotly_chart(fig_ts, use_container_width=True)
        except Exception as e:
            st.warning(f"Time series failed: {e}")
    else:
        st.info("Orders dataset or date column missing for time-series chart.")

    st.subheader("Rating Distribution (Feedback)")
    if not dfs.get("feedback", pd.DataFrame()).empty and "rating" in dfs["feedback"].columns:
        try:
            fig_hist = px.histogram(dfs["feedback"], x="rating", nbins=5, title="Rating Distribution", marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.warning(f"Ratings chart error: {e}")
    else:
        st.info("No ratings available in feedback dataset.")

    st.subheader("Delay vs Rating (customer/order-level)")
    if not customer_summary.empty and "avg_delay_minutes" in customer_summary.columns and "avg_rating" in customer_summary.columns:
        try:
            fig_scatter = px.scatter(customer_summary, x="avg_delay_minutes", y="avg_rating", color="risk_level",
                                     size="orders_count", hover_data=["customer_id", "CLV"], title="Avg Delay vs Avg Rating")
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception as e:
            st.warning(f"Scatter chart error: {e}")
    else:
        st.info("Delay and/or rating not present to build scatter.")

    # At-risk entities (customers or order-fallback) with interventions
    st.subheader("At-Risk Entities & Suggested Interventions")
    if not filtered_summary.empty:
        # add intervention recommendations
        out = filtered_summary.copy()
        out["suggested_intervention"] = out.apply(recommend_intervention, axis=1)
        # if this is fallback to orders, include mapping to order_id for clarity
        if out["is_order_fallback"].all():
            # join with merged to bring order-level columns (order_id, origin, destination, amount, etc.)
            lookup = merged[["order_id"] + [c for c in merged.columns if c not in ["order_id"]]].copy() if "order_id" in merged.columns else merged.copy()
            # best-effort: merge by synthetic customer_id -> order_id (extract)
            out = out.copy()
            out["order_id"] = out["customer_id"].apply(lambda x: x.split("ORDER::")[-1] if isinstance(x, str) and x.startswith("ORDER::") else None)
            # bring some order details
            details = merged.set_index("order_id")[["Origin","Destination","amount"]].reset_index() if "order_id" in merged.columns else pd.DataFrame()
            if not details.empty:
                out = out.merge(details, on="order_id", how="left")
        else:
            # not fallback -> we have true customer_id; optionally include last order id (best-effort)
            # get sample order_id per customer
            if "order_id" in merged.columns:
                last_orders = merged.groupby("customer_id")["order_id"].agg(lambda s: ",".join(sorted(s.unique()[:3]))).reset_index().rename(columns={"order_id":"sample_order_ids"})
                out = out.merge(last_orders, on="customer_id", how="left")
        st.dataframe(out.sort_values(["risk_score","recency_days"], ascending=[False, True]).reset_index(drop=True))
        # download
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download At-Risk List with Interventions (CSV)", data=csv, file_name="at_risk_with_interventions.csv", mime="text/csv")
    else:
        st.info("No at-risk customers/orders to show for current filters.")

    st.markdown("---")
    st.info("Note: If your dataset lacks a `customer_id`, the app treats each order as an entity (ORDER::<order_id>) to still surface risky items and interventions. For true customer-level aggregates, add a customer identifier column to orders or feedback files (e.g., Customer_ID).")

if __name__ == "__main__":
    main()
