Deployed link : https://ofiinternship.streamlit.app/



# 📊 Customer Experience Dashboard  

### 🧠 Overview  
The **Customer Experience Dashboard** is an interactive data analytics web application developed as part of the **OFI Case Study Internship Project**.  
It helps businesses monitor customer satisfaction, delivery performance, logistics costs, and warehouse efficiency — all in one place.  

Built using **Python** and **Streamlit**, the dashboard integrates **seven datasets** related to logistics and customer feedback to generate **real-time performance insights** and **actionable interventions** for at-risk customers or delayed orders.  

---

## 🚀 Key Features  
✅ **Automatic Data Integration** — Merges seven CSV datasets seamlessly (Orders, Delivery, Feedback, Costs, Routes, Fleet, and Warehouse).  
✅ **Customer Risk Analysis** — Detects at-risk customers based on delays, low ratings, and complaints.  
✅ **Intervention Recommendations** — Suggests corrective actions like discounts, outreach, or faster delivery options.  
✅ **Interactive Visualizations** — Displays KPIs and trends using dynamic Plotly charts.  
✅ **Data Export** — Download the list of at-risk customers/orders with recommended interventions.  
✅ **Error Handling & Fallbacks** — Works even when `customer_id` is missing by analyzing order-level risk.  
✅ **Streamlit UI** — User-friendly, responsive, and easily deployable on the Streamlit Cloud.  

---

## 🧩 Datasets Used  

| Dataset | File Name | Description |
|----------|------------|-------------|
| Orders | `orders.csv` | Order IDs, dates, values, customer segments, product types |
| Delivery Performance | `delivery_performance.csv` | Actual vs promised delivery days, delays, status, and cost |
| Customer Feedback | `customer_feedback.csv` | Ratings, issue categories, and feedback text |
| Cost Breakdown | `cost_breakdown.csv` | Detailed cost components per order (fuel, labor, maintenance, etc.) |
| Routes | `routes_distance.csv` | Distances, tolls, traffic delays, and route details |
| Vehicle Fleet | `vehicle_fleet.csv` | Vehicle info: type, status, efficiency, and emissions |
| Warehouse Inventory | `warehouse_inventory.csv` | Stock levels, reorder thresholds, and storage costs |

> ⚠️ Keep all datasets in the same folder as your Python file or specify the folder path inside the app.  

---

## 💻 Tech Stack  

| Layer | Technology |
|--------|-------------|
| **Frontend** | Streamlit (Python-based web framework) |
| **Backend / Logic** | Python (Pandas, NumPy) |
| **Visualization** | Plotly |
| **Data Source** | CSV Datasets |
| **Deployment** | Streamlit Community Cloud |

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SwayamAggarwal/OFI-.git
cd OFI-
```

### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate    # for Windows
# or source .venv/bin/activate for Mac/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app
```bash
streamlit run customer_experience_dashboard.py
```

Then open your browser and visit 👉 **http://localhost:8501**

---

## 📦 Dependencies  

Your `requirements.txt` should include:

```txt
streamlit>=1.20
pandas>=1.5
numpy
plotly
boto3   # only if reading data from AWS S3
```

Install them manually if needed:
```bash
pip install streamlit pandas numpy plotly boto3
```

---

## 📈 Visualizations Included  

| Chart Type | Description |
|-------------|--------------|
| 🥧 Pie Chart | Customer risk distribution (Low, Medium, High) |
| 📊 Bar + Line | Orders and revenue trends over time |
| 📉 Histogram | Rating distribution from customer feedback |
| ⚫ Scatter Plot | Relationship between delay and rating |
| 📦 Inventory Chart | Warehouse stock vs reorder levels |
| 🚚 Route Cost Analysis | Route distance vs cost insights |

---

## 📋 Key KPIs Displayed  

| KPI | Description |
|------|-------------|
| Total Customers / Orders | Total number of unique customers or orders |
| At-Risk Customers | Count and percentage of high/medium-risk customers |
| Average Rating | Mean customer satisfaction rating |
| Average Delay (mins) | Mean delivery delay duration |

---

## 🧠 Risk Scoring Logic  
Customers or orders are assigned a **risk score** based on:  
- **Delivery delay duration** (higher delay = higher risk)  
- **Average rating** (lower rating = higher risk)  
- **Complaint count**  
- **Recency** (inactive for > 90 days = higher risk)

**Risk Levels:**
| Risk Level | Score Range |
|-------------|--------------|
| Low | 0–1 |
| Medium | 2 |
| High | ≥3 |

---

## 💡 Interventions (Recommendations)

| Risk Level | Example Condition | Recommended Action |
|-------------|------------------|--------------------|
| **High** | Delay > 60 mins or Rating < 3 | Personal call + 30% coupon + investigate logistics |
| **Medium** | Delay 15–60 mins | Send apology + 10% discount |
| **Low** | On-time, rating ≥4 | No immediate action |

Users can **download** a CSV of at-risk customers/orders along with these interventions.

---


## 🧑‍💻 Author  

**Swayam Aggarwal**  
🎓 B.Tech (Computer Science & Engineering)  
📍 OFI Case Study Internship Participant  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/swayamaggarwal)  

---



## 📁 Recommended File Structure  

```
OFI-
├── customer_experience_dashboard.py
├── requirements.txt
├── README.md
└── Case study internship data/
    ├── orders.csv
    ├── delivery_performance.csv
    ├── customer_feedback.csv
    ├── routes_distance.csv
    ├── vehicle_fleet.csv
    ├── warehouse_inventory.csv
    └── cost_breakdown.csv
```

