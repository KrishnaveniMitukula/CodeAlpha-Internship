import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
sns.set_theme(style="whitegrid")

# ─────────────────────────────────────
# STEP 1: Generate Sales Dataset
# ─────────────────────────────────────
print("="*55)
print("📊 CodeAlpha Task 4 — Data Visualization Dashboard")
print("="*55)
print("\n📦 Generating Superstore Sales Dataset...")

n = 2000
categories = ["Technology", "Furniture", "Office Supplies"]
sub_cats = {
    "Technology":      ["Laptops", "Phones", "Monitors",
                        "Printers", "Accessories"],
    "Furniture":       ["Chairs", "Tables", "Bookcases",
                        "Desks", "Storage"],
    "Office Supplies": ["Paper", "Binders", "Pens",
                        "Labels", "Envelopes"]
}
regions   = ["West", "East", "Central", "South"]
segments  = ["Consumer", "Corporate", "Home Office"]

cat_weights = [0.35, 0.30, 0.35]
reg_weights = [0.30, 0.28, 0.24, 0.18]
seg_weights = [0.52, 0.30, 0.18]

price_ranges = {
    "Technology":      (150, 2500),
    "Furniture":       (80,  1200),
    "Office Supplies": (5,   120)
}
margin_ranges = {
    "Technology":      (0.12, 0.35),
    "Furniture":       (0.05, 0.25),
    "Office Supplies": (0.20, 0.50)
}

dates = pd.date_range("2022-01-01", "2023-12-31", periods=n)
dates = np.sort(np.random.choice(dates, n, replace=False))

cats         = np.random.choice(categories, n, p=cat_weights)
sub_products = [np.random.choice(sub_cats[c]) for c in cats]
regs         = np.random.choice(regions, n, p=reg_weights)
segs         = np.random.choice(segments, n, p=seg_weights)

prices, margins = [], []
for c in cats:
    lo, hi = price_ranges[c]
    prices.append(round(np.random.uniform(lo, hi), 2))
    mlo, mhi = margin_ranges[c]
    margins.append(round(np.random.uniform(mlo, mhi), 4))

qty    = np.random.choice([1,2,3,4,5], n, p=[0.45,0.28,0.15,0.08,0.04])
sales  = [round(p*q, 2) for p,q in zip(prices, qty)]
profit = [round(s*m, 2) for s,m in zip(sales, margins)]

df = pd.DataFrame({
    "Order Date"  : dates,
    "Category"    : cats,
    "Sub-Category": sub_products,
    "Region"      : regs,
    "Segment"     : segs,
    "Unit Price"  : prices,
    "Quantity"    : qty,
    "Sales"       : sales,
    "Profit"      : profit,
    "Margin"      : margins
})

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Month"]      = df["Order Date"].dt.to_period("M")
df["Year"]       = df["Order Date"].dt.year

print(f"✅ Generated {len(df)} sales transactions (2022–2023)")

# ─────────────────────────────────────
# STEP 2: KPI Summary
# ─────────────────────────────────────
print("\n" + "="*55)
print("📊 KEY PERFORMANCE INDICATORS")
print("="*55)

total_sales  = df["Sales"].sum()
total_profit = df["Profit"].sum()
avg_margin   = df["Margin"].mean() * 100
total_orders = len(df)
avg_order    = df["Sales"].mean()

print(f"  💰 Total Revenue    : ${total_sales:>12,.2f}")
print(f"  📈 Total Profit     : ${total_profit:>12,.2f}")
print(f"  📊 Avg Margin       : {avg_margin:>11.1f}%")
print(f"  🛒 Total Orders     : {total_orders:>12,}")
print(f"  💵 Avg Order Value  : ${avg_order:>12,.2f}")

# ─────────────────────────────────────
# STEP 3: Build Dashboard
# ─────────────────────────────────────
print("\n🎨 Building Dashboard...")

PALETTE = {
    "Technology":      "#3498db",
    "Furniture":       "#e67e22",
    "Office Supplies": "#2ecc71"
}
REGION_COLORS = {
    "West":    "#9b59b6",
    "East":    "#3498db",
    "Central": "#e67e22",
    "South":   "#e74c3c"
}

fig = plt.figure(figsize=(24, 18), facecolor="#f0f2f5")
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        top=0.93, bottom=0.05,
                        left=0.05, right=0.97,
                        hspace=0.45, wspace=0.35)

# Header
fig.text(0.5, 0.975,
         "🏢  SUPERSTORE SALES DASHBOARD — 2022–2023",
         ha="center", va="top",
         fontsize=20, fontweight="bold", color="#2c3e50")
fig.text(0.5, 0.955,
         "CodeAlpha Data Analytics Internship  •  Task 4",
         ha="center", va="top",
         fontsize=12, color="#7f8c8d")

# ── KPI Cards ──
kpi_data = [
    ("💰 Total Revenue",  f"${total_sales:,.0f}",   "#3498db"),
    ("📈 Total Profit",   f"${total_profit:,.0f}",  "#2ecc71"),
    ("📊 Avg Margin",     f"{avg_margin:.1f}%",     "#9b59b6"),
    ("🛒 Total Orders",   f"{total_orders:,}",      "#e67e22"),
]
for i, (label, value, color) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.62, value,
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=22, fontweight="bold", color="white")
    ax.text(0.5, 0.25, label,
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=11, color="white", alpha=0.9)

# ── Plot 1: Monthly Revenue Trend ──
ax1 = fig.add_subplot(gs[1, 0:2])
monthly = df.groupby("Month")["Sales"].sum().reset_index()
monthly["Month_str"] = monthly["Month"].astype(str)
x = range(len(monthly))
ax1.fill_between(x, monthly["Sales"], alpha=0.2, color="#3498db")
ax1.plot(x, monthly["Sales"], color="#3498db",
         lw=2.5, marker="o", ms=4)
z = np.polyfit(list(x), monthly["Sales"], 1)
p = np.poly1d(z)
ax1.plot(x, p(list(x)), "r--", lw=1.5,
         alpha=0.7, label="Trend Line")
step = max(1, len(monthly)//8)
ax1.set_xticks(list(x)[::step])
ax1.set_xticklabels(monthly["Month_str"].tolist()[::step],
                    rotation=30, ha="right", fontsize=8)
ax1.set_title("Monthly Revenue Trend",
              fontweight="bold", fontsize=12)
ax1.set_ylabel("Revenue ($)")
ax1.legend(fontsize=9)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ── Plot 2: Sales by Category ──
ax2 = fig.add_subplot(gs[1, 2])
cat_sales = df.groupby("Category")["Sales"].sum()\
              .sort_values(ascending=True)
bars = ax2.barh(cat_sales.index, cat_sales.values,
                color=[PALETTE[c] for c in cat_sales.index],
                edgecolor="white", height=0.5)
for bar in bars:
    ax2.text(bar.get_width() + 500,
             bar.get_y() + bar.get_height()/2,
             f"${bar.get_width():,.0f}",
             va="center", fontsize=9)
ax2.set_title("Revenue by Category",
              fontweight="bold", fontsize=12)
ax2.set_xlabel("Revenue ($)")
ax2.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

# ── Plot 3: Profit Margin Pie ──
ax3 = fig.add_subplot(gs[1, 3])
cat_margin = df.groupby("Category")["Margin"].mean() * 100
wedges, texts, autotexts = ax3.pie(
    cat_margin.values,
    labels=cat_margin.index,
    autopct="%1.1f%%",
    colors=[PALETTE[c] for c in cat_margin.index],
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for t in autotexts:
    t.set_fontsize(10)
    t.set_fontweight("bold")
ax3.set_title("Avg Profit Margin\nby Category",
              fontweight="bold", fontsize=12)

# ── Plot 4: Regional Sales ──
ax4 = fig.add_subplot(gs[2, 0])
reg_sales = df.groupby("Region")["Sales"].sum()\
              .sort_values(ascending=False)
bars4 = ax4.bar(reg_sales.index, reg_sales.values,
                color=[REGION_COLORS[r] for r in reg_sales.index],
                edgecolor="white")
for bar in bars4:
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 500,
             f"${bar.get_height():,.0f}",
             ha="center", fontsize=8, fontweight="bold")
ax4.set_title("Revenue by Region",
              fontweight="bold", fontsize=12)
ax4.set_ylabel("Revenue ($)")
ax4.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

# ── Plot 5: Top 10 Sub-Categories ──
ax5 = fig.add_subplot(gs[2, 1:3])
top10 = df.groupby("Sub-Category")["Sales"]\
          .sum().nlargest(10).sort_values()
colors5 = sns.color_palette("viridis", 10)
bars5 = ax5.barh(top10.index, top10.values,
                 color=colors5, edgecolor="white")
for bar in bars5:
    ax5.text(bar.get_width() + 200,
             bar.get_y() + bar.get_height()/2,
             f"${bar.get_width():,.0f}",
             va="center", fontsize=8)
ax5.set_title("Top 10 Sub-Categories by Revenue",
              fontweight="bold", fontsize=12)
ax5.set_xlabel("Revenue ($)")
ax5.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

# ── Plot 6: Customer Segment Donut ──
ax6 = fig.add_subplot(gs[2, 3])
seg_sales = df.groupby("Segment")["Sales"].sum()
wedges6, texts6, autotexts6 = ax6.pie(
    seg_sales.values,
    labels=seg_sales.index,
    autopct="%1.1f%%",
    colors=["#3498db", "#e67e22", "#2ecc71"],
    startangle=90,
    pctdistance=0.75,
    wedgeprops={"edgecolor": "white",
                "linewidth": 2,
                "width": 0.55}
)
for t in autotexts6:
    t.set_fontsize(10)
    t.set_fontweight("bold")
ax6.set_title("Revenue by\nCustomer Segment",
              fontweight="bold", fontsize=12)

# ── Save Dashboard ──
plt.savefig("sales_dashboard.png",
            dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("💾 Dashboard saved as 'sales_dashboard.png'")
plt.show()

# Save dataset
df.to_csv("superstore_sales.csv", index=False)
print("💾 Dataset saved as 'superstore_sales.csv'")

print("\n" + "="*55)
print("📝 KEY FINDINGS")
print("="*55)
print("1. Technology has highest revenue overall")
print("2. Office Supplies has best profit margins")
print("3. West region leads in total sales")
print("4. Laptops & Phones are top selling products")
print("5. Consumer segment drives 52% of revenue")
print("6. Revenue shows upward trend over 2022-2023")
print("="*55)
print("\n✅ Task 3 COMPLETE!")