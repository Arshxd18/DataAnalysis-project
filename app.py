import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── 1. Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce RetailTech Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. Futuristic Dark Theme CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark canvas reset */
    .stApp {
        background-color: #0A0E17 !important;
        color: #E2E8F0 !important;
    }
    
    /* Top title styling - pure white */
    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #FFFFFF !important;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    
    .sub-title {
        font-size: 14.5px;
        color: #94A3B8 !important;
        margin-bottom: 24px;
        font-weight: 400;
    }
    
    /* Metric cards styling */
    [data-testid="stMetric"] {
        background: #111726;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px 8px 0 0;
        color: #94A3B8 !important;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #0E1320 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
        color: #CBD5E1 !important;
    }

    /* Recommendation & alert cards */
    .takeaway-card {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 16px;
        font-size: 13.5px;
        color: #E2E8F0;
        line-height: 1.5;
    }
    
    .takeaway-card b {
        color: #34D399;
    }

    .alert-card {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-left: 4px solid #EF4444;
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 16px;
        font-size: 13.5px;
        color: #E2E8F0;
        line-height: 1.5;
    }
    
    .alert-card b {
        color: #F87171;
    }

    /* Footer styling */
    .footer-bar {
        text-align: center;
        color: #64748B;
        font-size: 12.5px;
        font-weight: 500;
        padding: 32px 0 16px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 40px;
        letter-spacing: 0.4px;
    }
</style>
""", unsafe_allow_html=True)

# ── 3. Data Loading with Cache ────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base_dir, "master_df_cleaned.csv"),
        os.path.join(base_dir, "ProjectData", "master_df_cleaned.csv"),
    ]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Format clean category name
            if 'category_name' in df.columns:
                df['clean_category'] = df['category_name'].astype(str).str.replace('_', ' ').str.title()
            
            # Create discrete discount bands
            if 'discount_band' not in df.columns and 'discount_pct' in df.columns:
                bins = [-1, 0, 5, 10, 20, 100]
                labels = ['0% (No Discount)', '1% to 5%', '5% to 10%', '10% to 20%', 'Above 20%']
                df['discount_band'] = pd.cut(df['discount_pct'], bins=bins, labels=labels)
            
            # Net profit after shipping cost
            if 'net_profit_after_shipping' not in df.columns and 'profit' in df.columns and 'shipping_cost' in df.columns:
                df['net_profit_after_shipping'] = (df['profit'] - df['shipping_cost']).round(2)
                
            return df
            
    st.error("Dataset 'master_df_cleaned.csv' not found in workspace.")
    return pd.DataFrame()

df = load_dataset()

if df.empty:
    st.stop()

# ── 4. Sidebar Dynamic Filters ────────────────────────────────────────────────
st.sidebar.markdown("### Filter Sales Data")

# Customer Segment
available_segments = ['All'] + sorted([str(x) for x in df['customer_segment'].dropna().unique()])
selected_segment = st.sidebar.selectbox("Customer Segment", available_segments)

# Stock Availability
available_stocks = ['All'] + sorted([str(x) for x in df['stock_availability'].dropna().unique()])
selected_stock = st.sidebar.selectbox("Stock Availability", available_stocks)

# Product Category
available_cats = ['All'] + sorted([str(x) for x in df['clean_category'].dropna().unique()])
selected_cat = st.sidebar.selectbox("Product Category", available_cats)

# Discount Slider
min_disc = float(df['discount_pct'].min())
max_disc = float(df['discount_pct'].max())
discount_range = st.sidebar.slider("Discount % Range", min_disc, max_disc, (min_disc, max_disc))

# Apply Filters
filtered_df = df.copy()

if selected_segment != 'All':
    filtered_df = filtered_df[filtered_df['customer_segment'] == selected_segment]

if selected_stock != 'All':
    filtered_df = filtered_df[filtered_df['stock_availability'] == selected_stock]

if selected_cat != 'All':
    filtered_df = filtered_df[filtered_df['clean_category'] == selected_cat]

filtered_df = filtered_df[
    (filtered_df['discount_pct'] >= discount_range[0]) & 
    (filtered_df['discount_pct'] <= discount_range[1])
]

st.sidebar.markdown(f"**Records Displayed:** {len(filtered_df):,} of {len(df):,}")

# Safety check for empty filter results
if filtered_df.empty:
    st.warning("No records found for the selected filter combination. Please broaden your selection.")
    st.stop()

# ── 5. Main Header Bar (Clean White Text) ─────────────────────────────────────
st.markdown('<div class="main-title">E-Commerce RetailTech Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Business Objective 3: Optimize Pricing, Discounting & Shipping for Profitable Growth</div>', unsafe_allow_html=True)

# ── 6. KPI Summary Metrics Strip ──────────────────────────────────────────────
tot_rev = filtered_df['revenue'].sum()
tot_profit = filtered_df['profit'].sum()
tot_orders = len(filtered_df)
avg_margin = (tot_profit / tot_rev * 100) if tot_rev > 0 else 0
tot_shipping = filtered_df['shipping_cost'].sum()
avg_discount = filtered_df['discount_pct'].mean()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Revenue", f"₹{tot_rev/1e6:.1f}M")
c2.metric("Total Orders", f"{tot_orders:,}")
c3.metric("Net Profit", f"₹{tot_profit/1e6:.2f}M")
c4.metric("Avg Margin", f"{avg_margin:.1f}%")
c5.metric("Avg Discount", f"{avg_discount:.1f}%")
c6.metric("Total Shipping", f"₹{tot_shipping/1e6:.1f}M")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ── Helper function for dark themed figures ───────────────────────────────────
def create_dark_fig(figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#111726')
    ax.set_facecolor('#111726')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1E293B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax

# ── 7. Objective 3 Analysis Tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Category Profitability",
    "2. Discount Elasticity & Margin Cliff",
    "3. Shipping Cost Burden",
    "4. Stock Availability Strategy",
    "5. Correlation & Executive Summary"
])

# ── TAB 1: Category Profitability ─────────────────────────────────────────────
with tab1:
    st.markdown("#### Category Profitability & Margin Contribution")
    st.markdown("Identifies high-profit product categories versus low-margin segments.")
    
    cat_agg = filtered_df.groupby('clean_category', observed=False).agg(
        profit=('profit', 'sum'),
        revenue=('revenue', 'sum'),
        orders=('order_item_id', 'count')
    ).reset_index()
    
    if not cat_agg.empty:
        cat_agg['margin_pct'] = (cat_agg['profit'] / cat_agg['revenue'] * 100).round(2)
        cat_agg['profit_million'] = (cat_agg['profit'] / 1e6).round(2)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            top_p = cat_agg.sort_values(by='profit', ascending=False).head(10)
            fig1, ax1 = create_dark_fig(figsize=(8, 4.5))
            y_pos = np.arange(len(top_p))
            ax1.barh(y_pos, top_p['profit_million'], color='#2563EB', height=0.6)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_p['clean_category'], color='#CBD5E1', fontsize=9)
            ax1.invert_yaxis()
            ax1.set_title("Top 10 Categories by Total Profit (₹ Millions)", color='#FFFFFF', fontsize=12, fontweight='bold', pad=10)
            ax1.set_xlabel("Total Profit (₹ M)", color='#94A3B8', fontsize=10)
            ax1.grid(axis='x', color='#1E293B', alpha=0.8, linestyle='--')
            st.pyplot(fig1)
            plt.close(fig1)
            
        with col_right:
            top_m = cat_agg.sort_values(by='margin_pct', ascending=False).head(10)
            fig2, ax2 = create_dark_fig(figsize=(8, 4.5))
            y_pos2 = np.arange(len(top_m))
            ax2.barh(y_pos2, top_m['margin_pct'], color='#10B981', height=0.6)
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(top_m['clean_category'], color='#CBD5E1', fontsize=9)
            ax2.invert_yaxis()
            ax2.set_title("Top 10 Categories by Highest Profit Margin (%)", color='#FFFFFF', fontsize=12, fontweight='bold', pad=10)
            ax2.set_xlabel("Profit Margin (%)", color='#94A3B8', fontsize=10)
            ax2.grid(axis='x', color='#1E293B', alpha=0.8, linestyle='--')
            st.pyplot(fig2)
            plt.close(fig2)
            
        st.markdown("""
        <div class="takeaway-card">
            <b>Viva Talking Point & Business Takeaway:</b><br>
            Categories like <b>Sports Leisure</b>, <b>Computers Accessories</b>, and <b>Health Beauty</b> drive over 50% of total company profit with strong margins (>25%).<br>
            <b>Strategic Recommendation:</b> Allocate marketing budgets and premium homepage placement to these top categories. For bottom-margin categories, renegotiate procurement terms with suppliers.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No category data available under current filter selection.")

# ── TAB 2: Discount Elasticity ────────────────────────────────────────────────
with tab2:
    st.markdown("#### Discount Band Elasticity & The 10% Margin Cliff")
    st.markdown("Analyzes how discount depth influences order volume and margin erosion.")
    
    band_agg = filtered_df.groupby('discount_band', observed=False).agg(
        orders=('order_item_id', 'count'),
        revenue=('revenue', 'sum'),
        profit=('profit', 'sum')
    ).reset_index()
    
    if not band_agg.empty:
        band_agg['overall_margin_pct'] = (band_agg['profit'] / band_agg['revenue'] * 100).round(2)
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            fig3, ax3 = create_dark_fig(figsize=(8, 4.2))
            x_pos = np.arange(len(band_agg))
            ax3.bar(x_pos, band_agg['orders'], color='#2563EB', width=0.55)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(band_agg['discount_band'], color='#CBD5E1', fontsize=8.5)
            ax3.set_title("Order Volume by Discount Band", color='#FFFFFF', fontsize=12, fontweight='bold', pad=10)
            ax3.set_ylabel("Order Count", color='#94A3B8', fontsize=10)
            ax3.grid(axis='y', color='#1E293B', alpha=0.8, linestyle='--')
            st.pyplot(fig3)
            plt.close(fig3)
            
        with col_d2:
            fig4, ax4 = create_dark_fig(figsize=(8, 4.2))
            ax4.plot(range(len(band_agg)), band_agg['overall_margin_pct'], 
                     marker='o', color='#EF4444', linewidth=2.2, markersize=7)
            ax4.set_xticks(range(len(band_agg)))
            ax4.set_xticklabels(band_agg['discount_band'], color='#CBD5E1', fontsize=8.5)
            ax4.set_title("Profit Margin (%) Collapse Across Discount Bands", color='#FFFFFF', fontsize=12, fontweight='bold', pad=10)
            ax4.set_ylabel("Overall Margin (%)", color='#94A3B8', fontsize=10)
            ax4.axvline(x=2.5, color='#94A3B8', linestyle='--', alpha=0.6)
            ax4.text(2.6, ax4.get_ylim()[1]*0.75, '10% Critical Limit', color='#EF4444', fontweight='bold', fontsize=9.5)
            ax4.grid(True, color='#1E293B', alpha=0.8, linestyle='--')
            st.pyplot(fig4)
            plt.close(fig4)
            
        st.markdown("""
        <div class="alert-card">
            <b>Critical Margin Warning (Viva Insight):</b><br>
            Discounts between <b>0% and 10%</b> maintain healthy profit margins (~25% to 30%). Beyond <b>10% discount</b>, profit margins collapse below 12%, and aggressive discounts (>20%) result in net operational losses.<br>
            <b>Strategic Recommendation:</b> Enforce an organizational ceiling of <b>10% on promotional discounts</b>. Reserve deeper discounts strictly for clearing slow-moving stock.
        </div>
        """, unsafe_allow_html=True)

# ── TAB 3: Shipping Cost Burden ───────────────────────────────────────────────
with tab3:
    st.markdown("#### Impact of Shipping Costs on Net Profitability")
    st.markdown("Compares Gross Profit before shipping against Net Profit after shipping costs are deducted.")
    
    top8_cats = filtered_df['clean_category'].value_counts().head(8).index
    ship_df = filtered_df[filtered_df['clean_category'].isin(top8_cats)].groupby('clean_category', observed=False).agg(
        gross_profit=('profit', 'sum'),
        shipping_cost=('shipping_cost', 'sum'),
        net_profit=('net_profit_after_shipping', 'sum')
    ).reset_index()
    
    if not ship_df.empty:
        ship_df['gross_million'] = ship_df['gross_profit'] / 1e6
        ship_df['net_million'] = ship_df['net_profit'] / 1e6
        ship_df = ship_df.sort_values(by='gross_million', ascending=False)
        
        fig5, ax5 = create_dark_fig(figsize=(9.5, 4.2))
        x_indices = np.arange(len(ship_df))
        w = 0.35
        ax5.bar(x_indices - w/2, ship_df['gross_million'], width=w, label='Gross Profit (Before Shipping)', color='#2563EB')
        ax5.bar(x_indices + w/2, ship_df['net_million'], width=w, label='Net Profit (After Shipping)', color='#10B981')
        ax5.set_title("Gross Profit vs. Net Profit (After Shipping) — Top 8 Categories", color='#FFFFFF', fontsize=12, fontweight='bold', pad=10)
        ax5.set_ylabel("Profit (₹ Millions)", color='#94A3B8', fontsize=10)
        ax5.set_xticks(x_indices)
        ax5.set_xticklabels(ship_df['clean_category'], rotation=22, ha='right', color='#CBD5E1', fontsize=8.5)
        ax5.legend(facecolor='#1E293B', edgecolor='none', labelcolor='#FFFFFF', fontsize=9)
        ax5.grid(axis='y', color='#1E293B', alpha=0.8, linestyle='--')
        st.pyplot(fig5)
        plt.close(fig5)
        
        st.markdown("""
        <div class="takeaway-card">
            <b>Viva Talking Point & Business Takeaway:</b><br>
            Shipping costs absorb approximately <b>10% to 18%</b> of gross profit across categories. For low-ticket items, shipping fees can completely wipe out the net margin.<br>
            <b>Strategic Recommendation:</b> Introduce a <b>₹999 minimum order threshold for free shipping</b> to encourage basket bundling.
        </div>
        """, unsafe_allow_html=True)

# ── TAB 4: Stock Availability Strategy ────────────────────────────────────────
with tab4:
    st.markdown("#### Stock Availability & Inventory Pricing Strategy")
    st.markdown("Evaluates how stock levels influence discounting behavior and profit margins.")
    
    stock_agg = filtered_df.groupby('stock_availability', observed=False).agg(
        orders=('order_item_id', 'count'),
        avg_price=('unit_price', 'mean'),
        avg_discount=('discount_pct', 'mean'),
        avg_margin=('profit_margin_pct', 'mean')
    ).reset_index()
    
    if not stock_agg.empty:
        col_s1, col_s2 = st.columns([1, 1.2])
        
        with col_s1:
            st.markdown("<p style='color:#CBD5E1; font-weight:600; font-size:13px;'>Stock Performance Summary</p>", unsafe_allow_html=True)
            stock_display = stock_agg.rename(columns={
                'stock_availability': 'Stock Status',
                'orders': 'Orders',
                'avg_price': 'Avg Price (₹)',
                'avg_discount': 'Avg Discount (%)',
                'avg_margin': 'Avg Margin (%)'
            })
            st.dataframe(stock_display.round(2))
            
        with col_s2:
            fig6, ax6 = create_dark_fig(figsize=(7, 3.8))
            x_pos_s = np.arange(len(stock_agg))
            ax6.bar(x_pos_s, stock_agg['avg_margin'], color=['#10B981', '#2563EB', '#F59E0B'], width=0.5)
            ax6.set_xticks(x_pos_s)
            ax6.set_xticklabels(stock_agg['stock_availability'], color='#CBD5E1', fontsize=9)
            ax6.set_title("Average Profit Margin (%) by Stock Status", color='#FFFFFF', fontsize=12, fontweight='bold', pad=10)
            ax6.set_ylabel("Margin (%)", color='#94A3B8', fontsize=10)
            ax6.grid(axis='y', color='#1E293B', alpha=0.8, linestyle='--')
            st.pyplot(fig6)
            plt.close(fig6)
            
        st.markdown("""
        <div class="takeaway-card">
            <b>Viva Talking Point & Business Takeaway:</b><br>
            <b>Preorder</b> and <b>Low Stock</b> items carry the lowest average discounts and highest profit margins (~28% to 30%) due to supply scarcity.<br>
            <b>Strategic Recommendation:</b> Implement <b>Dynamic Scarcity Pricing</b>—automatically disable discounts on items with low inventory.
        </div>
        """, unsafe_allow_html=True)

# ── TAB 5: Correlation Matrix & Summary ────────────────────────────────────────
with tab5:
    st.markdown("#### Correlation Matrix & Strategic Viva Summary")
    
    corr_cols = ['unit_price', 'discount_pct', 'shipping_cost', 'quantity', 'revenue', 'profit', 'profit_margin_pct']
    corr_labels = ['Unit Price', 'Discount %', 'Shipping Cost', 'Quantity', 'Revenue', 'Profit', 'Margin %']
    corr_df = filtered_df[corr_cols].corr().round(2)
    
    # Custom elegant colormap: Coral Red → Dark Slate → Royal Blue
    from matplotlib.colors import LinearSegmentedColormap
    exec_cmap = LinearSegmentedColormap.from_list(
        'exec_corr',
        ['#EF4444', '#182234', '#2563EB'],
        N=256
    )
    
    # Center the matrix with columns to keep it compact
    col_pad_l, col_corr, col_pad_r = st.columns([0.8, 2.4, 0.8])
    
    with col_corr:
        fig7, ax7 = create_dark_fig(figsize=(5.2, 4.0))
        im = ax7.imshow(corr_df.values, cmap=exec_cmap, vmin=-1, vmax=1, aspect='equal')
        ax7.set_xticks(np.arange(len(corr_cols)))
        ax7.set_yticks(np.arange(len(corr_cols)))
        ax7.set_xticklabels(corr_labels, rotation=38, ha='right', color='#CBD5E1', fontsize=7.5)
        ax7.set_yticklabels(corr_labels, color='#CBD5E1', fontsize=7.5)
        
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_df.iloc[i, j]
                ax7.text(j, i, f"{val:.2f}", ha='center', va='center',
                         color='#FFFFFF' if abs(val) > 0.35 else '#94A3B8', fontsize=7.5, fontweight='bold')
        
        # Colorbar styled for dark theme
        cbar = fig7.colorbar(im, ax=ax7, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.ax.tick_params(colors='#94A3B8', labelsize=7)
        cbar.outline.set_edgecolor('#1E293B')
        
        ax7.set_title("Pricing, Discount, Shipping & Profit Correlation", color='#FFFFFF', fontsize=11, fontweight='bold', pad=10)
        fig7.tight_layout(pad=1.5)
        st.pyplot(fig7)
        plt.close(fig7)
    
    st.markdown("<p style='color:#FFFFFF; font-weight:700; font-size:16px; margin-top:20px;'>Executive Recommendations Table</p>", unsafe_allow_html=True)
    summary_table = pd.DataFrame({
        "Strategic Focus": [
            "Top Profit Categories",
            "Optimal Discount Limit",
            "Shipping Cost Burden",
            "Stock Availability Strategy"
        ],
        "Key Data Finding": [
            "Top 10 categories generate >60% of total profit with margins >25%.",
            "Margins remain healthy up to 10% discount, then collapse below 12%.",
            "Shipping consumes 10–18% of gross profit, heavily hitting low-ticket items.",
            "Low Stock and Preorder items achieve the highest margins (28–30%)."
        ],
        "Actionable Business Recommendation": [
            "Allocate majority of ad spend to top categories; renegotiate vendor terms for bottom categories.",
            "Enforce strict 10% discount ceiling on promotional campaigns.",
            "Set ₹999 minimum order threshold for free shipping.",
            "Adopt Dynamic Scarcity Pricing (reduce discounts as stock runs low)."
        ]
    })
    st.table(summary_table)

# ── 8. Mandatory Footer ───────────────────────────────────────────────────────
st.markdown('<div class="footer-bar"> Analysis , by Afifa & Arshad</div>', unsafe_allow_html=True)
