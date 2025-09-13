"""
Reads one csv file and creates dashbaord of it

-> Need to change to read all csv files in GENERATED_CSV_FILES_DIR, and option to choose on the sidebar
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
from asg_expense_tracker.jio_payments_bank_reader import JioPaymentsBankReader
from asg_expense_tracker import GENERATED_CSV_FILES_DIR

# Page config
st.set_page_config( page_title="ASG Expense Tracker Dashboard",  page_icon="💰",  layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar-content {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the expense data"""
    try:
        csv_file_path = os.path.join(GENERATED_CSV_FILES_DIR, "jio_2025_08_29_to_2025_09_13.csv")
        jio_df = pd.read_csv(csv_file_path, parse_dates=['Date'])
        return jio_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 ASG Expense Tracker Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading your expense data..."):
        df = load_data()
    
    if df.empty:
        st.error("Unable to load data. Please check if the PDF file exists.")
        return
    
    # Sidebar filters
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.header("📊 Dashboard Filters")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Date'].min().date(), df['Date'].max().date()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )
    
    # Transaction type filter
    transaction_types = st.sidebar.multiselect(
        "Transaction Types",
        options=df['Transaction_Type'].unique(),
        default=df['Transaction_Type'].unique()
    )
    
    # Amount range filter
    min_amount, max_amount = st.sidebar.slider(
        "Amount Range (₹)",
        min_value=0,
        max_value=int(df[['Withdrawal', 'Deposit']].max().max()),
        value=(0, int(df[['Withdrawal', 'Deposit']].max().max())),
        step=100
    )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data
    if len(date_range) == 2:
        filtered_df = df[
            (df['Date'].dt.date >= date_range[0]) & 
            (df['Date'].dt.date <= date_range[1]) &
            (df['Transaction_Type'].isin(transaction_types)) &
            (((df['Withdrawal'].fillna(0) >= min_amount) & (df['Withdrawal'].fillna(0) <= max_amount)) |
             ((df['Deposit'].fillna(0) >= min_amount) & (df['Deposit'].fillna(0) <= max_amount)))
        ]
    else:
        filtered_df = df[df['Transaction_Type'].isin(transaction_types)]
    
    # Key Metrics Row
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    MONTH = "Sept"

    current_date = date(2025, 9, 14)  # Current date
    days_left = 30 - current_date.day
    total_expenses = filtered_df['Withdrawal'].sum()
    total_income = filtered_df['Deposit'].sum()
    avg_transaction = (total_expenses + total_income) / len(filtered_df) if len(filtered_df) > 0 else 0
    transaction_count = len(filtered_df)
    # Find salary date (largest deposit) and current balance
    salary_transactions = df[df['Deposit'] > 40000]
    if not salary_transactions.empty:
        latest_salary = salary_transactions.loc[salary_transactions['Date'].idxmax()]
        salary_date = latest_salary['Date'].strftime('%B %d, %Y')
        salary_amount = latest_salary['Deposit']
    else:
        salary_date = "No salary found"
        salary_amount = 0
    
    # Get current balance (latest balance in the data)
    if 'Balance' in df.columns:
        latest_idx = df['Date'].idxmax()
        current_balance = df.loc[latest_idx, 'Balance']
        balance_date = str(df.loc[latest_idx, 'Date']).split(' ')[0]  # Get date part only
    else:
        current_balance = 0
        balance_date = "N/A"


    with col1: st.metric(f"📅 Days Left in {MONTH}", days_left)
    with col2: st.metric("💰 Last Salary", f"₹{salary_amount:,.2f}", delta=f"Received on {salary_date}", delta_color="off")
    with col3: st.metric("💸 Total Expenses", f"₹{total_expenses:,.2f}")
    with col4: st.metric("💰 Total Income", f"₹{total_income:,.2f}")
    with col5: st.metric("🏦 Current Balance", f"₹{current_balance:,.2f}", delta=f"As of {balance_date}", delta_color="off")
    with col6: st.metric("🔄 Avg Transaction", f"₹{avg_transaction:,.2f}")
    with col7: st.metric("📈 Total Transactions", transaction_count)

    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📊 Analytics", "🏪 Merchants", "⏰ Time Analysis", "🎯 Custom Views"])
    
    with tab1:        
        col1, col2 = st.columns(2)
        with col1:
            # Combined Daily Trends - Expenses & Credits
            daily_spending = filtered_df.groupby(filtered_df['Date'].dt.date)['Withdrawal'].sum().fillna(0)
            daily_credits  = filtered_df.groupby(filtered_df['Date'].dt.date)['Deposit'].sum().fillna(0)
            
            fig_combined = go.Figure()
            fig_combined.add_trace(go.Scatter(
                x=daily_spending.index,
                y=daily_spending.values,
                mode='lines+markers',
                name='Daily Expenses',
                line=dict(color='#DC143C', width=3),
                marker=dict(color='#DC143C', size=4)
            ))
            fig_combined.add_trace(go.Scatter(
                x=daily_credits.index,
                y=daily_credits.values,
                mode='lines+markers', 
                name='Daily Credits',
                line=dict(color='#32CD32', width=3),
                marker=dict(color='#32CD32', size=4)
            ))
            fig_combined.update_layout(
                title="Daily Expenses vs Credits Trend",
                xaxis_title="Date",
                yaxis_title="Amount (₹)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_combined, use_container_width=True)
        
        with col2:
            # Income vs Expenses
            monthly_data = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).agg({'Withdrawal': 'sum', 'Deposit': 'sum'}).fillna(0)
            fig_monthly = go.Figure(data=[
                go.Bar(name='Expenses', x=monthly_data.index.astype(str), y=monthly_data['Withdrawal'], marker_color='#ff7f0e'),
                go.Bar(name='Income', x=monthly_data.index.astype(str), y=monthly_data['Deposit'], marker_color='#2ca02c')
            ])
            fig_monthly.update_layout(
                title="Monthly Income vs Expenses",
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Net daily flow (Credits - Expenses)
            daily_spending = filtered_df.groupby(filtered_df['Date'].dt.date)['Withdrawal'].sum().fillna(0)
            daily_credits = filtered_df.groupby(filtered_df['Date'].dt.date)['Deposit'].sum().fillna(0)
            daily_net_flow = daily_credits - daily_spending
            
            fig_net_flow = px.bar(
                x=daily_net_flow.index,
                y=daily_net_flow.values,
                title="Daily Net Flow (Credits - Expenses)",
                labels={'x': 'Date', 'y': 'Net Amount (₹)'},
                color=daily_net_flow.values,
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig_net_flow.update_traces(
                hovertemplate="<b>%{x}</b><br>Net Flow: ₹%{y:,.2f}<extra></extra>"
            )
            fig_net_flow.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_net_flow, use_container_width=True)
        
        with col2:
            # Daily balance trend - BLUE COLOR
            if 'Balance' in filtered_df.columns:
                daily_balance = filtered_df.groupby(filtered_df['Date'].dt.date)['Balance'].last()
                fig_daily_balance = px.line(
                    x=daily_balance.index, 
                    y=daily_balance.values,
                    title="Daily Balance Trend",
                    labels={'x': 'Date', 'y': 'Balance (₹)'}
                )
                fig_daily_balance.update_traces(line=dict(color='#1E90FF', width=3))  # Dodger blue
                fig_daily_balance.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_daily_balance, use_container_width=True)
            else:
                st.info("Balance data not available for trend analysis")
        
        # Category breakdown
        if 'Merchant' in filtered_df.columns:
            st.subheader("💳 Top Spending Categories")
            category_spending = filtered_df.groupby('Merchant')['Withdrawal'].sum().fillna(0).sort_values(ascending=False).head(10)
            
            fig_pie = px.pie(
                values=category_spending.values, 
                names=category_spending.index,
                title="Top 10 Merchants by Spending"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.header("📊 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekday vs Weekend spending
            filtered_df['Weekday'] = filtered_df['Date'].dt.day_name()
            filtered_df['Is_Weekend'] = filtered_df['Date'].dt.weekday >= 5
            
            weekend_spending = filtered_df.groupby('Is_Weekend')['Withdrawal'].sum().fillna(0)
            weekend_labels = ['Weekday', 'Weekend']
            
            fig_weekend = px.bar(
                x=weekend_labels,
                y=[weekend_spending[False], weekend_spending[True]],
                title="Weekday vs Weekend Spending",
                color=[weekend_spending[False], weekend_spending[True]],
                color_continuous_scale='Viridis'
            )
            fig_weekend.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_weekend, use_container_width=True)
        
        with col2:
            # Spending velocity (rolling average)
            daily_spending = filtered_df.groupby(filtered_df['Date'].dt.date)['Withdrawal'].sum().fillna(0)
            rolling_avg = daily_spending.rolling(window=7, min_periods=1).mean()
            
            fig_velocity = go.Figure()
            fig_velocity.add_trace(go.Scatter(
                x=daily_spending.index,
                y=daily_spending.values,
                mode='lines',
                name='Daily Spending',
                line=dict(color='lightblue', width=1),
                opacity=0.7
            ))
            fig_velocity.add_trace(go.Scatter(
                x=rolling_avg.index,
                y=rolling_avg.values,
                mode='lines',
                name='7-Day Average',
                line=dict(color='darkblue', width=3)
            ))
            fig_velocity.update_layout(
                title="Spending Velocity (7-Day Rolling Average)",
                height=400
            )
            st.plotly_chart(fig_velocity, use_container_width=True)
    
    with tab3:
        st.header("🏪 Merchant Analysis")
        
        if 'Merchant' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top merchants
                merchant_spending = filtered_df.groupby('Merchant').agg({
                    'Withdrawal': 'sum',
                    'Date': 'count'
                }).fillna(0)
                merchant_spending.columns = ['Total_Spent', 'Transaction_Count']
                merchant_spending = merchant_spending.sort_values('Total_Spent', ascending=False).head(15)
                
                fig_merchants = px.bar(
                    merchant_spending,
                    x=merchant_spending.index,
                    y='Total_Spent',
                    title="Top 15 Merchants by Spending",
                    labels={'x': 'Merchant', 'y': 'Total Spent (₹)'}
                )
                fig_merchants.update_xaxes(tickangle=45)
                fig_merchants.update_layout(height=500)
                st.plotly_chart(fig_merchants, use_container_width=True)
            
            with col2:
                # Merchant frequency vs amount
                merchant_analysis = filtered_df.groupby('Merchant').agg({
                    'Withdrawal': ['sum', 'mean', 'count']
                }).fillna(0)
                merchant_analysis.columns = ['Total_Spent', 'Avg_Transaction', 'Frequency']
                merchant_analysis = merchant_analysis[merchant_analysis['Frequency'] > 1].head(20)
                
                fig_bubble = px.scatter(
                    merchant_analysis,
                    x='Frequency',
                    y='Avg_Transaction',
                    size='Total_Spent',
                    hover_name=merchant_analysis.index,
                    title="Merchant Analysis: Frequency vs Average Transaction",
                    labels={'x': 'Number of Transactions', 'y': 'Average Transaction (₹)'}
                )
                fig_bubble.update_layout(height=500)
                st.plotly_chart(fig_bubble, use_container_width=True)
            
            # Merchant details table
            st.subheader("📋 Detailed Merchant Statistics")
            merchant_details = filtered_df.groupby('Merchant').agg({
                'Withdrawal': ['sum', 'mean', 'count', 'std'],
                'Date': ['min', 'max']
            }).fillna(0)
            merchant_details.columns = ['Total_Spent', 'Avg_Transaction', 'Frequency', 'Std_Dev', 'First_Transaction', 'Last_Transaction']
            merchant_details = merchant_details.sort_values('Total_Spent', ascending=False)
            
            # Format currency columns
            for col in ['Total_Spent', 'Avg_Transaction', 'Std_Dev']:
                merchant_details[col] = merchant_details[col].apply(lambda x: f"₹{x:,.2f}")
            
            st.dataframe(merchant_details, use_container_width=True)
    
    with tab4:
        st.header("⏰ Time-based Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week analysis
            weekday_spending = filtered_df.groupby(filtered_df['Date'].dt.day_name())['Withdrawal'].sum().fillna(0)
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_spending = weekday_spending.reindex(day_order, fill_value=0)
            
            fig_weekday = px.line_polar(
                r=weekday_spending.values,
                theta=weekday_spending.index,
                line_close=True,
                title="Weekly Spending Pattern"
            )
            fig_weekday.update_layout(height=400)
            st.plotly_chart(fig_weekday, use_container_width=True)
        with col2:
            # Calendar view
            calendar_data = filtered_df.groupby(filtered_df['Date'].dt.date)['Withdrawal'].sum().fillna(0)
            
            if len(calendar_data) > 0:
                # Create a DataFrame for the calendar scatter plot
                calendar_df = pd.DataFrame({'Date': calendar_data.index,'Amount': calendar_data.values, 'Y': [1] * len(calendar_data)})
                fig_calendar = px.scatter(calendar_df, x='Date', y='Y', size='Amount', 
                                          hover_data={'Y': False, 'Amount': ':,.2f'}, title="Daily Spending Calendar (Size = Amount)")
                fig_calendar.update_traces(hovertemplate="<b>%{x}</b><br>Spent: ₹%{customdata[0]:,.2f}<extra></extra>")
                fig_calendar.update_layout(yaxis=dict(visible=False), height=300, showlegend=False)
                st.plotly_chart(fig_calendar, use_container_width=True)

    
    with tab5:
        st.header("🎯 Custom Analysis Views")
        
        # Custom chart selector
        chart_type = st.selectbox(
            "Select Custom Chart Type",
            ["Transaction Timeline", "Spending Distribution", "Balance Tracker", "Transaction Size Analysis"]
        )
        
        if chart_type == "Transaction Timeline":
            fig_timeline = px.scatter(
                filtered_df,
                x='Date',
                y='Withdrawal',
                color='Transaction_Type',
                size='Withdrawal',
                hover_data=['Merchant'] if 'Merchant' in filtered_df.columns else None,
                title="Transaction Timeline"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        elif chart_type == "Spending Distribution":
            fig_dist = px.histogram(
                filtered_df[filtered_df['Withdrawal'] > 0],
                x='Withdrawal',
                nbins=30,
                title="Spending Amount Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        elif chart_type == "Balance Tracker":
            if 'Balance' in filtered_df.columns:
                fig_balance = px.line(
                    filtered_df.sort_values('Date'),
                    x='Date',
                    y='Balance',
                    title="Account Balance Over Time"
                )
                fig_balance.update_traces(line=dict(color='green', width=2))
                st.plotly_chart(fig_balance, use_container_width=True)
            else:
                st.info("Balance information not available in the dataset.")
        
        elif chart_type == "Transaction Size Analysis":
            # Categorize transactions by size
            conditions = [
                (filtered_df['Withdrawal'] <= 100),
                (filtered_df['Withdrawal'] <= 500),
                (filtered_df['Withdrawal'] <= 1000),
                (filtered_df['Withdrawal'] <= 5000),
                (filtered_df['Withdrawal'] > 5000)
            ]
            choices = ['Small (≤₹100)', 'Medium (₹101-500)', 'Large (₹501-1000)', 'Very Large (₹1001-5000)', 'Huge (>₹5000)']
            filtered_df['Size_Category'] = np.select(conditions, choices, default='Unknown')
            
            size_analysis = filtered_df.groupby('Size_Category').agg({
                'Withdrawal': ['count', 'sum', 'mean']
            }).fillna(0)
            size_analysis.columns = ['Count', 'Total', 'Average']
            
            fig_size = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Transaction Count by Size', 'Total Amount by Size'),
                specs=[[{'type': 'domain'}, {'type': 'domain'}]]
            )
            
            fig_size.add_trace(go.Pie(
                labels=size_analysis.index,
                values=size_analysis['Count'],
                name="Count"
            ), 1, 1)
            
            fig_size.add_trace(go.Pie(
                labels=size_analysis.index,
                values=size_analysis['Total'],
                name="Total"
            ), 1, 2)
            
            fig_size.update_traces(textposition='inside', textinfo='percent+label')
            fig_size.update_layout(title_text="Transaction Size Analysis")
            st.plotly_chart(fig_size, use_container_width=True)
    
    # # Footer
    # st.markdown("---")
    # st.markdown("### 📊 Dashboard Summary")
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     st.info(f"**Data Range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # with col2:
    #     st.info(f"**Total Records:** {len(df):,} transactions")
    
    # with col3:
    #     st.info(f"**Filtered Records:** {len(filtered_df):,} transactions")

if __name__ == "__main__":
    main()
