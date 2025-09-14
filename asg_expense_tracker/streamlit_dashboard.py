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
from asg_expense_tracker import GENERATED_CSV_FILES_DIR, STORED_PDF_FILES_DIR

# Page config
st.set_page_config( page_title="ASG Expense Tracker Dashboard",  page_icon="💰",  layout="wide", initial_sidebar_state="expanded")
EXPENSES_COLOR = "#ff7f0e"  # Orange color
DEPOSITS_COLOR = "#2ca02c"  # Green color
NEUTRAL_COLOR = "#888888"  # Grey color

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 4rem;
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
        JioPaymentsBankReader(os.path.join(STORED_PDF_FILES_DIR, "00272171124045614-Sep-2025 14_38_00.pdf"))
        csv_file_path = os.path.join(GENERATED_CSV_FILES_DIR, "jio_2025_08_29_to_2025_09_14.csv")
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

    current_date = datetime.now()
    days_left = (date(current_date.year, current_date.month, 1) + timedelta(days=32)).replace(day=1) - current_date.date()
    days_left = days_left.days
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
        # Sort by date and then by index to get the chronologically last transaction
        df_sorted = df.sort_values(['Date']).reset_index(drop=True)
        current_balance = df_sorted.iloc[-1]['Balance']  # Get the very last balance
        balance_date = str(df_sorted.iloc[-1]['Date']).split(' ')[0]  # Get date part only
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
    tab1, tab2 = st.tabs(["JioPB", "HDFC"])
    
    with tab1:        
        col1, col2 = st.columns(2)
        # Daily Income vs Expenses
        with col1:
            daily_expenses = filtered_df.groupby(filtered_df['Date'].dt.date)['Withdrawal'].sum().fillna(0)
            daily_income   = filtered_df.groupby(filtered_df['Date'].dt.date)['Deposit'].sum().fillna(0)

            fig_combined = go.Figure()
            fig_combined.add_trace(go.Scatter(
                x=daily_expenses.index,
                y=daily_expenses.values,
                mode='lines+markers',
                name='Daily Expenses',
                line=dict(color=EXPENSES_COLOR, width=3),
                marker=dict(color=EXPENSES_COLOR, size=4)
            ))
            fig_combined.add_trace(go.Scatter(
                x=daily_income.index,
                y=daily_income.values,
                mode='lines+markers', 
                name='Daily Income',
                line=dict(color=DEPOSITS_COLOR, width=3),
                marker=dict(color=DEPOSITS_COLOR, size=4)
            ))
            fig_combined.update_layout(
                title="Daily Income vs Expenses",
                xaxis_title="Date",
                yaxis_title="Amount (₹)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_combined, width=True)
        # Monthly ncome vs Expenses
        with col2:
            monthly_data = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).agg({'Withdrawal': 'sum', 'Deposit': 'sum'}).fillna(0)
            fig_monthly = go.Figure(data=[
                go.Bar(name='Monthly Expenses', x=monthly_data.index.astype(str), y=monthly_data['Withdrawal'], marker_color=EXPENSES_COLOR),
                go.Bar(name='Monthly Income', x=monthly_data.index.astype(str), y=monthly_data['Deposit'], marker_color=DEPOSITS_COLOR)
            ])
            fig_monthly.update_layout(
                title="Monthly Income vs Expenses",
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_monthly, width=True)
        
        col1, col2 = st.columns(2)
        # Net daily flow (Credits - Expenses)
        with col1:
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
            st.plotly_chart(fig_net_flow, width=True)
        # Balance trend per transaction with day regions
        with col2:
            if 'Balance' in filtered_df.columns:
                # Sort by date to ensure chronological order
                sorted_df = filtered_df.copy()
                fig_balance = px.line(
                    x=range(len(sorted_df)), 
                    y=sorted_df['Balance'],
                    title="Balance per Transaction (with Daily Regions)",
                    labels={'x': 'Transaction Number', 'y': 'Balance (₹)'}
                )
                fig_balance.update_traces(
                    line=dict(color='#1E90FF', width=3),
                    marker=dict(color='#1E90FF', size=6),
                    mode='lines+markers',
                    hovertemplate="<b>Transaction %{x}</b><br>Balance: ₹%{y:,.2f}<extra></extra>"
                )
                
                # Add vertical regions for each day
                sorted_df['Date_only'] = sorted_df['Date'].dt.date
                current_day = None
                start_idx = 0
                colors = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)', 
                          'rgba(255,255,0,0.2)', 'rgba(255,0,255,0.2)', 'rgba(0,255,255,0.2)',
                          'rgba(128,128,128,0.2)']  # More visible colors
                color_idx = 0
                
                for idx, current_transaction_date in enumerate(sorted_df['Date_only']):
                    if current_day != current_transaction_date:
                        if current_day is not None:
                            # Add vertical rectangle for previous day
                            fig_balance.add_vrect(
                                x0=start_idx-0.5, x1=idx-0.5,
                                fillcolor=colors[color_idx % len(colors)],
                                opacity=0.4,
                                layer="below",
                                line_width=0
                            )
                            # Add day label annotation
                            fig_balance.add_annotation(
                                x=(start_idx + idx - 1) / 2,  # Center of the region
                                y=sorted_df['Balance'].max() * 1.05,  # Top of chart
                                text=f"{current_day.strftime('%d')}",  # Day name + date
                                showarrow=False,
                                font=dict(size=17, color="black"),
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="gray",
                                borderwidth=1
                            )
                            color_idx += 1
                        current_day = current_transaction_date
                        start_idx = idx
                
                # Add rectangle and annotation for the last day
                if current_day is not None:
                    fig_balance.add_vrect(
                        x0=start_idx-0.5, x1=len(sorted_df)-0.5,
                        fillcolor=colors[color_idx % len(colors)],
                        opacity=0.4,
                        layer="below",
                        line_width=0
                    )
                    fig_balance.add_annotation(
                        x=(start_idx + len(sorted_df) - 1) / 2,  # Center of the region
                        y=sorted_df['Balance'].max() * 1.05,  # Top of chart
                        text=f"{current_day.strftime('%d')}",  # Day name + date
                        showarrow=False,
                        font=dict(size=17, color="black"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    )
                
                fig_balance.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_balance, width=True)
            else:
                st.info("Balance data not available for trend analysis")
        
        col1, col2 = st.columns(2)
        # Category breakdown
        with col1:
            if 'Merchant' in filtered_df.columns:
                category_spending = filtered_df.groupby('Merchant')['Withdrawal'].sum().fillna(0).sort_values(ascending=False).head(10)
                
                fig_pie = px.pie(values=category_spending.values,  names=category_spending.index, title="Top 10 Merchants by Spending")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, width=True)
        # Merchant details table
        with col2:
            st.subheader("📋 Detailed Merchant Statistics")
            merchant_details = filtered_df.groupby('Merchant').agg({'Withdrawal': ['sum', 'mean', 'count', 'std'], 'Date': ['min', 'max']}).fillna(0)
            merchant_details.columns = ['Total_Spent', 'Avg_Transaction', 'Frequency', 'Std_Dev', 'First_Transaction', 'Last_Transaction']
            merchant_details = merchant_details.sort_values('Total_Spent', ascending=False)
            
            # Format currency columns properly
            for col in ['Total_Spent', 'Avg_Transaction', 'Std_Dev']:
                merchant_details[col] = merchant_details[col].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) and x > 0 else "₹0.00")
            
            # Format date columns
            for col in ['First_Transaction', 'Last_Transaction']:
                merchant_details[col] = merchant_details[col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "")
            
            # Display with proper container width - centered using CSS
            st.dataframe(merchant_details,  width='content', height=400)

        # Complete transaction data table
        st.markdown("**All transactions in the selected date range:**")
        display_df = filtered_df.copy()
        if not display_df.empty:
            # Create a function to style the dataframe (before formatting)
            def color_rows(row):
                styles = [''] * len(row)
                
                # Check if it's a withdrawal (has value in Withdrawal column)
                withdrawal_val = row.get('Withdrawal', 0)
                deposit_val = row.get('Deposit', 0)
                
                if pd.notna(withdrawal_val) and withdrawal_val > 0:
                    styles = [f'color: {EXPENSES_COLOR};'] * len(row)  # Orange for withdrawals
                elif pd.notna(deposit_val) and deposit_val > 0:
                    styles = [f'color: {DEPOSITS_COLOR};'] * len(row)  # Green for deposits

                return styles
        
            # Function to style individual cells (for None values)
            def style_cells(x):
                # Create a DataFrame with the same shape, filled with empty strings
                df_styles = pd.DataFrame('', index=x.index, columns=x.columns)
                
                # Style None/zero values in currency columns
                for col in ['Withdrawal', 'Deposit', 'Balance']:
                    if col in x.columns:
                        # Check for zero values (which will become "None")
                        mask = (x[col] == 0) | pd.isna(x[col])
                        df_styles.loc[mask, col] = f'color: {NEUTRAL_COLOR}; font-style: italic;'  # Grey and italic

                return df_styles
            
            # Apply both row-level and cell-level styling
            styled_df = display_df.style.apply(color_rows, axis=1).apply(style_cells, axis=None)
            
            # Custom formatting functions to handle None display for zeros
            def format_currency(x):
                try:
                    if pd.isna(x) or x is None:
                        return ""  # Show empty for NaN values
                    elif isinstance(x, (int, float)):
                        if x == 0:
                            return None  # Show "None" for zero values
                        else:
                            return f"₹{x:,.2f}"  # Format non-zero values with 2 decimal places
                    else:
                        return str(x)
                except:
                    return ""
            
            def format_date(x):
                try:
                    if pd.notna(x) and hasattr(x, 'strftime'):
                        return x.strftime('%Y-%m-%d')
                    else:
                        return ""
                except:
                    return ""
            
            # Apply formatting
            format_dict = {}
            if 'Withdrawal' in display_df.columns: format_dict['Withdrawal'] = format_currency
            if 'Deposit' in display_df.columns:    format_dict['Deposit'] = format_currency
            if 'Balance' in display_df.columns:    format_dict['Balance'] = format_currency
            if 'Date' in display_df.columns:       format_dict['Date'] = format_date
            styled_df = styled_df.format(format_dict)
            st.dataframe(styled_df, height=400, width='content')
        else:
            st.info("No transactions found for the selected filters.")

        col1, col2 = st.columns(2)
        # Day of week analysis
        with col1:
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
            st.plotly_chart(fig_weekday, width=True)
        # Calendar view
        with col2:
            calendar_data = filtered_df.groupby(filtered_df['Date'].dt.date)['Withdrawal'].sum().fillna(0)
            
            if len(calendar_data) > 0:
                # Create a DataFrame for the calendar scatter plot
                calendar_df = pd.DataFrame({'Date': calendar_data.index,'Amount': calendar_data.values, 'Y': [1] * len(calendar_data)})
                fig_calendar = px.scatter(calendar_df, x='Date', y='Y', size='Amount', 
                                          hover_data={'Y': False, 'Amount': ':,.2f'}, title="Daily Spending Calendar (Size = Amount)")
                fig_calendar.update_traces(hovertemplate="<b>%{x}</b><br>Spent: ₹%{customdata[0]:,.2f}<extra></extra>")
                fig_calendar.update_layout(yaxis=dict(visible=False), height=300, showlegend=False)
                st.plotly_chart(fig_calendar, width=True)


if __name__ == "__main__":
    main()
