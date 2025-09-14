"""
Jio Payments Bank PDF Reader
Reads one given pdf file and creates multiple csv files in GENERATED_CSV_FILES_DIR of the format 
`jio_<start_date>_to_<end_date>.csv`
--> Each csv file has transactions between two salary deposits (credits > 40k)

Jio Payments Bank statements have password ARMA0304
We have statements like this - 
| TRANSACTION DATE | VALUE DATE | NARRATION                                | WITHDRAWALS | DEPOSITS | CLOSING BALANCE |
|------------------|------------|------------------------------------------|-------------|----------|-----------------|
| 30-Aug-2025      | 30-Aug-2025| UPI/DR/524232951252/N BOOKS/YESB/paytmqr | 281.00      | 748.00   | 37,433.61       |
"""

"""
There needs to be another functionality which will read all pdf files in STORED_PDF_FILES_DIR, check if they are 
JIO payment bank stattements or not, and if yes then call this class on them, to create csv files in GENERATED_CSV_FILES_DIR
"""
import pandas as pd
import pdfplumber
import os
import re
from dotenv import load_dotenv

from asg_expense_tracker import GENERATED_CSV_FILES_DIR, STORED_PDF_FILES_DIR

load_dotenv()

def get_or_none(lst, idx):
  if not lst:
    return None
  return lst[idx] if idx < len(lst) else None

class JioPaymentsBankReader:
    def __init__(self, pdf_path, password=None):
        self.pdf_path = pdf_path
        self.password = password or os.getenv("JIO_PASSWORD", None)
        self.date_pattern = r'\d{1,2}-[A-Za-z]{3}-\d{4}' # 13-Aug-2025
        self.date_lines_pattern = rf'^{self.date_pattern} {self.date_pattern}'
        self.amount_pattern = r'([\d,]+\.\d{2})' # matches amounts like 1,234.56
        self.df = pd.DataFrame() 
        self._initialize_dataframe()
        periods = self._clip_dataframe_and_store_to_files()
    
    def _initialize_dataframe(self):
        transactions = []
        try:
          with pdfplumber.open(self.pdf_path, password=self.password) as pdf:
              for page_num, page in enumerate(pdf.pages, 1):
                  page_lines = page.extract_text().split("\n")
                  
                  for i, line in enumerate(page_lines):
                    if re.match(self.date_pattern, line):
                        amounts = re.findall(self.amount_pattern, line)
                        dates = re.findall(self.date_pattern, line)
                        narration = re.sub(self.amount_pattern, '', re.sub(self.date_pattern, '', line)).strip()
                        
                        # Get additional info from next line if available
                        # There may be two lines for narration too, but we are assuming only one for now
                        additional_info = ""
                        if i + 1 < len(page_lines):
                          next_line = page_lines[i + 1].strip()
                          additional_info = next_line.replace('_', '').strip('/')

                        complete_narration = (narration + additional_info).split("/")
                        transaction_type = direction = transaction_id = merchant = bank_code = upi_vpa = comments = None
                        
                        ######### Splitting the narration #########
                        # UPI transaction:
                        # transaction_type/direction/transaction_id/merchant/bank_code/upi_vpa/comments
                        transaction_type = get_or_none(complete_narration, 0)
                        if transaction_type == 'UPI':
                          direction      = get_or_none(complete_narration, 1)
                          direction      = 'DEBIT' if direction == 'DR' else 'CREDIT' if direction == 'CR' else None
                          transaction_id = get_or_none(complete_narration, 2)
                          merchant       = get_or_none(complete_narration, 3)
                          bank_code      = get_or_none(complete_narration, 4)
                          upi_vpa        = get_or_none(complete_narration, 5)
                          comments       = get_or_none(complete_narration, 6)

                        if len(amounts) >= 3 and len(dates) >= 2:
                          transactions.append({
                            'Date': dates[0],
                            'Transaction_Type': transaction_type,
                            'Direction': direction,
                            'Transaction_ID': transaction_id,
                            'Merchant': merchant,
                            'Bank_Code': bank_code,
                            'UPI_VPA': upi_vpa,
                            'Comments': comments,
                            'Withdrawal': float(amounts[0].replace(',', '')),
                            'Deposit': float(amounts[1].replace(',', '')),
                            'Balance': float(amounts[2].replace(',', ''))
                          })
          
          df = pd.DataFrame(transactions)
          df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
          self.df = df
        except Exception as e:
            print(f"Error processing PDF: {e}")

    def _clip_dataframe_and_store_to_files(self) -> list[dict]:
        """
        Find transactions where credit > 40k (salary deposits) and create separate
        DataFrames for each salary period from one salary to the next.
        Stores each period as a separate CSV file.
        """
        if self.df.empty:
            print("No data to process")
            return []
        
        # Find all add expense amount transactions (credits > 40k)
        add_expense_amount_transactions = self.df[self.df['Deposit'] > 40000].copy()
        
        if len(add_expense_amount_transactions) == 0:
            print("No add expense amount transactions found (no deposits > ₹40,000)")
            return [{'dataframe': self.df}]  # Return the entire dataframe if no salary found

        # Sort by date to ensure chronological order
        salary_dates = add_expense_amount_transactions['Date'].tolist()
        
        print(f"Found {len(add_expense_amount_transactions)} 'add expense amount transaction(s)':")
        for idx, row in add_expense_amount_transactions.iterrows():
            print(f"  - {row['Date'].strftime('%Y-%m-%d')}: ₹{row['Deposit']:,.2f}")
        
        clipped_periods = []
        # Create periods between salary dates
        for i in range(len(salary_dates)):
            start_date = salary_dates[i]
            
            # If there's a next salary, end before it; otherwise, include all remaining transactions
            if i + 1 < len(salary_dates):
                end_date = salary_dates[i + 1]
                period_df = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] < end_date)].copy()
                period_name = f"jio_{start_date.strftime('%Y_%m_%d')}_to_{end_date.strftime('%Y_%m_%d')}"
            else: # Last period - from this salary to end of data
                period_df = self.df[self.df['Date'] >= start_date].copy()
                end_date = self.df['Date'].max()
                period_name = f"jio_{start_date.strftime('%Y_%m_%d')}_to_{end_date.strftime('%Y_%m_%d')}"
            
            if not period_df.empty:
                # Calculate period statistics
                total_expenses = period_df['Withdrawal'].sum()
                total_income = period_df['Deposit'].sum()
                net_savings = total_income - total_expenses
                transaction_count = len(period_df)
                
                print(f"\nPeriod {i+1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", end=' ')
                # Store to CSV file
                csv_filename = os.path.join(GENERATED_CSV_FILES_DIR, f"{period_name}.csv")
                os.makedirs(GENERATED_CSV_FILES_DIR, exist_ok=True)
                period_df.to_csv(csv_filename, index=False)
                print(f"  Saved to: {csv_filename}")
                
                clipped_periods.append({
                    'dataframe': period_df,
                    'filename': csv_filename,
                    'period_name': period_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'stats': {
                        'total_income': total_income,
                        'total_expenses': total_expenses,
                        'net_savings': net_savings,
                        'transaction_count': transaction_count
                    }
                })
        
        return clipped_periods

if __name__ == "__main__":
    pdf_name = "00272171124045614-Sep-2025 14_38_00.pdf"
    pdf_path = os.path.join(STORED_PDF_FILES_DIR, pdf_name)
    print(f"PDF path: {pdf_path}")
    PASSWORD = "ARMA0304"
    reader = JioPaymentsBankReader(pdf_path, password=PASSWORD)
    df = reader.df
    print(f"Extracted {len(df)} transactions")
    print(df.head())
    # df.to_csv("JioPaymentsBank.csv", index=False)