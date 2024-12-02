from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Directories
CHARTS_FOLDER = "static/saved_charts"
os.makedirs(CHARTS_FOLDER, exist_ok=True)

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE = "financial_data.db"

def initialize_database():
    """
    Create the database and table if they don't exist.
    """
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                rent REAL NOT NULL,
                utilities REAL NOT NULL,
                groceries REAL NOT NULL,
                other_expenses REAL NOT NULL,
                earnings REAL NOT NULL,
                total_expenses REAL NOT NULL,
                savings REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()

initialize_database()

# Visualization function for static charts
def visualize_investment_allocation(investment_amount, filename):
    """
    Create static visualizations for investment allocation and post-investment values.
    """
    data = {
        'Company': ['Apple', 'Amazon', 'Microsoft', 'Google', 'Tesla'],
        'Expected Return (%)': [12, 15, 10, 11, 20],
        'Risk (Volatility %)': [18, 22, 15, 17, 30],
    }
    df = pd.DataFrame(data)
    df['Weight'] = df['Expected Return (%)'] / df['Expected Return (%)'].sum()
    df['Investment ($)'] = df['Weight'] * investment_amount
    df['Post-Investment Amount ($)'] = df['Investment ($)'] * (
        1 + (df['Expected Return (%)'] - df['Risk (Volatility %)']) / 100
    )

    # Create Pie Chart
    pie_path = os.path.join(CHARTS_FOLDER, f"{filename}_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(df['Investment ($)'], labels=df['Company'], autopct=lambda p: f'${p * investment_amount / 100:,.2f}', startangle=140)
    plt.title('Investment Allocation by Company')
    plt.savefig(pie_path)
    plt.close()

    # Create Bar Chart for Expected Returns and Investment
    bar_path = os.path.join(CHARTS_FOLDER, f"{filename}_bar.png")
    plt.figure(figsize=(10, 6))
    plt.bar(df['Company'], df['Expected Return (%)'], color='skyblue', label='Expected Return (%)')
    plt.bar(df['Company'], df['Risk (Volatility %)'], bottom=df['Expected Return (%)'], color='lightcoral', label='Risk (Volatility %)')
    plt.xlabel('Company')
    plt.ylabel('Values (%)')
    plt.title('Expected Returns and Risks by Company')
    plt.legend()
    plt.savefig(bar_path)
    plt.close()

    # Create Line Chart for Cumulative Investment
    cumulative_path = os.path.join(CHARTS_FOLDER, f"{filename}_line.png")
    plt.figure(figsize=(10, 6))
    df['Cumulative Investment ($)'] = df['Investment ($)'].cumsum()
    plt.plot(df['Company'], df['Cumulative Investment ($)'], marker='o', color='green')
    plt.xlabel('Company')
    plt.ylabel('Cumulative Investment ($)')
    plt.title('Cumulative Investment by Company')
    plt.grid()
    plt.savefig(cumulative_path)
    plt.close()

    total_post_investment = df['Post-Investment Amount ($)'].sum()
    return pie_path, bar_path, cumulative_path, df, total_post_investment

def visualize_user_data(name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, filename):
    """
    Create additional visualizations based on user data.
    """
    # Monthly expenses breakdown
    categories = ['Rent', 'Utilities', 'Groceries', 'Other Expenses']
    values = [rent, utilities, groceries, other_expenses]

    # Pie Chart for Monthly Expenses Breakdown
    pie_expenses_path = os.path.join(CHARTS_FOLDER, f"{filename}_expenses_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=140)
    plt.title('Monthly Expenses Breakdown')
    plt.savefig(pie_expenses_path)
    plt.close()

    # Bar Chart for Savings vs Expenses
    bar_savings_path = os.path.join(CHARTS_FOLDER, f"{filename}_savings_bar.png")
    plt.figure(figsize=(8, 6))
    plt.bar(['Total Expenses', 'Savings'], [total_expenses, savings], color=['lightcoral', 'skyblue'])
    plt.title(f'{name} - Savings vs Total Expenses')
    plt.xlabel('Category')
    plt.ylabel('Amount ($)')
    plt.savefig(bar_savings_path)
    plt.close()

    # Line Chart for Savings Growth (Assume savings grow evenly over 12 months)
    line_savings_path = os.path.join(CHARTS_FOLDER, f"{filename}_savings_line.png")
    months = list(range(1, 13))
    monthly_savings = [savings / 12 * i for i in months]
    plt.figure(figsize=(8, 6))
    plt.plot(months, monthly_savings, marker='o', color='green')
    plt.title(f'{name} - Savings Growth Over a Year')
    plt.xlabel('Month')
    plt.ylabel('Cumulative Savings ($)')
    plt.grid()
    plt.savefig(line_savings_path)
    plt.close()

    return pie_expenses_path, bar_savings_path, line_savings_path
@app.post("/calculate", response_class=HTMLResponse)
async def calculate(
    request: Request,
    name: str = Form(...),
    rent: float = Form(0.0),
    utilities: float = Form(0.0),
    groceries: float = Form(0.0),
    other_expenses: float = Form(0.0),
    earnings: float = Form(...),
):
    try:
        # Calculate total expenses and savings
        total_expenses = (rent + utilities + groceries + other_expenses) * 12  # Annualize monthly expenses
        savings = earnings - total_expenses

        # Save data to database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_data (name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, timestamp))
            conn.commit()

        # Generate visualizations if there are savings
        if savings > 0:
            filename = f"investment_{timestamp.replace(':', '_').replace(' ', '_')}"
            pie_path, bar_path, line_path, df, total_post_investment = visualize_investment_allocation(savings, filename)
            pie_expenses_path, bar_savings_path, line_savings_path = visualize_user_data(
                name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, filename
            )
            investment_table = df[['Company', 'Investment ($)', 'Expected Return (%)', 'Risk (Volatility %)', 'Post-Investment Amount ($)']].to_dict('records')
        else:
            pie_path, bar_path, line_path, pie_expenses_path, bar_savings_path, line_savings_path, investment_table, total_post_investment = None, None, None, None, None, None, [], 0.0

        # Render the results page
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "name": name,
                "total_expenses": total_expenses,
                "earnings": earnings,
                "savings": savings,
                "pie_path": pie_path,
                "bar_path": bar_path,
                "line_path": line_path,
                "pie_expenses_path": pie_expenses_path,
                "bar_savings_path": bar_savings_path,
                "line_savings_path": line_savings_path,
                "investment_table": investment_table,
                "total_post_investment": total_post_investment,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": f"Error processing input: {e}"}
        )


@app.get("/calculate", response_class=HTMLResponse)
async def calculate_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "error": "Please submit the form instead of navigating directly to this URL."})


@app.post("/calculate", response_class=HTMLResponse)
async def calculate(
    request: Request,
    name: str = Form(...),
    rent: float = Form(0.0),
    utilities: float = Form(0.0),
    groceries: float = Form(0.0),
    other_expenses: float = Form(0.0),
    earnings: float = Form(...),
):
    try:
        # Calculate total expenses and savings
        total_expenses = (rent + utilities + groceries + other_expenses) * 12  # Annualize monthly expenses
        savings = earnings - total_expenses

        # Save data to database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_data (name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, timestamp))
            conn.commit()

        # Generate visualizations if there are savings
        if savings > 0:
            filename = f"investment_{timestamp.replace(':', '_').replace(' ', '_')}"
            pie_path, bar_path, line_path, df, total_post_investment = visualize_investment_allocation(savings, filename)
            investment_table = df[['Company', 'Investment ($)', 'Expected Return (%)', 'Risk (Volatility %)', 'Post-Investment Amount ($)']].to_dict('records')
        else:
            pie_path, bar_path, line_path, investment_table, total_post_investment = None, None, None, [], 0.0

        # Render the results page
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "name": name,
                "total_expenses": total_expenses,
                "earnings": earnings,
                "savings": savings,
                "pie_path": pie_path,
                "bar_path": bar_path,
                "line_path": line_path,
                "investment_table": investment_table,
                "total_post_investment": total_post_investment,
            },
        )
    except Exception as e:
        print(f"Error: {e}")
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": f"Error processing input: {e}"}
        )
