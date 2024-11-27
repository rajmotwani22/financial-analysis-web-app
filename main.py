from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Initialize the app
app = FastAPI()

# Directories
CHARTS_FOLDER = "static/saved_charts"
os.makedirs(CHARTS_FOLDER, exist_ok=True)

# Jinja2 templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def visualize_investment_allocation(investment_amount, filename):
    """
    Creates and saves visualizations for investment allocation.
    """
    # Sample data for top companies
    data = {
        'Company': ['Apple', 'Amazon', 'Microsoft', 'Google', 'Tesla'],
        'Expected Return (%)': [12, 15, 10, 11, 20],
        'Risk (Volatility %)': [18, 22, 15, 17, 30]
    }
    df = pd.DataFrame(data)

    # Calculate proportional allocation
    df['Weight'] = df['Expected Return (%)'] / df['Expected Return (%)'].sum()
    df['Investment ($)'] = df['Weight'] * investment_amount

    # Create Pie Chart
    pie_path = os.path.join(CHARTS_FOLDER, f"{filename}_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(df['Investment ($)'], labels=df['Company'], autopct=lambda p: f'${p * investment_amount / 100:,.2f}', startangle=140)
    plt.title('Investment Allocation by Company')
    plt.savefig(pie_path)
    plt.close()

    # Create Bar Chart for Expected Returns and Investment Amount
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

    return pie_path, bar_path, cumulative_path, df



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate", response_class=HTMLResponse)
async def calculate(
    request: Request,
    rent: float = Form(0.0),
    utilities: float = Form(0.0),
    groceries: float = Form(0.0),
    dining: float = Form(0.0),
    transportation: float = Form(0.0),
    entertainment: float = Form(0.0),
    shopping: float = Form(0.0),
    others: float = Form(0.0),
    earnings: float = Form(...),
):
    try:
        # Calculate total expenses and savings
        monthly_expenses = rent + utilities + groceries + dining + transportation + entertainment + shopping + others
        total_expenses = monthly_expenses * 12  # Annualize monthly expenses
        savings = earnings - total_expenses

        # Generate visualizations if there are savings
        if savings > 0:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            pie_path, bar_path, line_path, df = visualize_investment_allocation(savings, f"investment_{timestamp}")
            investment_table = df[['Company', 'Investment ($)', 'Expected Return (%)', 'Risk (Volatility %)']].to_dict('records')
        else:
            pie_path, bar_path, line_path, investment_table = None, None, None, []

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "total_expenses": total_expenses,
                "earnings": earnings,
                "savings": savings,
                "pie_path": pie_path,
                "bar_path": bar_path,
                "line_path": line_path,
                "investment_table": investment_table,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": f"Error processing input: {e}"}
        )
