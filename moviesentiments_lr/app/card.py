def create_cards(test_values: list) -> str:
    card_html = f"""
        <style>
            .card {{
                background-color: #F5F5F5;
                border-radius: 10px;
                margin: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-family: Arial, sans-serif;
            }}
            .card h3 {{
                font-size: 14px;
                font-style: italic;
                color: #333333;
                padding-left: 20px;
                margin-bottom: -10px;
            }}
            .card p {{
                font-size: 30px;
                font-weight: bold;
                color: #50C878;
                margin-top: -5px;
                margin-bottom: 10px;
            }}
            .container {{
                display: flex;
                flex-direction: row;
                justify-content: center;
                width: 100%;
            }}

        </style>

        <div class="container">
            <div class="card">
                <h3>Test Accuracy</h3>
                <p>{test_values[0]:.2%}</p>
            </div>
            <div class="card">
                <h3>Precision</h3>
                <p>{test_values[1]:.2%}</p>
            </div>
            <div class="card">
                <h3>Recall</h3>
                <p>{test_values[2]:.2%}</p>
            </div>
            <div class="card">
                <h3>F1 Score</h3>
                <p>{test_values[3]:.2%}</p>
            </div>
        </div>
        """
    return card_html

