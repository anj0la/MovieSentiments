"""
File: create_cards.py

Author: Anjola Aina
Date Modified: October 30th, 2024

This module generates HTML-styled cards displaying key model performance metrics for visualization within the Streamlit application.

Functions:
    create_cards: Constructs a stylized HTML snippet for displaying model evaluation metrics (accuracy, precision, recall, F1 score) as cards.

External Dependencies:
    - None: Only requires standard Python libraries.
"""

def create_cards(test_values: list) -> str:
    """
    Generates HTML content for displaying model evaluation metrics (Test Accuracy, Precision, Recall, F1 Score) in a card format.

    Args:
        test_values (list): A list of floats representing the model's evaluation metrics in the following order:
            - test_values[0]: Test Accuracy
            - test_values[1]: Precision
            - test_values[2]: Recall
            - test_values[3]: F1 Score

    Returns:
        str: An HTML string containing the styled card components displaying the provided metrics.
    
    Example:
        >>> test_values = [0.85, 0.80, 0.78, 0.79]
        >>> html_content = create_cards(test_values)
        >>> # Displays metrics in card format in the Streamlit app
    """
    card_html = f"""
        <style>
            .card {{
                border: 2px solid rgba(0, 0, 0, 0.05);
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

