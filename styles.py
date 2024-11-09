STYLE_CSS = """
/* Custom styles for your Streamlit app */

/* Styling the buttons */
.stButton > button {
    width: 90%;
    margin: 5px 0;
    padding: 8px 0;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
}

/* Styling sidebar grid layout */
.sidebar-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}
.sidebar-grid .stButton {
    flex: 1 1 48%;
}

/* Additional styling */
.centered {
    text-align: center;
}

.pdf-box, .summary-box {
    border: 1px solid #ccc;
    padding: 10px;
    margin-bottom: 10px;
}
"""