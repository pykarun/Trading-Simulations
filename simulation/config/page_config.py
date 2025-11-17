"""Page configuration and styling for the Streamlit app."""
import streamlit as st


def setup_page():
    """Configure Streamlit page settings and custom CSS."""
    st.set_page_config(
        page_title="Leveraged ETF Trader",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for mobile optimization
    st.markdown("""
    <style>
        /* Hide sidebar completely */
        [data-testid="stSidebar"] {
           display: none;
        }
        
        /* Fix header spacing - prevent overlap with deploy button */
        .block-container {
           padding-top: 3rem;
           padding-bottom: 1rem;
        }
        
        /* Additional top margin for main content */
        .main .block-container {
           margin-top: 2rem;
        }
        
        /* Ensure title is visible */
        h1:first-of-type {
           margin-top: 1rem;
           padding-top: 1rem;
        }
        
        /* Responsive text */
        @media (max-width: 768px) {
           h1 {
               font-size: 1.5rem !important;
           }
           h2 {
               font-size: 1.3rem !important;
           }
           h3 {
               font-size: 1.1rem !important;
           }
           
           /* More top padding on mobile */
           .block-container {
               padding-top: 4rem;
           }
        }
        
        /* Better button spacing on mobile */
        .stButton button {
           width: 100%;
        }
        
        /* Fix for Streamlit's top toolbar */
        header[data-testid="stHeader"] {
           background-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 Leveraged ETF System Trader")
    st.caption("Trade Leveraged ETF with confidence")
