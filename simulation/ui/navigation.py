"""Wizard navigation component."""
import streamlit as st

try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun


def render_navigation():
    """Render the wizard navigation with clickable step indicators in the sidebar."""
    wizard_step = st.session_state.wizard_step
    
    with st.sidebar:
        st.title("Trading Simulation")
        st.markdown("---")
        st.header("Steps")

        if st.button(
            "Step 1: Introduction",
            key="nav_step1",
            use_container_width=True,
            type="primary" if wizard_step == 1 else "secondary"
        ):
            st.session_state.wizard_step = 1
            rerun()

        if st.button(
            "Step 2: Find Best Signals",
            key="nav_step2",
            use_container_width=True,
            type="primary" if wizard_step == 2 else "secondary"
        ):
            st.session_state.wizard_step = 2
            rerun()

        if st.button(
            "Step 3: Verify Strategy",
            key="nav_step3",
            use_container_width=True,
            type="primary" if wizard_step == 3 else "secondary"
        ):
            st.session_state.wizard_step = 3
            rerun()

        if st.button(
            "Step 4: Testing",
            key="nav_step4",
            use_container_width=True,
            type="primary" if wizard_step == 4 else "secondary"
        ):
            st.session_state.wizard_step = 4
            rerun()

        if st.button(
            "Step 5: AI Summary",
            key="nav_step5",
            use_container_width=True,
            type="primary" if wizard_step == 5 else "secondary"
        ):
            st.session_state.wizard_step = 5
            rerun()
