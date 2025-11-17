"""Wizard navigation component."""
import streamlit as st

try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun


def render_navigation():
    """Render the wizard navigation with clickable step indicators."""
    wizard_step = st.session_state.wizard_step
    
    # Create visual progress bar with clickable steps
    progress_cols = st.columns(5)
    
    with progress_cols[0]:
        if st.button(
            "🔵 **Step 1: Introduction**" if wizard_step == 1 else "✅ Step 1: Introduction",
            key="nav_step1",
            use_container_width=True,
            type="primary" if wizard_step == 1 else "secondary"
        ):
            st.session_state.wizard_step = 1
            rerun()
    
    with progress_cols[1]:
        step2_label = (
            "🔵 **Step 2: Find Best Signals**" if wizard_step == 2
            else ("✅ Step 2: Find Best Signals" if wizard_step > 2 else "⚪ Step 2: Find Best Signals")
        )
        if st.button(
            step2_label,
            key="nav_step2",
            use_container_width=True,
            type="primary" if wizard_step == 2 else "secondary"
        ):
            st.session_state.wizard_step = 2
            rerun()
    
    with progress_cols[2]:
        step3_label = (
            "🔵 **Step 3: Verify Strategy**" if wizard_step == 3
            else ("✅ Step 3: Verify Strategy" if wizard_step > 3 else "⚪ Step 3: Verify Strategy")
        )
        if st.button(
            step3_label,
            key="nav_step3",
            use_container_width=True,
            type="primary" if wizard_step == 3 else "secondary"
        ):
            st.session_state.wizard_step = 3
            rerun()
    
    with progress_cols[3]:
        step4_label = (
            "🔵 **Step 4: Testing**" if wizard_step == 4
            else ("✅ Step 4: Testing" if wizard_step > 4 else "⚪ Step 4: Testing")
        )
        if st.button(
            step4_label,
            key="nav_step4",
            use_container_width=True,
            type="primary" if wizard_step == 4 else "secondary"
        ):
            st.session_state.wizard_step = 4
            rerun()
    
    with progress_cols[4]:
        step5_label = "🔵 **Step 5: AI Summary**" if wizard_step == 5 else "⚪ Step 5: AI Summary"
        if st.button(
            step5_label,
            key="nav_step5",
            use_container_width=True,
            type="primary" if wizard_step == 5 else "secondary"
        ):
            st.session_state.wizard_step = 5
            rerun()
    
    st.markdown("---")
