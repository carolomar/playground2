import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Call Center Analytics", layout="wide")


def extract_call_type(subject):
    """Extract the last part of the subject field after '>'"""
    if '>' in subject:
        return subject.split('>')[-1].strip()
    return subject


def load_and_process_data(uploaded_file):
    """Load and process the CSV file"""
    df = pd.read_csv(uploaded_file)

    # Extract call types
    df['Call Type'] = df['Subject'].apply(extract_call_type)

    # Convert date columns to datetime
    date_columns = ['Opened', 'Updated', 'Closed']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Extract month from Opened date
    df['Month'] = df['Opened'].dt.strftime('%Y-%m')

    # Calculate resolution time
    df['Resolution Time'] = (df['Closed'] - df['Opened']).dt.total_seconds() / 3600  # in hours

    return df


def create_heatmap(df, selected_month):
    """Create a heatmap of call types by agent"""
    # Filter data for selected month
    monthly_data = df[df['Month'] == selected_month]

    # Create pivot table
    pivot_data = pd.pivot_table(
        monthly_data,
        values='Number',
        index='Assigned to',
        columns='Call Type',
        aggfunc='count',
        fill_value=0
    )

    # Create heatmap using plotly
    fig = px.imshow(
        pivot_data,
        aspect='auto',
        color_continuous_scale='Reds',
        title=f'Call Distribution Heatmap - {selected_month}'
    )

    fig.update_layout(
        xaxis_title="Call Type",
        yaxis_title="Agent",
        height=600
    )

    return fig


def create_comparative_heatmap(df, month1, month2):
    """Create a comparison heatmap showing the difference between two months"""
    # Create pivot tables for both months
    pivot1 = pd.pivot_table(
        df[df['Month'] == month1],
        values='Number',
        index='Assigned to',
        columns='Call Type',
        aggfunc='count',
        fill_value=0
    )

    pivot2 = pd.pivot_table(
        df[df['Month'] == month2],
        values='Number',
        index='Assigned to',
        columns='Call Type',
        aggfunc='count',
        fill_value=0
    )

    # Calculate difference
    diff_pivot = pivot2 - pivot1

    # Create heatmap using plotly
    fig = px.imshow(
        diff_pivot,
        aspect='auto',
        color_continuous_scale='RdBu',
        title=f'Change in Call Distribution ({month2} vs {month1})'
    )

    fig.update_layout(
        xaxis_title="Call Type",
        yaxis_title="Agent",
        height=600
    )

    return fig


def create_trend_analysis(df, selected_call_type):
    """Create trend analysis over time for selected call type"""
    # Group by month and call type
    monthly_trends = df[df['Call Type'] == selected_call_type].groupby('Month').agg({
        'Number': 'count',
        'Resolution Time': 'mean'
    }).reset_index()

    # Create subplot with dual y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add call volume line
    fig.add_trace(
        go.Scatter(
            x=monthly_trends['Month'],
            y=monthly_trends['Number'],
            name="Call Volume",
            line=dict(color='blue')
        ),
        secondary_y=False
    )

    # Add resolution time line
    fig.add_trace(
        go.Scatter(
            x=monthly_trends['Month'],
            y=monthly_trends['Resolution Time'],
            name="Avg Resolution Time (hours)",
            line=dict(color='red')
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=f"Trend Analysis for {selected_call_type}",
        xaxis_title="Month",
        height=400
    )

    fig.update_yaxes(title_text="Call Volume", secondary_y=False)
    fig.update_yaxes(title_text="Resolution Time (hours)", secondary_y=True)

    return fig


def create_agent_analysis(df, selected_call_type):
    """Create agent analysis for selected call type"""
    # Filter data for selected call type
    call_type_data = df[df['Call Type'] == selected_call_type]

    # Get call counts by agent
    agent_counts = call_type_data['Assigned to'].value_counts()

    # Get all agents
    all_agents = df['Assigned to'].unique()

    # Create two lists: active and inactive agents
    active_agents = []
    inactive_agents = []

    for agent in all_agents:
        count = agent_counts.get(agent, 0)
        if count > 0:
            active_agents.append((agent, count))
        else:
            inactive_agents.append((agent, 0))

    return active_agents, inactive_agents


# Main app
st.title("Call Center Analytics Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load and process data
    df = load_and_process_data(uploaded_file)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Heatmap View", "Comparison View", "Agent Analysis"])

    with tab1:
        st.header("Call Distribution Heatmap")

        # Month selector
        available_months = sorted(df['Month'].unique())
        selected_month = st.selectbox(
            "Select Month",
            available_months,
            key="single_month"
        )

        # Create and display heatmap
        heatmap = create_heatmap(df, selected_month)
        st.plotly_chart(heatmap, use_container_width=True)

    with tab2:
        st.header("Month-over-Month Comparison")

        # Create two columns for month selection
        col1, col2 = st.columns(2)

        with col1:
            base_month = st.selectbox(
                "Select Base Month",
                available_months,
                key="base_month"
            )

        with col2:
            compare_month = st.selectbox(
                "Select Comparison Month",
                available_months,
                key="compare_month"
            )

        # Display comparison heatmap
        if base_month != compare_month:
            comparison_heatmap = create_comparative_heatmap(df, base_month, compare_month)
            st.plotly_chart(comparison_heatmap, use_container_width=True)

            # Calculate and display summary statistics
            base_total = len(df[df['Month'] == base_month])
            compare_total = len(df[df['Month'] == compare_month])
            percent_change = ((compare_total - base_total) / base_total) * 100

            st.markdown(f"""
            ### Summary Statistics
            - Base Month ({base_month}): {base_total} calls
            - Comparison Month ({compare_month}): {compare_total} calls
            - Change: {percent_change:.1f}%
            """)

    with tab3:
        st.header("Agent Call Type Analysis")

        # Call type selector
        call_types = sorted(df['Call Type'].unique())
        selected_call_type = st.selectbox(
            "Select Call Type",
            call_types
        )

        # Display trend analysis
        trend_fig = create_trend_analysis(df, selected_call_type)
        st.plotly_chart(trend_fig, use_container_width=True)

        # Get agent analysis
        active_agents, inactive_agents = create_agent_analysis(df, selected_call_type)

        # Display results in columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Active Agents")
            for agent, count in active_agents:
                st.write(f"{agent}: {count} calls")

        with col2:
            st.subheader("Inactive Agents")
            for agent, count in inactive_agents:
                st.write(f"{agent}: {count} calls")

        # Create bar chart of active agents
        if active_agents:
            fig = px.bar(
                x=[agent[0] for agent in active_agents],
                y=[agent[1] for agent in active_agents],
                title=f"Call Distribution for {selected_call_type}",
                labels={'x': 'Agent', 'y': 'Number of Calls'}
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to begin analysis.")