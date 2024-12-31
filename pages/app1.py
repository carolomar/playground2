import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import re

# Set page config
st.set_page_config(page_title="Help Desk Ticket Analysis", layout="wide")


def analyze_agent_tickets(df, month, ticket_suffix, did_handle=True):
    """
    Analyze which agents did/didn't handle specific ticket types
    """
    monthly_data = df[df['Month'] == month].copy()
    monthly_data['Subject'] = monthly_data['Subject'].astype(str)

    # Find tickets ending with the specified suffix
    matching_tickets = monthly_data['Subject'].str.endswith(ticket_suffix)

    # Get all agents and agents who handled matching tickets
    all_agents = set(monthly_data['Assigned to'])
    agents_with_ticket = set(monthly_data[matching_tickets]['Assigned to'])

    if did_handle:
        result_agents = agents_with_ticket
        message = f"Agents who handled tickets ending with '{ticket_suffix}' in {month}:"
    else:
        result_agents = all_agents - agents_with_ticket
        message = f"Agents who didn't handle tickets ending with '{ticket_suffix}' in {month}:"

    return {
        'agents': sorted(list(result_agents)),
        'message': message,
        'matching_tickets': sorted(monthly_data[matching_tickets]['Subject'].unique())
    }


# Function to load and preprocess data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Opened'] = pd.to_datetime(df['Opened'])
    df['Month'] = df['Opened'].dt.strftime('%Y-%m')
    return df


# Function to parse natural language query
def parse_query(query, df, month):
    monthly_data = df[df['Month'] == month]

    # Convert query to lowercase for easier matching
    query = query.lower()

    # Pattern for "who didn't do X"
    not_doing_pattern = r"who didn't do (.+?)(?:\?|$)"

    if "who didn't do" in query:
        match = re.search(not_doing_pattern, query)
        if match:
            ticket_type = match.group(1).strip()

            # Convert all Subject values to strings and lowercase for comparison
            monthly_data['Subject'] = monthly_data['Subject'].astype(str).str.lower()

            # Find agents who didn't handle this ticket type
            agents_with_ticket = set(monthly_data[monthly_data['Subject'] == ticket_type]['Assigned to'])
            all_agents = set(monthly_data['Assigned to'])
            agents_without_ticket = all_agents - agents_with_ticket

            # Add debugging information
            st.write("Debug Info:")
            st.write(f"Searching for ticket type: '{ticket_type}'")
            st.write("Available ticket types:", sorted(monthly_data['Subject'].unique()))

            return {
                'type': 'agent_list',
                'result': sorted(list(agents_without_ticket)),
                'message': f"Agents who didn't handle '{ticket_type}' tickets in {month}:"
            }

    return {
        'type': 'error',
        'result': None,
        'message': "I couldn't understand that query. Try patterns like 'who didn't do [ticket type]?'"
    }

# [Previous functions remain the same: create_heatmap, get_summary_metrics, create_workload_comparison, download_dataframe]
# ... [Keep all the previous functions unchanged] ...

# Function to create heatmap
def create_heatmap(df, month):
    monthly_data = df[df['Month'] == month]
    pivot_table = pd.crosstab(monthly_data['Assigned to'], monthly_data['Subject'])

    fig = px.imshow(pivot_table,
                    labels=dict(x="Ticket Type", y="Agent", color="Count"),
                    aspect="auto",
                    color_continuous_scale="Blues")

    fig.update_layout(
        title=f"Agent-Ticket Type Distribution for {month}",
        xaxis_title="Ticket Type",
        yaxis_title="Agent",
        height=400
    )
    return fig


# Function to calculate summary metrics
def get_summary_metrics(df, month):
    monthly_data = df[df['Month'] == month]
    return {
        'total_tickets': len(monthly_data),
        'num_agents': monthly_data['Assigned to'].nunique(),
        'num_ticket_types': monthly_data['Subject'].nunique(),
        'avg_tickets_per_agent': len(monthly_data) / monthly_data['Assigned to'].nunique()
    }


# Function to create workload comparison
def create_workload_comparison(df, month1, month2):
    workload1 = df[df['Month'] == month1]['Assigned to'].value_counts()
    workload2 = df[df['Month'] == month2]['Assigned to'].value_counts()

    workload_comp = pd.DataFrame({
        month1: workload1,
        month2: workload2
    }).fillna(0)
    workload_comp['difference'] = workload_comp[month2] - workload_comp[month1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=month1,
        x=workload_comp.index,
        y=workload_comp[month1],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name=month2,
        x=workload_comp.index,
        y=workload_comp[month2],
        marker_color='darkblue'
    ))

    fig.update_layout(
        title="Agent Workload Comparison",
        xaxis_title="Agent",
        yaxis_title="Number of Tickets",
        barmode='group',
        height=400
    )
    return fig, workload_comp


# Function to download dataframe
def download_dataframe(df, file_format='csv'):
    buffer = io.BytesIO()
    if file_format == 'csv':
        df.to_csv(buffer, index=True)
        mime_type = 'text/csv'
        file_extension = 'csv'
    else:  # Excel
        df.to_excel(buffer, index=True)
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        file_extension = 'xlsx'

    buffer.seek(0)
    return buffer, mime_type, file_extension


# Main app
def main():
    st.title("Help Desk Ticket Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        available_months = sorted(df['Month'].unique())

        # Agent Analysis Section
        st.header("Agent Ticket Analysis")

        # Get unique ticket suffixes (everything after the last '>') with type handling
        all_tickets = df['Subject'].astype(str)  # Convert all to strings first
        ticket_suffixes = []
        for ticket in all_tickets:
            if ">" in ticket:
                suffix = ticket.split(">")[-1].strip()
            else:
                suffix = ticket.strip()
            if suffix:  # Only add non-empty suffixes
                ticket_suffixes.append(suffix)
        ticket_suffixes = sorted(set(ticket_suffixes))  # Get unique values and sort

        # Analysis Controls
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            selected_month = st.selectbox("Select Month:", available_months, key='analysis_month')
        with col2:
            selected_suffix = st.selectbox("Select Ticket Type:", ticket_suffixes)
        with col3:
            analysis_type = st.selectbox("Show agents who:", ["Did handle", "Didn't handle"])
            # Perform analysis
            if selected_suffix:
                result = analyze_agent_tickets(
                    df,
                    selected_month,
                    selected_suffix,
                    did_handle=(analysis_type == "Did handle")
                )

            # Display results
            st.subheader(result['message'])
            if result['agents']:
                for agent in result['agents']:
                    st.write(f"- {agent}")
            else:
                st.write("No agents found matching this criteria.")

            # Show matching tickets
            with st.expander("Show matching ticket types"):
                st.write("Matched these ticket types:")
                for ticket in result['matching_tickets']:
                    st.write(f"- {ticket}")

        # Monthly Comparison Section
        st.header("Monthly Comparison")
        col1, col2 = st.columns(2)
        with col1:
            month1 = st.selectbox("Select first month", available_months, index=0, key='comp_month1')
        with col2:
            month2 = st.selectbox("Select second month", available_months, index=min(1, len(available_months) - 1),
                                  key='comp_month2')

        # Summary metrics comparison
        st.header("Summary Metrics Comparison")
        metrics1 = get_summary_metrics(df, month1)
        metrics2 = get_summary_metrics(df, month2)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tickets",
                      metrics1['total_tickets'],
                      metrics2['total_tickets'] - metrics1['total_tickets'])
        with col2:
            st.metric("Number of Agents",
                      metrics1['num_agents'],
                      metrics2['num_agents'] - metrics1['num_agents'])
        with col3:
            st.metric("Ticket Types",
                      metrics1['num_ticket_types'],
                      metrics2['num_ticket_types'] - metrics1['num_ticket_types'])
        with col4:
            st.metric("Avg Tickets/Agent",
                      round(metrics1['avg_tickets_per_agent'], 1),
                      round(metrics2['avg_tickets_per_agent'] - metrics1['avg_tickets_per_agent'], 1))

        # Heatmaps
        st.header("Agent-Ticket Type Distribution")
        tab1, tab2 = st.tabs([f"Month: {month1}", f"Month: {month2}"])

        with tab1:
            st.plotly_chart(create_heatmap(df, month1), use_container_width=True)
        with tab2:
            st.plotly_chart(create_heatmap(df, month2), use_container_width=True)

        # Workload comparison
        st.header("Agent Workload Comparison")
        workload_fig, workload_comp = create_workload_comparison(df, month1, month2)
        st.plotly_chart(workload_fig, use_container_width=True)

        # Ticket type distribution comparison
        st.header("Ticket Type Distribution")
        type_dist1 = df[df['Month'] == month1]['Subject'].value_counts()
        type_dist2 = df[df['Month'] == month2]['Subject'].value_counts()

        type_comp = pd.DataFrame({
            month1: type_dist1,
            month2: type_dist2
        }).fillna(0)

        fig = px.bar(type_comp,
                     barmode='group',
                     title="Ticket Type Distribution Comparison")
        st.plotly_chart(fig, use_container_width=True)

        # Download options
        st.header("Download Analysis")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Download as CSV"):
                buffer, mime_type, file_extension = download_dataframe(workload_comp, 'csv')
                st.download_button(
                    label="Download workload comparison",
                    data=buffer,
                    file_name=f'workload_comparison.{file_extension}',
                    mime=mime_type
                )

        with col2:
            if st.button("Download as Excel"):
                buffer, mime_type, file_extension = download_dataframe(workload_comp, 'excel')
                st.download_button(
                    label="Download workload comparison",
                    data=buffer,
                    file_name=f'workload_comparison.{file_extension}',
                    mime=mime_type
                )


if __name__ == "__main__":
    main()