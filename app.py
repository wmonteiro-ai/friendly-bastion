import pulp
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from optimization import *

# settings
st.set_page_config(layout="wide", page_title="Merit Optimization Model - Wellington")

# custom colors to match Comp style
st.markdown("""
<style>
    /* Target the specific button using its label */
    button[kind="primary"] {
        background-color: rgb(243, 27, 52) !important;
        color: white !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: rgb(210, 20, 45) !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)


def run_optimization(df, _weights, _budget_pct, _max_increase_percentage_bound):
    """
    Runs the full optimization process.
    The decorator @st.cache_data makes sure the Optimization will only run again if we modify the weights or constraints.
    """
    
    # calculating the priority scores based on the (new) weights
    df_processed = calculate_priority_scores(df, _weights)

    # calculating the maximum budget
    total_budget = df_processed['gross_base_salary'].sum() * _budget_pct
        
    # optimizing through PuLP
    model, gets_pct = get_diminishing_returns_model(df_processed, total_budget, _max_increase_percentage_bound)
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # calculating the results
    return update_dataframe_with_results(df_processed, model, gets_pct, total_budget, _max_increase_percentage_bound)


# loading data
@st.cache_data
def load_data():
    df = pd.read_csv('sample_data_for_optimization.tsv', sep='\t', index_col='employee_id')
    return df

df_original = load_data()

st.title("Merit Optimization Model")
st.markdown("This is a tool I've created with Streamlit to make it easy to test and run different scenarios. You can adjust the weights on the left and re-run the simulations.")

# sidebar
with st.sidebar:
    st.header("Strategy")

    with st.form("form", border=False):
        st.subheader("Total Budget")
        budget_pct = st.slider(
            "% of the current payroll", 
            min_value=1.0, max_value=10.0, value=2.5, step=0.1, format="%.1f%%"
        ) / 100
        
        max_increase_percentage_bound = st.slider("Maximum salary increase for an employee (%)", 1, 20, 15)
    
        st.subheader("Weights (0 to 100)")
        st.caption("Under the hood we will balance all weights, making sure the total of all weights always equals 100%")
        slider_weights = {
            'is_talent': st.slider("Importance of being a Talent", 0, 100, 50),
            'is_exceeding_expectations_last_year': st.slider("Importance of being a High Performer", 0, 100, 50),
            'high_performers_low_band': st.slider("Importance of prioritizing High Performers in lower salary bands", 0, 100, 50),
            
            'is_critical_position_or_successor': st.slider("Importance of being in a critical position or a critical position successor", 0, 100, 50),
            
            'zscore_years_without_promotion_or_merit_for_hr_group': st.slider("Importance of being too long without merit or promotion (vs. pairs)", 0, 100, 50),
            
            'percentage_current_band_inv': st.slider("Importance of being in lower salary bands", 0, 100, 50),
            'percentage_increase_last_12months_inv': st.slider("Importance of receiving lower salary increases in the last 12 months", 0, 100, 50),
        }

        submit = st.form_submit_button('Run optimization', use_container_width=True)

if submit:
    if 'results' in st.session_state:
        del st.session_state['results']

    results, summary = run_optimization(df_original, slider_weights, budget_pct, max_increase_percentage_bound)
    if results is not None:
        results.index.names = ['Employee ID']
        display_df = results[DISPLAY_COLUMNS]
        for col in ['percentage_current_band', 'proposed_increase_pct']:
            display_df[col] = display_df[col]/100

        for col in ['is_critical_position_or_successor', 'high_performers_low_band']:
            display_df[col] = display_df[col].astype(bool)
        
        display_df = display_df.sort_values(by="priority_score", ascending=False).rename(
            columns=RENAMED_LABELS).style.format({
            RENAMED_LABELS["priority_score"]: "{:.2f}",
            RENAMED_LABELS["years_without_promotion_or_merit"]: "{:.2f}",
            RENAMED_LABELS["gross_base_salary"]: "R$ {:,.2f}",
            RENAMED_LABELS["percentage_current_band"]: "{:.1%}",
            RENAMED_LABELS["proposed_increase_pct"]: "{:.1%}",
            RENAMED_LABELS["recommended_increase"]: "R$ {:,.2f}",
            RENAMED_LABELS["lead_monthly_budget"]: "R$ {:,.2f}",
            RENAMED_LABELS["recommended_increase"]: "R$ {:,.2f}",
            RENAMED_LABELS["new_gross_base_salary"]: "R$ {:,.2f}"
        })

        st.session_state['display'] = display_df
        st.session_state['results'] = results
        st.session_state['summary'] = summary
        st.session_state['weights'] = slider_weights
    else:
        st.error("We were unable to find a feasible solution. Try adjusting your weights and try again.")
    submit = False


# main results section
if 'results' in st.session_state:
    st.header("Optimization Results")
    display_df = st.session_state.display
    results_df = st.session_state.results
    summary = st.session_state.summary
    weights = st.session_state.weights
    
    # 1. Resumo Executivo
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Budget Used", f"R$ {summary['total_cost']:,.0f}", f"{summary['total_cost']/summary['budget_limit']:.2%} used from the limit")
    col2.metric("Employees with increases", f"{summary['employees_with_increase']}", f"{summary['employees_with_increase']/summary['total_employees']:.0%} from the total")
    col3.metric("Average increases (merited)", f"R$ {summary['avg_increase_merited']:.2f}", f"As percentage: {summary['avg_increase_pct_merited']:.2f}%")
    col4.metric("Average increases (total)", f"R$ {summary['avg_increase_all']:.2f}", f"As percentage: {summary['avg_increase_pct_all']:.2f}%")

    st.divider()

    # analysis and details
    tab1, tab2, tab3 = st.tabs(["Strategic Dashboard", "Distribution Analysis and Talent Impact", "Recommendation Table"])

    with tab1:
        st.header("Strategic Dashboard")
        st.subheader("ROI per salary increase")
        st.markdown("This chart shows the return on investment for each salary increase. Are we investing the most in our highest-priority employees?")
        
        fig_impact = px.scatter(
            results_df[results_df['proposed_increase_pct'] > 0].reset_index(drop=False),
            x='priority_score',
            y='proposed_increase_pct',
            size='recommended_increase',  # size of the dot reflects the cost
            color='talent_segment', # color-code by segment
            hover_data=['Employee ID', 'gross_base_salary', 'recommended_increase', 'percentage_current_band'],
            title="Priority Score vs. Proposed Increase",
            labels={
                "employee_id": "Employee ID",
                "gross_base_salary": "Old gross base salary (R$)",
                "priority_score": "Employee priority score (Why)",
                "proposed_increase_pct": "Proposed increase (%) (What)",
                "recommended_increase": "Cost of increase (R$)",
                "percentage_current_band": "Current salary band (%)",
                "talent_segment": "Segment"
            },
            category_orders={
                "talent_segment": ["Talent and High Performer", "Talent and Not High Performer", "Not Talent and High Performer", "Not Talent and Not High Performer"]
            }
        )
        fig_impact.update_layout(xaxis_title="← Lower Priority | Higher Priority →", yaxis_title="Increase %")
        st.plotly_chart(fig_impact, use_container_width=True)
        
        st.subheader("Priority score breakdown")
        st.markdown("This chart shows how our weights are driving the priority scores for different employee segments.")

        fig_sunburst = px.sunburst(
            get_components_for_sunburst_plot(results_df, weights),
            path=['talent_segment', 'Component'],
            values='Total Weighted Score',
            title="Breakdown of What Drives Priority Scores Across the Company",
            color='talent_segment',
            labels={
                "employee_id": "Employee ID",
                "talent_segment": "Segment"
            }
        )
        fig_sunburst.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig_sunburst, use_container_width=True)

        st.header("Budget allocation by strategic priority")
        st.markdown("This chart shows how the total salary increase budget is being allocated based on the primary driver of each employee's priority score.")
        
        fig_budget_alloc = px.bar(
            get_components_for_budget_allocation_plot(results_df, weights),
            x='recommended_increase',
            y='primary_driver',
            color='primary_driver',
            title="Budget Spend per Strategic Driver",
            labels={
                'primary_driver': 'Strategic Priority',
                'recommended_increase': 'Total Increase Cost (R$)'
            }
        )
        fig_budget_alloc.update_layout(showlegend=False) # The colors are self-explanatory
        st.plotly_chart(fig_budget_alloc, use_container_width=True)

        anomaly_threshold = 0.2172693419268501

        st.subheader("Retention investment quadrant")
        st.markdown("This chart helps us understand if we are investing our retention budget on the right at-risk employees.")
        
        # filter for at-risk employees (you can adjust the threshold)
        at_risk_df = results_df[results_df['zscore_years_without_promotion_or_merit_for_hr_group'] > anomaly_threshold].copy() # e.g., > 1.5 standard deviations (before normalization)
        
        # defining the quadrant boundaries (e.g., the median or a specific business threshold)
        median_stagnation = at_risk_df['zscore_years_without_promotion_or_merit_for_hr_group'].median()
        median_priority = at_risk_df['priority_score'].median()
        
        fig_quadrant = px.scatter(
            at_risk_df.reset_index(drop=False),
            x='zscore_years_without_promotion_or_merit_for_hr_group',
            y='priority_score',
            size='recommended_increase',
            color='talent_segment',
            hover_data=['Employee ID'],
            title="Stagnation vs. Priority for At-Risk Employees",
            labels={
                "zscore_years_without_promotion_or_merit_for_hr_group": "Stagnation Score (Higher = More At-Risk)",
                "priority_score": "Overall Priority Score",
                "talent_segment": "Segment",
                "employee_id": "At-Risk Employee ID",
                "recommended_increase": "Recommended Increase"
            }
        )
        
        # add quadrant lines to the plot
        fig_quadrant.add_shape(type="line", x0=median_stagnation, y0=0, x1=median_stagnation, y1=at_risk_df['priority_score'].max(), line=dict(color="Gray", width=2, dash="dash"))
        fig_quadrant.add_shape(type="line", x0=0, y0=median_priority, x1=at_risk_df['zscore_years_without_promotion_or_merit_for_hr_group'].max(), y1=median_priority, line=dict(color="Gray", width=2, dash="dash"))
        
        # add annotations for the quadrants
        fig_quadrant.add_annotation(x=median_stagnation, y=at_risk_df['priority_score'].max(), text="Critical to retain", showarrow=False, xanchor='left', yanchor='top', font=dict(color="green"))
        fig_quadrant.add_annotation(x=median_stagnation, y=0, text="Review and monitor", showarrow=False, xanchor='left', yanchor='bottom', font=dict(color="red"))
        
        st.plotly_chart(fig_quadrant, use_container_width=True)
        
        st.subheader("Stagnation distribution and cost overlay")
        st.markdown("This visual shows the distribution of employee stagnation and how much budget is allocated to each level.")
        
        # create bins for the stagnation score to group employees
        results_df['stagnation_bin'] = pd.cut(
            results_df['zscore_years_without_promotion_or_merit_for_hr_group'],
            bins=10, # You can adjust the number of bins
            labels=False,
            include_lowest=True
        )
        
        # aggregate data by bin
        stagnation_summary = results_df.reset_index(drop=False).groupby('stagnation_bin').agg(
            employee_count=('Employee ID', 'count'),
            total_increase_cost=('recommended_increase', 'sum'),
            avg_stagnation_score=('zscore_years_without_promotion_or_merit_for_hr_group', 'mean')
        ).reset_index()
        
        # creating a dual-axis chart
        fig_hist = go.Figure()
        
        # bar chart for employee count
        fig_hist.add_trace(go.Bar(
            x=stagnation_summary['avg_stagnation_score'],
            y=stagnation_summary['employee_count'],
            name='Number of Employees',
            marker_color='darkgray'
        ))
        
        # line chart for total cost
        fig_hist.add_trace(go.Scatter(
            x=stagnation_summary['avg_stagnation_score'],
            y=stagnation_summary['total_increase_cost'],
            name='Total Investment (R$)',
            mode='lines+markers',
            yaxis='y2', # Assign to the secondary y-axis
            line=dict(color='rgb(243,27,52)', width=3)
        ))
        
        # now updating the chart with two y-axes
        fig_hist.update_layout(
            title="Investment vs. employee count by stagnation level",
            xaxis_title="Average Stagnation Score (to the right of the dashed line = stagnation anomalies)",
            yaxis=dict(title="Number of Employees", color="darkgray"),
            yaxis2=dict(title="Total Investment (R$)", overlaying='y', side='right', color="rgb(243,27,52)"),
            legend=dict(x=0.7, y=0.9, xanchor="left", yanchor="top")
        )
        
        fig_hist.add_shape(type="line", x0=anomaly_threshold, y0=stagnation_summary['employee_count'].min(), 
                           x1=anomaly_threshold, y1=stagnation_summary['employee_count'].max() * 1.1, 
                           line=dict(color="Gray", width=2, dash="dash"))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with tab2:
        st.subheader("Salary increase distribution")
        fig_dist = px.histogram(results_df[results_df['proposed_increase_pct'] > 0], 
                                x='proposed_increase_pct',
                                title="Frequency of each suggested increase",
                                color_discrete_sequence=["rgb(243,27,52)"],
                                labels={"proposed_increase_pct": "Increase percentage (%)"})
        fig_dist.update_layout(yaxis_title="# employees")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Strategy breakdown per employee type")
        fig_talent_impact = px.box(results_df, 
                                   x='talent_segment', 
                                   y='proposed_increase_pct', 
                                   color='talent_segment',
                                   title="Talent Segment Comparison",
                                   labels={"talent_segment": "Segment", "proposed_increase_pct": "Increase percentage (%)"},
                                   points="all")
        st.plotly_chart(fig_talent_impact, use_container_width=True)

    with tab3:
        st.subheader("Detailed recommendations for all employees")
        st.dataframe(display_df, use_container_width=True)

else:
    st.info("Modify the weights in the sidebar and click on the _Run optimization_ button to see the results.")
