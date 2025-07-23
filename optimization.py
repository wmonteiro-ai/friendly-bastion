import pulp
import numpy as np
import pandas as pd

from typing import Tuple
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
from pulp.constants import LpStatus

# Default parameters
DEFAULT_PRIORITY_WEIGHTS = {
    'is_talent': 1,
    'is_exceeding_expectations_last_year': 1, 
    'high_performers_low_band': 1,

    'is_critical_position_or_successor': 1,
    
    'zscore_years_without_promotion_or_merit_for_hr_group': 1, # using z-score as it is more robust than just the years
    
    'percentage_current_band_inv': 1, # inverting it, as we want to give more priority to lower percentages
    'percentage_increase_last_12months_inv': 1 # inverting it, as we want to give more priority to lower percentages
}

RENAMED_LABELS = {
    'is_talent': 'Is Talent?',
    'is_exceeding_expectations_last_year': 'Exceeding Expectations?', 
    'high_performers_low_band': 'High Performer and Low Salary Band?',
    'is_critical_position_or_successor': 'Is Critical Position or Successor?',
    'zscore_years_without_promotion_or_merit_for_hr_group': 'Years Without Promotion/Merit (Anomaly)',
    'percentage_current_band_inv': 'Current Band (%)',
    'percentage_increase_last_12months_inv': 'Increase Last 12 Months (%)',
    'gross_base_salary': 'Old Gross Base Salary',
    'new_gross_base_salary': 'New Gross Base Salary',
    'proposed_increase_pct': 'Proposed Increase (%)',
    'recommended_increase': 'Proposed Increase (R$)',
    'priority_score': 'Priority Score',
    'employee_id': 'Employee ID',
    'lead_id': 'Lead ID',
    'lead_monthly_budget': 'Lead Increase Budget (R$)',
    'talent_segment': 'Segment',
    'percentage_current_band': 'Current Band (%)',
    'years_without_promotion_or_merit': 'Years Without Promotion/Merit'
}

NORMALIZED_COLUMNS = list(DEFAULT_PRIORITY_WEIGHTS.keys())
DISPLAY_COLUMNS = ['gross_base_salary', 'new_gross_base_salary', 'proposed_increase_pct', 'recommended_increase',
                   'priority_score', 'talent_segment', 'high_performers_low_band', 'is_critical_position_or_successor',
                   'percentage_current_band', 'years_without_promotion_or_merit', 'lead_id', 'lead_monthly_budget']
OTHER_REQUIRED_COLUMNS = ['employee_id', 'lead_id', 'gross_base_salary', 'lead_monthly_budget', 'talent_segment', 'stagnation_compensation_quadrant', 'percentage_current_band', 'percentage_increase_last_12months', 'years_without_promotion_or_merit']

# helper functions
def calculate_priority_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Calculating our score.
    Higher scores for an employee = higher priority to have an increase.    
    """
    
    total_weights = sum(weights.values())
    
    df['priority_score'] = 0
    for feature, weight in weights.items():
        df['priority_score'] += df[feature] * (weight / total_weights)
    
    return df


def update_dataframe_with_results(df: pd.DataFrame, model: pulp.LpProblem, gets_pct: dict, total_budget: float, max_increase: int) -> Tuple[pd.DataFrame, dict]:
    """
    Updates the DataFrame with the optimization results from a solved PuLP model.

    This function adds two new columns:
    - 'proposed_increase_pct': The integer percentage increase for each employee.
    - 'new_gross_base_salary': The new salary after applying the increase.
    """
    # creating a copy to avoid modifying the original DataFrame, then creating new columns with default values
    df_results = df.copy()
    df_results['proposed_increase_pct'] = 0
    df_results['new_gross_base_salary'] = df_results['gross_base_salary']

    # then, we will ceck if the model has an optimal solution
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Optimal solution not found. Returning original DataFrame.")
        return None, None

    # retrieving the range of percentage points from the variable keys
    pct_points = list(range(1, max_increase + 1))
    for i in df_results.index:
        total_increase = sum(gets_pct[i][p].varValue for p in pct_points)
        df_results.loc[i, 'proposed_increase_pct'] = int(total_increase)
    
    # calculate the new salary based on the proposed increase
    df_results['recommended_increase'] = df_results['gross_base_salary'] * (df_results['proposed_increase_pct'] / 100.0)
    df_results['new_gross_base_salary'] = df_results['gross_base_salary'] + df_results['recommended_increase']
    
    # running the summary
    summary = {
        "budget_limit": total_budget,
        "total_cost": (df_results['new_gross_base_salary'] - df_results['gross_base_salary']).sum(),
        "employees_with_increase": (df_results['recommended_increase'] > 0).sum(),
        "total_employees": len(df_results),
        "avg_increase_all": df_results['recommended_increase'].mean(),
        "avg_increase_merited": df_results[df_results['recommended_increase'] > 0]['recommended_increase'].mean(),
        "avg_increase_pct_all": df_results['proposed_increase_pct'].mean(),
        "avg_increase_pct_merited": df_results[df_results['recommended_increase'] > 0]['proposed_increase_pct'].mean(),
    }
    return df_results, summary


def get_components_for_impact_plots(results_df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    impact_df = results_df.copy()
    component_cols = weights.keys()
    for col in component_cols:
        # the weighted score is the feature's value multiplied by the slider's value
        impact_df[f'w_{col}'] = impact_df[col] * weights[col]
    
    return impact_df


def get_components_for_sunburst_plot(results_df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    # melting data into a long format suitable for Plotly
    # aggregating data for the visualization
    sunburst_data = get_components_for_impact_plots(results_df, weights)
    weighted_cols = [c for c in sunburst_data.columns if c.startswith('w_')]
    
    sunburst_data = sunburst_data.groupby('talent_segment')[weighted_cols].sum().reset_index().melt(
        id_vars=['talent_segment'],
        var_name='Component',
        value_name='Total Weighted Score'
    )
    
    # cleaning up the component names for the legend
    sunburst_data['Component'] = sunburst_data['Component'].str.replace('w_', '').replace(RENAMED_LABELS)
    return sunburst_data


def get_components_for_budget_allocation_plot(results_df, weights):
    df_components = get_components_for_impact_plots(results_df, weights)
    weighted_cols = [c for c in df_components.columns if c.startswith('w_')]
    df_components['primary_driver'] = df_components[weighted_cols].idxmax(axis=1).str.replace('w_', '').replace(RENAMED_LABELS)
    
    # aggregate the cost by the primary driver
    return df_components.groupby('primary_driver')['recommended_increase'].sum().reset_index()

    
# optimization functions
def get_single_objective_optimization_model(df: pd.DataFrame, total_budget: float, alpha: float = 0.5, max_increase_percentage_bound: int = 15) -> Tuple[pulp.LpProblem, pulp.LpVariable, pulp.LpVariable]:
    employees = df.index.tolist()
    # creating our model as a maximization problem
    model = pulp.LpProblem("Salary_Increase_Optimization", pulp.LpMaximize)
    
    # decision variables
    # first, we will allow any salary increase between 0% and 15% as integers (i.e., no increases such as 3.63%)
    increase_pct = pulp.LpVariable.dicts("IncreasePct", employees, lowBound=0, upBound=max_increase_percentage_bound, cat=pulp.LpInteger)
    
    # second, not all employees will get a salary increase.
    # the i-th employee will either get an increase (gets_increase[i] = 1) or not (gets_increase[i] = 0)
    gets_increase = pulp.LpVariable.dicts("GetsIncrease", employees, cat=pulp.LpBinary)

    # with that, we can set the optimization problem.
    # first, we want to maximize the increases for all employees (meaning less people will get more increases)
    # second, we want to give increases to as much as people as possible (meaning less individual increases, but more people will be affected)
    # the balance between both is set by alpha
    objective_magnitude = pulp.lpSum([df.loc[i, 'priority_score'] * increase_pct[i] for i in employees])
    objective_breadth = pulp.lpSum([df.loc[i, 'priority_score'] * gets_increase[i] for i in employees])
    
    # pre-calculate the normalization constants
    norm_breadth = df['priority_score'].sum()
    norm_magnitude = df['priority_score'].sum() * max_increase_percentage_bound

    model += (alpha * (objective_magnitude / norm_magnitude) +
              (1 - alpha) * (objective_breadth / norm_breadth)), "Normalized_Balanced_Objective"
    
    # then, we can define the constraints.
    # first, we must not exceed the total budget.
    model += pulp.lpSum([df.loc[i, 'gross_base_salary'] * (increase_pct[i] / 100.0) for i in employees]) <= total_budget, "Budget_Constraint"

    # second, we need to match gets_increase and increase_pct
    # that is, someone with gets_increase[i] = 1 must have increase_pct[i] > 0
    # also, someone with gets_increase[i] = 0 must have increase_pct[i] == 0
    for i in employees:
        model += increase_pct[i] <= max_increase_percentage_bound * gets_increase[i], f"Link_Upper_{i}"
        model += increase_pct[i] >= 1 * gets_increase[i], f"Link_Lower_{i}"

    # third, the sum of all increases for each lead's team must not exceed that lead's specific monthly budget.
    unique_leads = df['lead_id'].unique()
    for lead in unique_leads:
        # identify all employees who report to the current lead.
        employees_of_lead = df[df['lead_id'] == lead].index.tolist()

        # retrieving the budget for this lead (it's the same for all employees on the team).
        lead_budget = df.loc[employees_of_lead[0], 'lead_monthly_budget']

        # get the sum of the new salaries for this team
        sum_of_new_increases = pulp.lpSum([increase_pct[i] for i in employees_of_lead])

        # adding the constraint to the model.
        model += sum_of_new_increases <= lead_budget, f"Lead_Budget_Constraint_{lead}"
        
    return model, gets_increase, increase_pct


def get_diminishing_returns_model(df: pd.DataFrame, total_budget: float, max_increase: int = 15) -> Tuple[pulp.LpProblem, pulp.LpVariable]:
    employees = df.index.tolist()
    
    # creating our model as a maximization problem
    model = pulp.LpProblem("Diminishing_Returns_Salary_Opt", pulp.LpMaximize)
    
    # first, we will create weights for each percentage point (e.g., using a sqrt curve)
    # the first percent is worth 1.0, the second is worth ~0.7, etc.
    # our idea is to incentivize the solver to distribute the budget more broadly
    # as it will always prefer to "buy" the high-value initial percentage points for many employees
    # over the low-value final percentage points for just a few
    pct_points = list(range(1, max_increase + 1))
    diminishing_weights = {p: np.sqrt(p) - np.sqrt(p-1) for p in pct_points}
    
    # decision variables
    # first, we will allow any salary increase between 0% and 15% as integers (i.e., no increases such as 3.63%)
    gets_pct = pulp.LpVariable.dicts("GetsPct", (employees, pct_points), cat=pulp.LpBinary)
    
    # with that, we can set the optimization problem.
    # we want to maximize the priority-weighted sum of the value of each percentage point granted
    model += pulp.lpSum([
        df.loc[i, 'priority_score'] * diminishing_weights[p] * gets_pct[i][p]
        for i in employees for p in pct_points
    ]), "Maximize_Diminishing_Weighted_Impact"
    
    # then, we can define the constraints.
    # first, we must not exceed the total budget.
    model += pulp.lpSum([
        df.loc[i, 'gross_base_salary'] * (gets_pct[i][p] / 100.0)
        for i in employees for p in pct_points
    ]) <= total_budget, "Budget_Constraint"
    
    # second, we also need to keep contiguity
    # that is, we should not give 3% increase if we also can't give 1% of 2%
    # if we don't have it, the model might understand that giving higher increases is not that impactful
    for i in employees:
        for p in range(2, max_increase + 1):
            model += gets_pct[i][p] <= gets_pct[i][p-1], f"Contiguity_{i}_{p}"

        # third, the sum of all increases for each lead's team must not exceed that lead's specific monthly budget.
    unique_leads = df['lead_id'].unique()
    for lead in unique_leads:
        # identify all employees who report to the current lead.
        employees_of_lead = df[df['lead_id'] == lead].index.tolist()

        # retrieving the budget for this lead (it's the same for all employees on the team).
        lead_budget = df.loc[employees_of_lead[0], 'lead_monthly_budget']

        # get the sum of the new salaries for this team
        sum_of_new_increases = pulp.lpSum([
            pulp.lpSum([df.loc[i, 'gross_base_salary'] * (gets_pct[i][p] / 100.0) for p in pct_points]) for i in employees_of_lead
        ])

        # adding the constraint to the model.
        model += sum_of_new_increases <= lead_budget, f"Lead_Budget_Constraint_{lead}"
    
    return model, gets_pct

