"""
Cloud eQMS ROI Simulator - Monte Carlo Engine (MBA TQM Thesis , University of Piraeus)
Author: Dimitrios Vergis
Thesis: Digitalization of Quality Processes in the Pharmaceutical Industry
through eQMS
Description:
    Enterprise simulation with configurable inputs to adopt to different business scenarios and regions based on complexity driven logic.
    Features HC3 robust regression, Module-based analytics, and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms

# =======================================================
# 0. CONFIGURATION & VISUAL THEME
# =======================================================
UNIPI_PALETTE = ['#7f8c8d', '#003366']  # Legacy (Grey) vs Digital (Navy)
sns.set_palette(UNIPI_PALETTE)
sns.set_style("whitegrid")

# =======================================================
# 1. BASELINE PROCESS METADATA (SME Validated)
# =======================================================
# Updated with specific volumes for Work Instructions and Forms
PROCESS_METADATA = [
    # Module, Event Name, Annual Vol, Leg Hrs, Dig Hrs, Leg Days, Dig Days, Complexity

    # --- TRAINING ---
    ("Training", "Training Assignment (Read&Und)", 1500, 0.5, 0.1, 5.0, 1.0, 1.0),
    ("Training", "OJT Assessment / Qual", 200, 2.5, 1.0, 10.0, 3.0, 2.0),
    ("Training", "Ext. Training Verification", 100, 1.0, 0.3, 7.0, 2.0, 1.0),

    # --- DOCUMENTS ---
    ("Documents", "SOP Revision (Minor/Admin)", 50, 6.0, 3.0, 20.0, 10.0, 2.0),
    ("Documents", "SOP Creation / Major Rev", 30, 15.0, 8.0, 45.0, 20.0, 3.5),
    ("Documents", "SOP Periodic Review", 80, 8.0, 4.0, 30.0, 15.0, 2.5),
    ("Documents", "Work Instruction (New/Rev)", 500, 4.0, 2.0, 15.0, 5.0, 1.5),
    ("Documents", "Form/Logbook Issuance & Rev", 1000, 0.5, 0.1, 2.0, 0.5, 1.0),

    # --- DEVIATIONS ---
    ("Deviations", "Deviation (Minor / Event)", 500, 8.0, 5.0, 25.0, 15.0, 2.0),
    ("Deviations", "Deviation (Major / Non-Conf)", 50, 20.0, 12.0, 45.0, 25.0, 3.5),
    ("Deviations", "Deviation (Critical)", 10, 40.0, 25.0, 60.0, 30.0, 5.0),

    # --- OOS / OOT ---
    ("OOS/OOT", "OOT / Lab Error (Minor)", 500, 10.0, 6.0, 15.0, 5.0, 2.0),
    ("OOS/OOT", "OOS Confirmed (Major)", 100, 25.0, 15.0, 30.0, 15.0, 4.0),
    ("OOS/OOT", "OOS Critical / Field Alert", 10, 80.0, 50.0, 45.0, 20.0, 5.0),

    # --- CAPA ---
    ("CAPA", "CAPA Plan & Execution", 40, 15.0, 10.0, 60.0, 40.0, 3.5),
    ("CAPA", "Effectiveness Check (EC)", 30, 8.0, 4.0, 90.0, 45.0, 3.0),

    # --- CHANGE CONTROL ---
    ("Change Ctrl", "CC (Like-for-Like / Admin)", 50, 6.0, 3.0, 15.0, 5.0, 2.0),
    ("Change Ctrl", "CC (Minor / Process Adj)", 30, 12.0, 6.0, 30.0, 15.0, 3.0),
    ("Change Ctrl", "CC (Major / Equipment)", 15, 30.0, 18.0, 90.0, 45.0, 4.5),
    ("Change Ctrl", "CC (Emergency / Temporary)", 25, 20.0, 12.0, 5.0, 2.0, 4.5),

    # --- COMPLAINTS ---
    ("Complaints", "Complaint (Non-Medical)", 60, 12.0, 7.0, 30.0, 15.0, 3.0),
    ("Complaints", "Complaint (Medical / AE)", 5, 25.0, 15.0, 15.0, 5.0, 5.0),

    # --- AUDITS ---
    ("Audits", "Internal Audit (Self-Insp)", 4, 40.0, 25.0, 30.0, 15.0, 3.5),
    ("Audits", "Supplier Audit (Desk/Remote)", 20, 10.0, 5.0, 30.0, 10.0, 2.5),
    ("Audits", "Supplier Audit (On-Site)", 30, 50.0, 30.0, 45.0, 20.0, 4.0),
    ("Audits", "Regulatory Insp. Prep", 5, 200.0, 120.0, 15.0, 10.0, 5.0),
]

# Initialize DataFrame
DEFAULT_DF = pd.DataFrame(PROCESS_METADATA, columns=[
    "Module", "Event_Name", "Annual_Volume", "Legacy_Hours", "Target_Dig_Hours",
    "Legacy_Cycle_Days", "Target_Dig_Cycle_Days", "Complexity"
])


# =======================================================
# 2. CORE SIMULATION ENGINE
# =======================================================
@st.cache_data
def run_simulation(hourly_rate, n_events, process_df, penalties, currency_symbol, savings_factor):
    # Weights based on Annual Volume
    total_vol = process_df['Annual_Volume'].sum()
    weights = process_df['Annual_Volume'] / total_vol

    data_rows = []

    # Penalties
    p_capa = float(penalties.get('CAPA', 0.0))
    p_cc = float(penalties.get('CC', 0.0))
    p_doc = float(penalties.get('DOC', 0.0))
    p_comp = float(penalties.get('COMPLAINT', 0.0))

    # Stochastic Selection
    chosen_indices = np.random.choice(process_df.index, size=n_events, p=weights)

    for i, idx in enumerate(chosen_indices):
        row = process_df.iloc[idx]
        evt_id = f"EVT-{i + 1:05d}"

        module = row['Module']
        proc_name = row['Event_Name']

        # --- COMPLEXITY DRIVER ---
        base_comp = float(row['Complexity'])
        comp_noise = np.random.lognormal(0, 0.2) - 1.0
        complexity = max(1.0, min(5.0, base_comp + comp_noise))

        # --- DEPARTMENTS ---
        avg_depts = 1.0 + (complexity * 0.7)
        dept_count = max(1, min(10, np.random.poisson(avg_depts)))

        # --- LINKAGES ---
        prob_link = min(0.95, complexity * 0.18)

        prob_capa = 0.0 if module == 'CAPA' else prob_link
        prob_cc = min(0.95, prob_link * 1.2) if module == 'Documents' else prob_link
        prob_doc = 0.0 if module == 'Documents' else prob_link

        linked_capa = 1 if np.random.random() < prob_capa else 0
        linked_cc = 1 if np.random.random() < prob_cc else 0
        linked_doc = 1 if np.random.random() < prob_doc else 0
        linked_comp = 1 if np.random.random() < (prob_link * 0.5) else 0

        base_link_penalty = (linked_capa * p_capa) + (linked_cc * p_cc) + \
                            (linked_doc * p_doc) + (linked_comp * p_comp)

        # --- STATE GENERATION (Matched Pairs) ---
        for state in [0, 1]:
            if state == 0:  # LEGACY
                noise = np.random.normal(0, 1.2)
                base_h = float(row['Legacy_Hours'])
                comp_factor = complexity * 5.0
                dept_factor = dept_count * 3.0
                link_penalty = base_link_penalty

                target_cycle = float(row['Legacy_Cycle_Days'])
                cycle_days = int(np.random.lognormal(mean=np.log(target_cycle), sigma=0.5))

            else:  # DIGITAL
                noise = np.random.normal(0.5, 1.2)

                leg_h = float(row['Legacy_Hours'])
                dig_h = float(row['Target_Dig_Hours'])
                base_h = leg_h - ((leg_h - dig_h) * savings_factor)

                comp_mult = 5.0 - (3.0 * savings_factor)
                comp_factor = complexity * comp_mult
                dept_mult = 3.0 - (2.2 * savings_factor)
                dept_factor = dept_count * dept_mult
                link_penalty = base_link_penalty * (1 - savings_factor)

                target_cycle = float(row['Target_Dig_Cycle_Days'])
                cycle_days = int(np.random.lognormal(mean=np.log(target_cycle), sigma=0.25))

            cycle_days = max(1, cycle_days)
            if total_hours := max(0.5, base_h + comp_factor + dept_factor + link_penalty + noise): pass
            cost = total_hours * hourly_rate

            data_rows.append([
                evt_id, module, proc_name, state, complexity, dept_count,
                linked_capa, linked_cc, linked_doc, linked_comp,
                cycle_days, round(total_hours, 1), round(cost, 2)
            ])

    # Consolidation
    cols = ['Event_ID', 'Module', 'Process_Name', 'System_State_Binary', 'Complexity', 'Dept_Count',
            'Link_CAPA', 'Link_CC', 'Link_Doc', 'Link_Comp',
            'Cycle_Days', 'Manhours', f'Total_Cost_{currency_symbol}']
    df = pd.DataFrame(data_rows, columns=cols)

    # Regression
    X = df[['System_State_Binary', 'Complexity', 'Dept_Count', 'Link_CAPA', 'Link_CC', 'Link_Doc', 'Link_Comp']]
    X = sm.add_constant(X)
    Y = df[f'Total_Cost_{currency_symbol}']

    model_ols = sm.OLS(Y, X).fit()
    try:
        _, white_p, _, _ = sms.het_white(model_ols.resid, model_ols.model.exog)
    except:
        white_p = 0.0

    model_hc3 = sm.OLS(Y, X).fit(cov_type='HC3')
    beta_1 = model_hc3.params['System_State_Binary']
    annual_savings = beta_1 * n_events

    # --- DIAGNOSTICS (With Outlier Filtering for Charts) ---
    resid = model_ols.resid
    fitted = model_ols.fittedvalues

    # Filter 1st-99th percentile for cleaner plots
    mask = (resid > resid.quantile(0.01)) & (resid < resid.quantile(0.99))
    resid_filtered = resid[mask]
    fitted_filtered = fitted[mask]
    Y_filtered = Y[mask]

    # Q-Q Plot (Filtered)
    fig_qq = sm.qqplot(resid_filtered, line='s')

    # Residuals vs Fitted (Filtered)
    fig_resid, ax_resid = plt.subplots(figsize=(10, 6))
    ax_resid.scatter(fitted_filtered, resid_filtered, alpha=0.5, color=UNIPI_PALETTE[1])
    ax_resid.axhline(0, color='red', linestyle='--')
    ax_resid.set_title('Residuals vs. Fitted Values (Outliers Removed)')

    # OLS Predicted vs Observed (Filtered) - NEW GRAPH
    fig_ols, ax_ols = plt.subplots(figsize=(8, 6))
    ax_ols.scatter(Y_filtered, fitted_filtered, alpha=0.3, color='#003366', edgecolors='w')
    min_val = min(Y_filtered.min(), fitted_filtered.min())
    max_val = max(Y_filtered.max(), fitted_filtered.max())
    ax_ols.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax_ols.set_xlabel('Observed Cost (Actual)')
    ax_ols.set_ylabel('Predicted Cost (Model)')
    ax_ols.set_title('OLS Fit: Observed vs. Predicted (Outliers Removed)')

    return df, model_hc3, beta_1, annual_savings, fig_qq, fig_resid, fig_ols, white_p


# =======================================================
# 3. USER INTERFACE
# =======================================================
st.set_page_config(page_title="Quality ROI Calculator", layout="wide", initial_sidebar_state="expanded")

if 'simulation_run' not in st.session_state:
    st.session_state['simulation_run'] = False

st.title("Cloud eQMS ROI Simulator")
st.caption("Strategic Decision Support Tool | Powered by Monte Carlo Simulation | UNIPI MBA TQM Thesis | Author:D.Vergis Supervisor:G.Bohoris")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Financial Parameters")
    HOURLY_RATE = st.number_input("Avg Labor Rate (Per Hour)", min_value=1.0, value=50.0, step=5.0)
    CURRENCY_SYMBOL = st.text_input("Currency", "€")

    st.markdown("---")
    st.header("2. Efficiency Assumptions")
    SAVINGS_FACTOR = st.slider("Man-Hour Savings Realization (%)", 0, 100, 50)

    st.markdown("---")
    st.header("3. Risk & COPQ")
    AVG_FAILURE_COST = st.number_input("Avg Cost of Failure", value=5000.0, step=500.0)
    LEGACY_FAIL_RATE = st.slider("Legacy Failure Rate (%)", 0.0, 10.0, 2.5)
    DIGITAL_FAIL_RATE = st.slider("Digital Failure Rate (%)", 0.0, 10.0, 0.5)

    st.markdown("---")
    st.header("4. Linkage Costs")
    PENALTIES = {
        'CAPA': float(st.number_input("CAPA Penalty", value=5.0, min_value=0.0, step=0.5)),
        'CC': float(st.number_input("CC Penalty", value=6.0, min_value=0.0, step=0.5)),
        'DOC': float(st.number_input("Doc Penalty", value=2.5, min_value=0.0, step=0.5)),
        'COMPLAINT': float(st.number_input("Complaint Penalty", value=4.0, min_value=0.0, step=0.5))
    }

# --- MAIN INPUT ---
st.header("5. Enterprise QMS Configuration")
st.markdown("_SME staging table for a full-scale pharma eQMS rollout-configurable_")

editable_df = st.data_editor(
    DEFAULT_DF,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Module": st.column_config.SelectboxColumn("Module",
                                                   options=["Training", "Documents", "Deviations", "OOS/OOT", "CAPA",
                                                            "Change Ctrl", "Complaints", "Audits", "Supplier"],
                                                   required=True),
        "Annual_Volume": st.column_config.NumberColumn("Volume", min_value=0, step=1),
        "Complexity": st.column_config.NumberColumn("Complexity (1-5)", min_value=1.0, max_value=5.0, step=0.5,
                                                    format="%.1f")
    }
)

total_events = int(editable_df['Annual_Volume'].sum())
st.info(f"📊 **Total Enterprise Volume:** {total_events:,} events per year")

if st.button("▶️ Run Enterprise Simulation", type="primary"):
    if total_events <= 0:
        st.error("Volume must be > 0")
        st.stop()

    SIM_LIMIT = 5000
    if total_events > SIM_LIMIT:
        sim_n = SIM_LIMIT
        scale_factor = total_events / SIM_LIMIT
        st.warning(
            f"⚠️ Volume is high ({total_events:,}). Simulating {SIM_LIMIT} representative events and scaling results by {scale_factor:.2f}x.")
    else:
        sim_n = total_events
        scale_factor = 1.0

    with st.spinner(f'Simulating {sim_n} events...'):
        df_res, model, beta_1, ann_save, fig_qq, fig_res, fig_ols, white_p = run_simulation(
            HOURLY_RATE, sim_n, editable_df, PENALTIES, CURRENCY_SYMBOL, SAVINGS_FACTOR / 100.0
        )

        final_annual_savings = ann_save * scale_factor

        st.session_state.update({
            'df_res': df_res, 'model': model, 'beta_1': beta_1, 'ann_save': final_annual_savings,
            'fig_qq': fig_qq, 'fig_res': fig_res, 'fig_ols': fig_ols, 'white_p': white_p, 'simulation_run': True,
            'scale_factor': scale_factor
        })

# --- RESULTS ---
if st.session_state['simulation_run']:
    st.divider()
    t1, t2, t3, t4, t5 = st.tabs(["📊 ROI Overview", "🧩 ROI by Module", "🛡️ Risk", "📈 Diagnostics", "📋 Data"])

    scale = st.session_state.get('scale_factor', 1.0)

    # TAB 1: Overview
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Saving / Event", f"{CURRENCY_SYMBOL}{abs(st.session_state['beta_1']):,.2f}")
        c2.metric("Total Annual ROI", f"{CURRENCY_SYMBOL}{abs(st.session_state['ann_save']):,.0f}")

        avg_leg = st.session_state['df_res'][st.session_state['df_res']['System_State_Binary'] == 0][
            'Cycle_Days'].mean()
        avg_dig = st.session_state['df_res'][st.session_state['df_res']['System_State_Binary'] == 1][
            'Cycle_Days'].mean()
        c3.metric("Avg Speed Improvement", f"{avg_leg - avg_dig:.1f} Days")

        fig, ax = plt.subplots(figsize=(8, 4))
        # EXCLUDING OUTLIERS for better visual scale
        sns.boxplot(x='System_State_Binary', y=f'Total_Cost_{CURRENCY_SYMBOL}',
                    data=st.session_state['df_res'], palette=UNIPI_PALETTE, ax=ax, showfliers=False)
        ax.set_xticklabels(['Legacy (Paper)', 'Digital (eQMS)'])
        ax.set_title("Cost Variance Analysis (Outliers Hidden for Clarity)")
        st.pyplot(fig)

    # TAB 2: ROI by Module
    with t2:
        st.subheader("Financial Impact by Quality Module")

        df_mod = st.session_state['df_res']
        # Group & Sum (and scale if needed)
        leg_cost = df_mod[df_mod['System_State_Binary'] == 0].groupby('Module')[
                       f'Total_Cost_{CURRENCY_SYMBOL}'].sum() * scale
        dig_cost = df_mod[df_mod['System_State_Binary'] == 1].groupby('Module')[
                       f'Total_Cost_{CURRENCY_SYMBOL}'].sum() * scale
        counts = (df_mod.groupby('Module')['Event_ID'].count() / 2) * scale

        mod_summary = pd.DataFrame({'Legacy Total': leg_cost, 'Digital Total': dig_cost, 'Volume': counts})
        mod_summary['Net Annual Savings'] = mod_summary['Legacy Total'] - mod_summary['Digital Total']
        mod_summary['Avg Saving per Event'] = mod_summary['Net Annual Savings'] / mod_summary['Volume']
        mod_summary = mod_summary.sort_values('Net Annual Savings', ascending=False)

        st.dataframe(
            mod_summary[['Volume', 'Net Annual Savings', 'Avg Saving per Event']].style.format({
                'Volume': '{:.0f}',
                'Net Annual Savings': f'{CURRENCY_SYMBOL}{{:.0f}}',
                'Avg Saving per Event': f'{CURRENCY_SYMBOL}{{:.2f}}'
            }),
            use_container_width=True
        )

        st.markdown("#### Annual Spend Comparison: Legacy vs. Digital")

        # Prepare data for plotting
        plot_data = mod_summary[['Legacy Total', 'Digital Total']].reset_index().melt(
            id_vars='Module', var_name='State', value_name='Annual Cost'
        )
        plot_data['State'] = plot_data['State'].replace(
            {'Legacy Total': 'Legacy (Paper)', 'Digital Total': 'Digital (eQMS)'})

        fig_mod, ax_mod = plt.subplots(figsize=(10, 6))
        sns.barplot(data=plot_data, x='Module', y='Annual Cost', hue='State', palette=UNIPI_PALETTE, ax=ax_mod)
        ax_mod.set_ylabel(f"Total Annual Cost ({CURRENCY_SYMBOL})")
        ax_mod.set_title("Operational Cost Comparison by Module")
        st.pyplot(fig_mod)

    # TAB 3: Risk
    with t3:
        leg_risk = total_events * (LEGACY_FAIL_RATE / 100) * AVG_FAILURE_COST
        dig_risk = total_events * (DIGITAL_FAIL_RATE / 100) * AVG_FAILURE_COST

        k1, k2, k3 = st.columns(3)
        k1.metric("Legacy Risk", f"{CURRENCY_SYMBOL}{leg_risk:,.0f}")
        k2.metric("Digital Risk", f"{CURRENCY_SYMBOL}{dig_risk:,.0f}")
        k3.metric("Avoidance Benefit", f"{CURRENCY_SYMBOL}{leg_risk - dig_risk:,.0f}")

        eff_leg = st.session_state['df_res'][st.session_state['df_res']['System_State_Binary'] == 0][
                      f'Total_Cost_{CURRENCY_SYMBOL}'].sum() * scale
        eff_dig = st.session_state['df_res'][st.session_state['df_res']['System_State_Binary'] == 1][
                      f'Total_Cost_{CURRENCY_SYMBOL}'].sum() * scale

        fig_r, ax_r = plt.subplots(figsize=(6, 4))
        pd.DataFrame({
            'State': ['Legacy', 'Digital'],
            'Labor Cost': [eff_leg, eff_dig],
            'Risk Liability': [leg_risk, dig_risk]
        }).set_index('State').plot(kind='bar', stacked=True, color=['#7f8c8d', '#f39c12'], ax=ax_r)
        ax_r.set_ylabel(f"Total Cost of Ownership ({CURRENCY_SYMBOL})")
        st.pyplot(fig_r)

    # TAB 4: Diagnostics
    with t4:
        st.markdown(f"**White Test p-value:** {st.session_state['white_p']:.4f}")
        st.caption("Note: Low p-value (<0.05) indicates Heteroscedasticity. Model uses HC3 Robust Standard Errors.")

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(st.session_state['fig_qq'])
            st.caption("**Normality Check:** Data should track the red line.")
        with c2:
            st.pyplot(st.session_state['fig_ols'])
            st.caption("**Goodness of Fit:** Points should cluster around the diagonal.")

        st.code(st.session_state['model'].summary().as_text())

    # TAB 5: Data
    with t5:
        st.dataframe(st.session_state['df_res'], use_container_width=True)
        st.download_button("📥 Download CSV", st.session_state['df_res'].to_csv(index=False).encode('utf-8'),
                           "sim_data.csv", "text/csv")
