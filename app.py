%%writefile app.py
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EV Battery Enclosure Optimizer", page_icon="🔋", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('surrogate_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('fea_results.csv'), pd.read_csv('fea_benchmarks.csv')

try:
    model = load_model()
    df, df_bench = load_data()
except Exception as e:
    st.error("⚠️ Missing dataset or model! Run 'prepare_data.py' and 'train_model.py' first.")
    st.info(f"Details: {e}")
    st.stop()

st.title("⚡ EV Battery Pack Enclosure: ANSYS Surrogate Optimizer")
st.markdown("**Loading Condition:** 10 kPa Uniform Upward Pressure on Baseplate | **Lateral vs. Transverse Rib Optimization**")
st.divider()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🛠️ CAD & Material Controls")

selected_material = st.sidebar.selectbox(
    "Aluminum Alloy Grade",
    options=["6061-T6", "5052-H32"]
)

selected_pattern = st.sidebar.selectbox(
    "Select Rib Pattern",
    options=["Lateral", "Transverse"]
)

match selected_pattern:
    case "Lateral":
        pattern_title = "Lateral Rib Pattern (Longitudinal)"
        num_ribs = st.sidebar.slider("Number of Lateral Ribs", min_value=2, max_value=4, value=2, step=1)
        pattern_notes = "2–4 longitudinal ribs along the tray span; all 7 ANSYS runs stay under 2.0 mm deformation."
    case "Transverse":
        pattern_title = "Transverse Rib Pattern (Cross-Width)"
        num_ribs = st.sidebar.slider("Number of Transverse Ribs", min_value=6, max_value=8, value=7, step=1)
        pattern_notes = "6–8 cross-width ribs across the shorter baseplate axis."
    case _:
        st.error("Unsupported rib pattern selected.")
        st.stop()

rib_thickness = st.sidebar.slider("Rib Thickness (mm)", min_value=3.0, max_value=5.0, value=3.0, step=0.5)
rib_height = st.sidebar.slider("Rib Height (mm)", min_value=10.0, max_value=30.0, value=20.0, step=5.0)

st.sidebar.divider()
st.sidebar.header("📏 Safety Line Threshold")
SAFETY_LINE_MM = st.sidebar.slider(
    "Max Safe Deformation Line (mm)",
    min_value=1.00, max_value=2.50, value=2.00, step=0.05,
    help="Models UNDER this horizontal line are SAFE; models ABOVE this line bend too much."
)
MIN_SAFE_FOS = 2.0

# --- PREDICTION ENGINE ---
input_df = pd.DataFrame([{
    'Rib_Pattern': selected_pattern,
    'Material': selected_material,
    'Num_Ribs': num_ribs,
    'Rib_Thickness_mm': rib_thickness,
    'Rib_Height_mm': rib_height
}])

pred_mass, pred_stress, pred_def = model.predict(input_df)[0]

base_row = df_bench[(df_bench['Rib_Pattern'] == 'Without Ribs') & (df_bench['Material'] == selected_material)].iloc[0]
mass_diff = pred_mass - base_row['Mass_kg']
def_reduction_pct = ((base_row['Max_Deformation_mm'] - pred_def) / base_row['Max_Deformation_mm']) * 100

yield_strength = 292.83 if selected_material == "6061-T6" else 192.13
max_safe_stress = yield_strength / MIN_SAFE_FOS
fos = yield_strength / pred_stress

# --- SAFETY CHECKS ---
is_under_safety_line = pred_def <= SAFETY_LINE_MM
is_stress_safe = pred_stress <= max_safe_stress
is_fos_safe = fos >= MIN_SAFE_FOS
is_overall_safe = is_under_safety_line and is_stress_safe and is_fos_safe

# --- METRICS DISPLAY ---
st.header(f"📊 Predicted Performance ({pattern_title} — {selected_material})")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("1. Enclosure Mass", f"{pred_mass:.3f} kg", delta=f"{mass_diff:+.3f} kg vs unribbed", delta_color="off")
    st.info("ℹ️ Trade-off weight")

with c2:
    st.metric("2. Max Von Mises Stress", f"{pred_stress:.2f} MPa", delta=f"Safe Limit ≤ {max_safe_stress:.1f} MPa", delta_color="off")
    if is_stress_safe:
        st.success("✅ **SAFE** (Below yield limit)")
    else:
        st.error("❌ **NOT SAFE** (Stress too high)")

with c3:
    st.metric("3. Max Deformation", f"{pred_def:.3f} mm", delta=f"-{def_reduction_pct:.1f}% vs unribbed", delta_color="off")
    if is_under_safety_line:
        st.success(f"✅ **SAFE** (Under {SAFETY_LINE_MM:.2f} mm line)")
    else:
        st.error(f"❌ **NOT SAFE** (Above {SAFETY_LINE_MM:.2f} mm line)")

with c4:
    st.metric("4. Factor of Safety (FOS)", f"{fos:.2f}", delta="Target ≥ 2.0", delta_color="off")
    if is_fos_safe:
        st.success("✅ **SAFE** (FOS ≥ 2.0)")
    else:
        st.error("❌ **NOT SAFE** (FOS < 2.0)")

# --- ULTIMATE VERDICT ---
if is_overall_safe:
    st.success(
        f"### ✅ ULTIMATE MODEL IS SAFE\n"
        f"Your selected model (**{pred_def:.3f} mm** deformation, **FOS {fos:.2f}**) sits **UNDER the {SAFETY_LINE_MM:.2f} mm Safety Line** in the Green Safe Zone."
    )
else:
    st.error(
        f"### ❌ ULTIMATE MODEL IS NOT SAFE\n"
        f"Your selected model (**{pred_def:.3f} mm** deformation) is **ABOVE the {SAFETY_LINE_MM:.2f} mm Safety Line** in the Red Unsafe Zone. "
        f"Switch to **Lateral** ribs or increase rib height/thickness to bring the star below the line."
    )

st.divider()

# --- SCATTER PLOT WITH HORIZONTAL SAFETY LINE ---
st.header("🎯 Mass vs. Deformation Trade-Off (With Safety Line)")
plot_df = pd.concat([df, df_bench], ignore_index=True)
sub_df = plot_df[plot_df['Material'] == selected_material]

x_min = min(sub_df['Mass_kg'].min(), pred_mass) - 0.25
x_max = max(sub_df['Mass_kg'].max(), pred_mass) + 0.35
y_min = 0.50
y_max = max(sub_df['Max_Deformation_mm'].max(), pred_def, SAFETY_LINE_MM) + 0.30

fig = px.scatter(
    sub_df,
    x="Mass_kg",
    y="Max_Deformation_mm",
    color="Rib_Pattern",
    size="Max_Von_Mises_MPa",
    hover_data=["Num_Ribs", "Rib_Thickness_mm", "Rib_Height_mm", "Max_Von_Mises_MPa"],
    labels={
        "Mass_kg": "Total Enclosure Mass (kg) [Left = Lighter]",
        "Max_Deformation_mm": "Max Deformation (mm) [Down = Stiffer / Safer]",
        "Rib_Pattern": "Design Type"
    },
    title=f"Designs UNDER the Red Line Are SAFE ({selected_material} Alloy)",
    template="plotly_white"
)

# Green shaded region UNDER the safety line
fig.add_hrect(
    y0=y_min,
    y1=SAFETY_LINE_MM,
    fillcolor="rgba(46, 204, 113, 0.14)",
    line_width=0,
    layer="below",
    annotation_text="✅ SAFE ZONE (Under Safety Line)",
    annotation_position="bottom left",
    annotation_font=dict(color="green", size=13)
)

# Red shaded region ABOVE the safety line
fig.add_hrect(
    y0=SAFETY_LINE_MM,
    y1=y_max,
    fillcolor="rgba(231, 76, 60, 0.10)",
    line_width=0,
    layer="below",
    annotation_text="❌ UNSAFE ZONE (Above Safety Line — Excessive Bending)",
    annotation_position="top left",
    annotation_font=dict(color="red", size=13)
)

# Horizontal Safety Line
fig.add_hline(
    y=SAFETY_LINE_MM,
    line_dash="dash",
    line_color="red",
    line_width=3,
    annotation_text=f"SAFETY LINE ({SAFETY_LINE_MM:.2f} mm)",
    annotation_position="top right",
    annotation_font=dict(color="red", size=13)
)

# Selected Model Star (Green if under line, Red if above line)
fig.add_scatter(
    x=[pred_mass],
    y=[pred_def],
    mode="markers+text",
    marker=dict(
        size=24,
        color="green" if is_overall_safe else "red",
        symbol="star",
        line=dict(width=2, color="black")
    ),
    text=["✅ SAFE MODEL" if is_overall_safe else "❌ UNSAFE MODEL"],
    textposition="top center",
    name="Selected Model"
)

fig.update_xaxes(range=[x_min, x_max])
fig.update_yaxes(range=[y_min, y_max])

st.plotly_chart(fig, use_container_width=True)
