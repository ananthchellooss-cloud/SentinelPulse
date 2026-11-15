import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentinelpulse.data import get_demo_data, get_live_weather_forecast, ABU_DHABI_AREAS, DEMO_OPERATIONAL_PROFILES
from sentinelpulse.models import score_signals, add_forecast, build_heat_risk_outlook, score_live_weather_outlook, build_predictive_intelligence

st.set_page_config(page_title="SentinelPulse | Health Early Warning", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
.main {background:#f5f7fb;}
.block-container {padding-top:1.2rem; padding-bottom:2rem;}
[data-testid="stMetric"] {background:#ffffff; border:1px solid #dce3ec; padding:14px; border-radius:12px;}
[data-testid="stMetric"] label, [data-testid="stMetric"] [data-testid="stMetricLabel"] {color:#344054 !important;}
[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#172b4d !important;}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {color:#475467 !important;}
.sp-hero {background:linear-gradient(115deg,#172b4d,#245b70);padding:22px 26px;border-radius:16px;color:white;margin-bottom:16px;}
.sp-hero h1 {color:white;margin:0;font-size:30px;}
.sp-hero p {color:#d9e7f1;margin:7px 0 0 0;}
.small-note {color:#68778a;font-size:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sp-hero"><h1>🛰️ SentinelPulse</h1><p>AI-powered multi-signal population & workforce health early warning</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Demo controls")
    days = st.slider("Historical window (days)", 30, 90, 60, 5)
    live = st.toggle("Use public historical weather", value=False, help="Fetches Open-Meteo archive weather. Workplace and health-service signals remain synthetic.")
    live_forecast_enabled = st.toggle("Enable live weather forecast", value=True, help="Fetches the next 24 hours of public forecast weather for the five Abu Dhabi areas.")
    if st.button("Refresh weather data"):
        st.cache_data.clear()
        st.rerun()
    selected_area = st.selectbox("Focus area", ["All areas"] + list(ABU_DHABI_AREAS.keys()))
    st.caption("Guest demonstration • No login required")
    st.divider()
    st.caption("Data provenance is shown in the dashboard. Synthetic operational records are not real patient or worker data.")

@st.cache_data(ttl=1800, show_spinner="Loading data and running the intelligence engine…")
def load(days, live, live_forecast_enabled):
    df, provenance = get_demo_data(days, live)
    scored = score_signals(df)
    forecasted, model_info = add_forecast(scored)
    heat_outlook, key_horizons = build_heat_risk_outlook(scored, horizons=(6, 24))
    if live_forecast_enabled:
        live_weather, live_weather_errors = get_live_weather_forecast(forecast_hours=24)
        live_outlook = score_live_weather_outlook(live_weather, scored) if not live_weather.empty else pd.DataFrame()
    else:
        live_weather, live_weather_errors, live_outlook = pd.DataFrame(), [], pd.DataFrame()
    return forecasted, provenance, model_info, heat_outlook, key_horizons, live_outlook, live_weather_errors

try:
    data, provenance, model_info, heat_outlook, key_horizons, live_outlook, live_weather_errors = load(days, live, live_forecast_enabled)
except Exception as e:
    st.error(f"Unable to initialize the demonstration: {e}")
    st.stop()

st.caption(f"**Data provenance:** {provenance}  |  **Model:** Isolation Forest + gradient-boosting service forecast")
if selected_area != "All areas":
    view = data[data.area == selected_area].copy()
else:
    view = data.copy()

latest_ts = view.timestamp.max()
latest = view[view.timestamp == latest_ts]
alerts = view[view.risk_level.isin(["Elevated","High"])]
high_count = int((latest.risk_level == "High").sum())
elevated_count = int((latest.risk_level == "Elevated").sum())
avg_risk = float(latest.fused_risk_score.mean()) if len(latest) else 0
sites = view.area.nunique()

m1,m2,m3,m4 = st.columns(4)
m1.metric("Areas monitored", sites)
m2.metric("High-risk areas now", high_count)
m3.metric("Elevated areas now", elevated_count)
m4.metric("Mean fused risk", f"{avg_risk:.2f}/1.00")

tabs = st.tabs(["Command Center","Geographic Intelligence","AI Explainability","Alerts & Actions","Model Validation","Heat Risk Outlook","Predictive AI"])

with tabs[0]:
    st.subheader("Population health signal overview")
    latest_sorted = latest.sort_values("fused_risk_score", ascending=False)
    fig = px.line(view.groupby("timestamp",as_index=False).agg(
        risk=("fused_risk_score","mean"),
        environment=("environment_score","mean"),
        service=("service_anomaly_score","mean")
    ), x="timestamp", y=["risk","environment","service"], labels={"value":"Score","variable":"Signal"})
    fig.update_layout(height=340, legend_title="", margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Latest area status**")
    cols=["area","risk_level","fused_risk_score","top_contributor","wet_bulb_approx_c","first_aid_reports","heat_related_clinic_visits"]
    st.dataframe(latest_sorted[cols].rename(columns={
        "area":"Area","risk_level":"Risk tier","fused_risk_score":"Fused score",
        "top_contributor":"Leading signal","wet_bulb_approx_c":"Approx. wet-bulb °C",
        "first_aid_reports":"Synthetic first-aid reports","heat_related_clinic_visits":"Synthetic clinic visits"
    }).round(2), width="stretch", hide_index=True)

with tabs[1]:
    st.subheader("Geographic risk intelligence")
    mapdata = latest.copy()
    mapdata["lat"] = mapdata.area.map(lambda x: ABU_DHABI_AREAS[x][0])
    mapdata["lon"] = mapdata.area.map(lambda x: ABU_DHABI_AREAS[x][1])
    figmap = px.scatter_map(mapdata, lat="lat", lon="lon", color="fused_risk_score",
        size="fused_risk_score", size_max=34, hover_name="area",
        hover_data={"risk_level":True,"top_contributor":True,"fused_risk_score":":.2f",
                    "wet_bulb_approx_c":":.1f","temperature_c":":.1f",
                    "lat":False,"lon":False},
        color_continuous_scale=["#2b8cbe","#fdae61","#d7191c"], range_color=(0,1),
        center={"lat":24.20,"lon":54.45}, zoom=7.4,
        map_style="open-street-map", height=560)
    figmap.update_layout(margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(figmap, width="stretch")
    st.caption("Area coordinates are illustrative reference points. This view is area-level, not worker tracking.")


with tabs[2]:
    st.subheader("Why did the system flag a signal?")
    area_for_explain = st.selectbox("Select area to explain", list(view.area.unique()), key="explain_area")
    area_history = view[view.area == area_for_explain].sort_values("timestamp").copy()
    explain = area_history.iloc[-1]

    k1, k2, k3 = st.columns(3)
    k1.metric("Persistent risk score", f"{explain.persistent_risk_score:.2f}", str(explain.risk_level))
    k2.metric("Latest hourly score", f"{explain.fused_risk_score:.2f}")
    k3.metric("Leading signal", explain.top_contributor)

    st.caption("The risk tier is assigned from the three-hour persistent score. The hourly score shows the latest unsmoothed reading.")
    contrib = pd.DataFrame({
        "Signal":["Environmental conditions","Health-service deviations","Workplace exposure","Multivariate anomaly"],
        "Component score":[explain.environment_score,explain.service_anomaly_score,explain.workplace_exposure_score,explain.anomaly_score]
    }).sort_values("Component score", ascending=True)
    figc=px.bar(contrib,x="Component score",y="Signal",orientation="h",range_x=[0,1],
                color="Component score",
                color_continuous_scale=["#8ecae6","#f4a261","#d62828"],
                text=contrib["Component score"].map(lambda x: f"{x:.2f}"))
    figc.update_traces(textposition="outside", cliponaxis=False)
    figc.update_layout(height=330,showlegend=False,margin=dict(l=5,r=35,t=10,b=5),
                       xaxis_title="Component index (0–1)", yaxis_title="")
    st.plotly_chart(figc,width="stretch")

    st.markdown("**Risk trend · previous 48 hours**")
    trend = area_history.tail(48)
    figtrend = go.Figure()
    figtrend.add_trace(go.Scatter(
        x=trend["timestamp"], y=trend["fused_risk_score"],
        mode="lines", name="Hourly score", line=dict(width=1.5, dash="dot")
    ))
    figtrend.add_trace(go.Scatter(
        x=trend["timestamp"], y=trend["persistent_risk_score"],
        mode="lines", name="Persistent score", line=dict(width=3)
    ))
    figtrend.update_layout(height=300, yaxis=dict(range=[0,1], title="Risk index"),
                           xaxis_title="", margin=dict(l=10,r=10,t=15,b=10),
                           legend=dict(orientation="h", y=1.12))
    st.plotly_chart(figtrend,width="stretch")

    st.info(f"Leading contributor: **{explain.top_contributor}**. Component values describe the model's inputs to the composite index; they are not causal explanations or clinical probabilities.")
    st.markdown("**Latest signal context**")
    context = {
        "Temperature (°C)": round(float(explain.temperature_c),1),
        "Relative humidity (%)": round(float(explain.relative_humidity_pct),1),
        "Approx. wet-bulb (°C)": round(float(explain.wet_bulb_approx_c),1),
        "Workload index": round(float(explain.workload_index),2),
        "Shift exposure index": round(float(explain.shift_exposure_index),2),
        "Synthetic first-aid reports": int(explain.first_aid_reports),
        "Synthetic clinic visits": int(explain.heat_related_clinic_visits),
        "Synthetic absence reports": int(explain.absence_reports),
        "Input completeness": f"{explain.data_completeness:.0%}"
    }
    st.json(context)

with tabs[3]:
    st.subheader("Alert and action center")
    st.caption("Session-persistent demonstration workflow. Alerts and actions are synthetic; no real person is notified and no real safety intervention is recorded.")

    # Session-state store: persists across Streamlit reruns, but not server restarts.
    if "sp_action_records" not in st.session_state:
        st.session_state.sp_action_records = {}
    if "sp_action_audit" not in st.session_state:
        st.session_state.sp_action_audit = []

    # Controlled simulated cases make the end-to-end workflow demonstrable
    # even when live/historical risk signals are below the alert threshold.
    st.markdown("**Demo controls**")
    demo_col1, demo_col2, demo_col3 = st.columns([1, 1, 1.4])
    if demo_col1.button("Generate High-risk demo alert", key="sp_generate_high"):
        area = "DEMO · Logistics Yard"
        now = pd.Timestamp.now(tz="Asia/Dubai")
        stamp = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        st.session_state.sp_action_records[area] = {
            "area": area, "severity": "High", "risk_score": 0.82,
            "trigger_time": stamp, "status": "New", "owner": "Site Safety Officer",
            "action": "Simulated: activate the site heat-safety protocol; verify cooling access and work-rest controls.",
            "resolution_note": "", "created_at": stamp, "updated_at": stamp,
            "is_simulated": True
        }
        st.session_state.sp_action_audit.append({"timestamp":stamp,"area":area,"event":"Simulated High alert generated","from_status":"—","to_status":"New","owner":"Site Safety Officer","note":"Demo-only record; not a real-world alert"})
        st.rerun()
    if demo_col2.button("Generate Elevated demo alert", key="sp_generate_elevated"):
        area = "DEMO · Industrial Shift"
        now = pd.Timestamp.now(tz="Asia/Dubai")
        stamp = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        st.session_state.sp_action_records[area] = {
            "area": area, "severity": "Elevated", "risk_score": 0.57,
            "trigger_time": stamp, "status": "New", "owner": "Site Safety Officer",
            "action": "Simulated: increase monitoring; reinforce hydration/rest guidance and verify preventive controls.",
            "resolution_note": "", "created_at": stamp, "updated_at": stamp,
            "is_simulated": True
        }
        st.session_state.sp_action_audit.append({"timestamp":stamp,"area":area,"event":"Simulated Elevated alert generated","from_status":"—","to_status":"New","owner":"Site Safety Officer","note":"Demo-only record; not a real-world alert"})
        st.rerun()
    if demo_col3.button("Reset simulated demo alerts", key="sp_reset_demo"):
        demo_areas = [a for a, r in st.session_state.sp_action_records.items() if r.get("is_simulated")]
        for area in demo_areas:
            del st.session_state.sp_action_records[area]
        st.session_state.sp_action_audit.append({"timestamp":pd.Timestamp.now(tz="Asia/Dubai").strftime("%Y-%m-%d %H:%M:%S %Z"),"area":"All demo alerts","event":"Simulated alerts reset","from_status":"—","to_status":"—","owner":"—","note":f"Removed {len(demo_areas)} simulated alert(s)"})
        st.rerun()
    st.caption("Generated cases are clearly simulated and intended only to demonstrate assignment, acknowledgement, action, resolution and audit history.")

    alert_view = view.sort_values("timestamp").drop_duplicates("area", keep="last").copy()
    alert_view = alert_view[alert_view["risk_level"].isin(["Elevated", "High"])].copy()
    now_text = pd.Timestamp.now(tz="Asia/Dubai").strftime("%Y-%m-%d %H:%M:%S %Z")

    # Create a stable demo case per area/current alert episode.
    for _, row in alert_view.iterrows():
        area = str(row["area"])
        if area not in st.session_state.sp_action_records:
            st.session_state.sp_action_records[area] = {
                "area": area,
                "severity": str(row["risk_level"]),
                "risk_score": float(row["fused_risk_score"]),
                "trigger_time": str(row["timestamp"]),
                "status": "New",
                "owner": "Site Safety Officer",
                "action": ("Activate site heat-safety protocol; verify conditions, cooling access and work-rest controls."
                           if row["risk_level"] == "High" else
                           "Increase monitoring; reinforce hydration/rest guidance and verify site preventive controls."),
                "resolution_note": "",
                "created_at": now_text,
                "updated_at": now_text
            }
        else:
            # Refresh risk evidence without overwriting action ownership or status.
            record = st.session_state.sp_action_records[area]
            record["severity"] = str(row["risk_level"])
            record["risk_score"] = float(row["fused_risk_score"])
            record["trigger_time"] = str(row["timestamp"])

    records = list(st.session_state.sp_action_records.values())
    active_count = sum(r["status"] != "Resolved" for r in records)
    high_count = sum(r["severity"] == "High" and r["status"] != "Resolved" for r in records)
    pending_count = sum(r["status"] in ["New", "Acknowledged", "In Progress"] for r in records)
    resolved_count = sum(r["status"] == "Resolved" for r in records)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active alerts", active_count)
    k2.metric("High priority", high_count)
    k3.metric("Pending actions", pending_count)
    k4.metric("Resolved actions", resolved_count)

    status_filter = st.selectbox(
        "Filter by action status",
        ["All", "New", "Acknowledged", "In Progress", "Resolved"],
        key="sp_alert_status_filter"
    )
    role_options = ["Site Safety Officer", "Site Manager", "Occupational Health Lead", "HSE Coordinator"]
    visible_records = [r for r in records if status_filter == "All" or r["status"] == status_filter]
    if not visible_records:
        st.info("No alert/action records match this filter. Current risk signals may also be below the alert threshold.")
    for record in sorted(visible_records, key=lambda r: (r["severity"] != "High", r["risk_score"]), reverse=False):
        area = record["area"]
        with st.expander(f"{record['severity']} · {area} · score {record['risk_score']:.2f} · {record['status']}", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Status:** {record['status']}")
            c2.write(f"**Owner:** {record['owner']}")
            c3.write(f"**Updated:** {record['updated_at']}")
            st.write(f"**Evidence:** Latest area signal at {record['trigger_time']}; risk score {record['risk_score']:.2f}.")
            st.write(f"**Recommended action:** {record['action']}")
            owner = st.selectbox("Assign responsible role", role_options,
                                 index=role_options.index(record["owner"]) if record["owner"] in role_options else 0,
                                 key=f"sp_owner_{area}")
            if owner != record["owner"]:
                record["owner"] = owner
                record["updated_at"] = pd.Timestamp.now(tz="Asia/Dubai").strftime("%Y-%m-%d %H:%M:%S %Z")
                st.session_state.sp_action_audit.append({
                    "timestamp": record["updated_at"], "area": area,
                    "event": "Assignment changed", "from_status": record["status"],
                    "to_status": record["status"], "owner": owner, "note": ""
                })
            st.text_area("Action / hand-off note (required to resolve)", key=f"sp_note_{area}",
                         placeholder="Record what was checked or completed...")
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Acknowledge", key=f"sp_ack_{area}", disabled=record["status"] not in ["New"]):
                old = record["status"]; record["status"] = "Acknowledged"
                record["updated_at"] = pd.Timestamp.now(tz="Asia/Dubai").strftime("%Y-%m-%d %H:%M:%S %Z")
                st.session_state.sp_action_audit.append({"timestamp":record["updated_at"],"area":area,"event":"Status changed","from_status":old,"to_status":record["status"],"owner":record["owner"],"note":""})
                st.rerun()
            if b2.button("Start action", key=f"sp_start_{area}", disabled=record["status"] not in ["Acknowledged"]):
                old = record["status"]; record["status"] = "In Progress"
                record["updated_at"] = pd.Timestamp.now(tz="Asia/Dubai").strftime("%Y-%m-%d %H:%M:%S %Z")
                st.session_state.sp_action_audit.append({"timestamp":record["updated_at"],"area":area,"event":"Status changed","from_status":old,"to_status":record["status"],"owner":record["owner"],"note":""})
                st.rerun()
            if b3.button("Resolve", key=f"sp_resolve_{area}", disabled=record["status"] not in ["Acknowledged", "In Progress"]):
                note = st.session_state.get(f"sp_note_{area}", "").strip()
                if not note:
                    st.error("Add a resolution / hand-off note before resolving this action.")
                else:
                    old = record["status"]; record["status"] = "Resolved"
                    record["resolution_note"] = note
                    record["updated_at"] = pd.Timestamp.now(tz="Asia/Dubai").strftime("%Y-%m-%d %H:%M:%S %Z")
                    st.session_state.sp_action_audit.append({"timestamp":record["updated_at"],"area":area,"event":"Status changed","from_status":old,"to_status":"Resolved","owner":record["owner"],"note":note})
                    st.rerun()
            if b4.button("Reopen", key=f"sp_reopen_{area}", disabled=record["status"] != "Resolved"):
                old = record["status"]; record["status"] = "In Progress"
                record["updated_at"] = pd.Timestamp.now(tz="Asia/Dubai").strftime("%Y-%m-%d %H:%M:%S %Z")
                st.session_state.sp_action_audit.append({"timestamp":record["updated_at"],"area":area,"event":"Action reopened","from_status":old,"to_status":record["status"],"owner":record["owner"],"note":"Reopened for further review"})
                st.rerun()
            if record["resolution_note"]:
                st.caption(f"Resolution note: {record['resolution_note']}")

    st.divider()
    st.markdown("**Audit trail**")
    audit = pd.DataFrame(st.session_state.sp_action_audit)
    if audit.empty:
        st.caption("No actions recorded yet. Assignment and status changes will appear here.")
    else:
        st.dataframe(audit.sort_values("timestamp", ascending=False), hide_index=True, width="stretch")
        st.download_button("Download audit trail (CSV)", audit.to_csv(index=False).encode("utf-8"),
                           "sentinelpulse_action_audit.csv", "text/csv")
    if records:
        export = pd.DataFrame(records)
        st.download_button("Download alert/action register (CSV)", export.to_csv(index=False).encode("utf-8"),
                           "sentinelpulse_alert_action_register.csv", "text/csv")
    st.caption("Demo limitation: action records and audit history persist only for the current Streamlit session. They reset when the server session is lost or the app restarts.")


with tabs[4]:
    st.subheader("Temporal holdout validation")
    st.caption("The anomaly detector is trained on the earlier 70% of timestamps. Metrics below are calculated on the later holdout segment against deliberately injected synthetic event labels.")
    holdout = view[view["is_validation_holdout"]].copy()
    if holdout.empty or "injected_event" not in holdout:
        st.info("No labelled holdout events are available in this dataset.")
    else:
        threshold = st.slider("Alert threshold for validation", 0.20, 0.90, 0.46, 0.01)
        pred = (holdout["persistent_risk_score"] >= threshold).astype(int)
        truth = holdout["injected_event"].astype(int)
        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        tn = int(((pred == 0) & (truth == 0)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0
        precision = tp / (tp + fp) if tp + fp else 0
        specificity = tn / (tn + fp) if tn + fp else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Sensitivity", f"{sensitivity:.1%}")
        c2.metric("Precision", f"{precision:.1%}")
        c3.metric("Specificity", f"{specificity:.1%}")
        c4.metric("F1 score", f"{f1:.2f}")
        st.write(f"**Holdout rows:** {len(holdout):,} · **Injected-event rows:** {int(truth.sum()):,} · **TP / FP / FN / TN:** {tp:,} / {fp:,} / {fn:,} / {tn:,}")
        st.write(f"**Model training cutoff:** {pd.to_datetime(data['model_training_cutoff'].iloc[0]).strftime('%Y-%m-%d %H:%M')} (local demo timestamp)")
        cm = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual normal", "Actual injected event"],
            columns=["Predicted no alert", "Predicted alert"]
        )
        st.markdown("**Confusion matrix**")
        st.dataframe(cm, width="stretch")
        st.info(f"Forecast model: **{model_info['model']}** · Training rows: **{model_info['training_rows']:,}**. Forecast output is a demonstration and requires rolling-origin evaluation.")
        st.warning("These results measure detection of synthetic injected events only. They are not clinical validation, real-world predictive performance, or evidence of reduced illness. Threshold selection should be pre-specified and evaluated on independent, locally representative data.")
        st.download_button("Download current scored dataset (CSV)",view.to_csv(index=False).encode("utf-8"),"sentinelpulse_demo_scored_data.csv","text/csv")


with tabs[5]:
    st.subheader("Predictive heat-risk intelligence")
    st.caption("Live public weather forecast combined with hour-of-day workplace exposure profiles. Operational screening index only; not a clinical prediction.")
    if not live_forecast_enabled:
        st.info("Enable 'Live weather forecast' in Demo controls to display current forecast data.")
    elif live_outlook.empty:
        st.warning("Live weather forecast could not be retrieved. Check your internet connection and use Refresh weather data.")
        for err in live_weather_errors:
            st.caption(err)
    else:
        live_view = live_outlook.copy()
        if selected_area != "All areas":
            live_view = live_view[live_view["area"] == selected_area]
        if live_view.empty:
            st.info("No live forecast rows are available for this area.")
        else:
            live_now = live_view["forecast_timestamp"].min()
            live_6h = live_view[live_view["forecast_timestamp"] <= live_now + pd.Timedelta(hours=6)]
            live_24h = live_view[live_view["forecast_timestamp"] <= live_now + pd.Timedelta(hours=24)]
            a,b,c = st.columns(3)
            a.metric("Areas with live forecast", live_view["area"].nunique())
            b.metric("Peak index · next 6h", f"{live_6h['projected_heat_risk_index'].max():.2f}" if not live_6h.empty else "—")
            c.metric("Peak index · next 24h", f"{live_24h['projected_heat_risk_index'].max():.2f}" if not live_24h.empty else "—")
            live_fig = px.line(live_view, x="forecast_timestamp", y="projected_heat_risk_index",
                               color="area", markers=True,
                               labels={"forecast_timestamp":"Dubai local time",
                                       "projected_heat_risk_index":"Operational heat-risk index",
                                       "area":"Area"})
            for y, label in [(0.30,"Guarded"),(0.46,"Elevated"),(0.62,"High")]:
                live_fig.add_hline(y=y, line_dash="dot", annotation_text=label)
            live_fig.update_layout(height=380, yaxis=dict(range=[0,1]), margin=dict(l=10,r=10,t=15,b=10))
            st.plotly_chart(live_fig, width="stretch")
            display = live_view[["area","forecast_timestamp","temperature_c","relative_humidity_pct",
                                 "wet_bulb_forecast_c","wind_speed_kmh","workload_index",
                                 "shift_exposure_index","projected_heat_risk_index",
                                 "projected_risk_tier"]].copy()
            display = display.rename(columns={
                "area":"Area","forecast_timestamp":"Forecast time (Asia/Dubai)",
                "temperature_c":"Temperature °C","relative_humidity_pct":"Humidity %",
                "wet_bulb_forecast_c":"Forecast wet-bulb °C","wind_speed_kmh":"Wind km/h",
                "projected_heat_risk_index":"Risk index","projected_risk_tier":"Risk tier"
            })
            st.dataframe(display, hide_index=True, width="stretch")
            st.caption("Weather source: Open-Meteo Forecast API. Workload and shift exposure use historical medians for the matching area and forecast hour; these operational profiles are synthetic.")
            if live_weather_errors:
                st.warning("Some areas could not be fetched: " + "; ".join(live_weather_errors))
            st.download_button("Download live weather outlook (CSV)", live_view.to_csv(index=False).encode("utf-8"),
                               "sentinelpulse_live_weather_outlook.csv", "text/csv")
    st.divider()
    st.subheader("Historical-baseline scenario outlook")
    st.caption("A separate comparison based on recent same-hour historical patterns; not a live forecast.")
    area_options = ["All areas"] + sorted(heat_outlook["area"].unique().tolist())
    outlook_area = st.selectbox("Area for outlook", area_options, index=0, key="outlook_area")
    selected_outlook = heat_outlook.copy()
    if outlook_area != "All areas":
        selected_outlook = selected_outlook[selected_outlook["area"] == outlook_area].copy()

    if selected_outlook.empty:
        st.warning("Insufficient historical data to construct an outlook.")
    else:
        future24 = selected_outlook[selected_outlook["horizon_hours"] == 24]
        sixhour = selected_outlook[selected_outlook["horizon_hours"] == 6]
        c1, c2, c3 = st.columns(3)
        c1.metric("Areas in outlook", selected_outlook["area"].nunique())
        c2.metric("Elevated/high at +6h", int(sixhour["projected_risk_tier"].isin(["Elevated","High"]).sum()))
        c3.metric("Elevated/high at +24h", int(future24["projected_risk_tier"].isin(["Elevated","High"]).sum()))

        chart = selected_outlook.copy()
        figout = px.line(chart, x="forecast_timestamp", y="projected_heat_risk_index",
                         color="area", markers=True,
                         labels={"forecast_timestamp":"Forecast hour",
                                 "projected_heat_risk_index":"Projected heat-exposure index",
                                 "area":"Area"})
        figout.add_hline(y=0.30, line_dash="dot", annotation_text="Guarded threshold")
        figout.add_hline(y=0.46, line_dash="dot", annotation_text="Elevated threshold")
        figout.add_hline(y=0.62, line_dash="dot", annotation_text="High threshold")
        figout.update_layout(height=390, yaxis=dict(range=[0,1]), margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(figout, width="stretch")

        st.markdown("**6-hour and 24-hour outlook by area**")
        summary = key_horizons.copy()
        if outlook_area != "All areas":
            summary = summary[summary["area"] == outlook_area]
        summary = summary[["area","horizon_hours","projected_risk_tier",
                           "projected_heat_risk_index","projected_temperature_c",
                           "projected_wet_bulb_approx_c","outlook_confidence",
                           "baseline_samples_same_hour"]]
        summary = summary.rename(columns={
            "area":"Area","horizon_hours":"Horizon (hours)",
            "projected_risk_tier":"Projected tier","projected_heat_risk_index":"Heat index",
            "projected_temperature_c":"Temp °C","projected_wet_bulb_approx_c":"Approx. wet-bulb °C",
            "outlook_confidence":"Baseline confidence","baseline_samples_same_hour":"Same-hour samples"
        })
        st.dataframe(summary, hide_index=True, width="stretch")
        st.info("Confidence indicates historical sample sufficiency and stability only. It is not a calibrated probability or prediction interval.")
        st.markdown("**Suggested operational use**")
        st.write("- Review areas with a projected Elevated or High tier before the forecast hour.")
        st.write("- Verify local weather and site conditions before changing work-rest schedules.")
        st.write("- Consider hydration, shaded rest, cooling access, acclimatization checks and supervisor communication.")
        st.warning("The outlook repeats recent same-hour historical medians. It does not ingest future weather forecasts, worker physiology, or real incident data. Treat it as a scenario-planning aid until back-tested and locally validated.")


with tabs[6]:
    st.subheader("Explainable predictive intelligence")
    st.caption("Area-level forecast trajectory, peak timing, dominant signal and operational actions. Uses live public weather plus synthetic hour-matched workplace profiles.")
    if not live_forecast_enabled or live_outlook.empty:
        st.info("Enable live weather forecast in Demo controls and refresh data to run the predictive intelligence layer.")
    else:
        predictive = build_predictive_intelligence(live_outlook, data)
        if selected_area != "All areas":
            predictive = predictive[predictive["area"] == selected_area].copy()
        if predictive.empty:
            st.warning("No predictive summaries are available for the selected area.")
        else:
            p1, p2, p3 = st.columns(3)
            p1.metric("Areas analyzed", predictive["area"].nunique())
            p2.metric("Rising trajectories", int((predictive["trajectory_direction"] == "Rising").sum()))
            p3.metric("Elevated/high forecast peaks", int(predictive["forecast_peak_tier"].isin(["Elevated", "High"]).sum()))

            st.markdown("### Synthetic workplace scenario assignments")
            st.caption("This table makes the fictional workplace archetype assigned to each monitoring area explicit. These are controlled demo assumptions—not claims about actual employers, industries, shifts or workers in those locations.")
            assignment_rows = []
            for _, row in predictive.iterrows():
                profile = DEMO_OPERATIONAL_PROFILES.get(row["area"], {})
                assignment_rows.append({
                    "Monitoring area": row["area"],
                    "Fictional scenario": row["demo_operational_profile"],
                    "Illustrative shift (Dubai time)": (
                        f"{int(profile['start']):02d}:00–{int(profile['end']) + 1:02d}:00"
                        if profile else "Not specified"
                    ),
                    "Workload baseline": profile.get("base", np.nan),
                    "Workload variation": profile.get("amplitude", np.nan),
                    "Shift-exposure multiplier": profile.get("exposure", np.nan),
                    "Mean forecast workload": row["mean_projected_workload"],
                    "Mean forecast shift exposure": row["mean_projected_shift_exposure"],
                })
            assignment_view = pd.DataFrame(assignment_rows)
            st.dataframe(assignment_view.round(3), hide_index=True, width="stretch")
            st.caption("Indices and profile parameters are synthetic, dimensionless demo values on a 0–1 scale. The shift window is an illustrative configuration only. Replace these assumptions with authorized, locally validated site schedules before operational use.")

            st.markdown("**Forecast peak and trajectory**")
            st.caption("Area-level weather is fetched from the public forecast API. Workplace scenarios are fictional demo archetypes used to test how different shift patterns can change the operational index.")
            predictive_chart = live_outlook.copy()
            if selected_area != "All areas":
                predictive_chart = predictive_chart[predictive_chart["area"] == selected_area]
            fig_pred = px.line(
                predictive_chart, x="forecast_timestamp", y="projected_heat_risk_index",
                color="area", markers=True,
                labels={"forecast_timestamp":"Forecast time (Asia/Dubai)",
                        "projected_heat_risk_index":"Operational risk index","area":"Area"}
            )
            for y, label in [(0.30, "Guarded"), (0.46, "Elevated"), (0.62, "High")]:
                fig_pred.add_hline(y=y, line_dash="dot", annotation_text=label)
            fig_pred.update_layout(height=390, yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=15, b=10))
            st.plotly_chart(fig_pred, width="stretch")

            st.markdown("**Geographic comparison — peak index by area**")
            comparison = predictive.sort_values("forecast_peak_index", ascending=False).copy()
            fig_compare = px.bar(
                comparison, x="area", y="forecast_peak_index", color="forecast_peak_index",
                hover_data={"demo_operational_profile": True, "leading_driver": True,
                            "environment_contribution_pct": ":.1f",
                            "workplace_contribution_pct": ":.1f"},
                labels={"area":"Monitoring area", "forecast_peak_index":"Peak operational index",
                        "demo_operational_profile":"Synthetic demo scenario",
                        "leading_driver":"Leading contributor"},
                color_continuous_scale="YlOrRd"
            )
            fig_compare.update_layout(height=330, yaxis=dict(range=[0, 1]),
                                      coloraxis_showscale=False, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig_compare, width="stretch")

            summary_cols = [
                "area", "demo_operational_profile", "forecast_peak_index", "forecast_peak_tier", "forecast_peak_time",
                "trajectory_pattern", "trajectory_direction", "forecast_change",
                "guarded_hours", "elevated_hours", "high_hours", "leading_driver",
                "environment_contribution_pct", "workplace_contribution_pct",
                "mean_projected_workload", "mean_projected_shift_exposure", "context_coverage"
            ]
            summary = predictive[summary_cols].rename(columns={
                "area":"Area", "demo_operational_profile":"Synthetic demo scenario",
                "forecast_peak_index":"Peak index",
                "forecast_peak_tier":"Peak tier", "forecast_peak_time":"Peak time (Asia/Dubai)",
                "trajectory_pattern":"Trajectory pattern", "trajectory_direction":"Net direction",
                "forecast_change":"Outlook change", "guarded_hours":"Guarded hours",
                "elevated_hours":"Elevated hours", "high_hours":"High hours",
                "leading_driver":"Leading driver",
                "environment_contribution_pct":"Environment contribution %",
                "workplace_contribution_pct":"Workplace contribution %",
                "mean_projected_workload":"Mean workload index",
                "mean_projected_shift_exposure":"Mean shift-exposure index",
                "context_coverage":"Operational context coverage"
            })
            numeric_cols = summary.select_dtypes(include=["number"]).columns
            summary.loc[:, numeric_cols] = summary.loc[:, numeric_cols].round(3)
            st.dataframe(summary, hide_index=True, width="stretch")

            st.markdown("**Area-level explanation and recommended response**")
            area_pick = st.selectbox("Select area for explanation", predictive["area"].tolist(), key="predictive_area")
            chosen = predictive[predictive["area"] == area_pick].iloc[0]
            d1, d2 = st.columns(2)
            d1.metric("Forecast peak", f"{chosen['forecast_peak_index']:.2f}", str(chosen["forecast_peak_tier"]))
            d2.metric("Trajectory pattern", str(chosen["trajectory_pattern"]), f"{chosen['trajectory_direction']} · {chosen['forecast_change']:+.2f} net change")
            st.write(f"**Peak expected:** {pd.to_datetime(chosen['forecast_peak_time']).strftime('%d %b %Y, %H:%M')} Dubai time")
            st.write(f"**Illustrative scenario:** {chosen['demo_operational_profile']}")
            st.write(f"**Leading contributor:** {chosen['leading_driver']}")
            st.write(f"**Driver mix:** Environment {chosen['environment_contribution_pct']:.0f}% · Workplace profile {chosen['workplace_contribution_pct']:.0f}%")
            st.write(f"**Forecast duration:** {int(chosen['guarded_hours'])} guarded+ hours · {int(chosen['elevated_hours'])} elevated+ hours · {int(chosen['high_hours'])} high hours")
            st.write(f"**Context coverage:** {chosen['context_coverage']} ({int(chosen['minimum_hourly_profile_samples'])} minimum matching-hour samples)")
            st.info(f"**Suggested operational response:** {chosen['suggested_operational_action']}")
            st.caption("Driver attribution is a weighted component comparison, not causal inference. Context coverage describes the amount of matching-hour synthetic history, not model confidence.")
            st.warning("This is an explainable decision-support layer, not a clinically validated prediction. Workplace profiles remain synthetic; verify site conditions and follow approved occupational heat-safety procedures. Do not use individual-level or employment decisions.")
            st.download_button(
                "Download predictive intelligence summary (CSV)",
                predictive.to_csv(index=False).encode("utf-8"),
                "sentinelpulse_predictive_intelligence.csv", "text/csv"
            )

st.divider()
st.markdown('<p class="small-note">SentinelPulse is a research and operational decision-support prototype. It does not diagnose, predict an individual’s illness, replace clinical judgement, or automate employment decisions. Environmental calculations and synthetic signals require local validation before operational use.</p>',unsafe_allow_html=True)
