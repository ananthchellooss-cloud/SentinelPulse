import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentinelpulse.data import make_synthetic_data
from sentinelpulse.models import score_signals, add_forecast, build_heat_risk_outlook, score_live_weather_outlook, build_predictive_intelligence

def test_pipeline():
    df = make_synthetic_data(days=35)
    scored = score_signals(df)
    assert len(scored) == len(df)
    assert scored["fused_risk_score"].between(0, 1).all()
    assert scored["persistent_risk_score"].between(0, 1).all()
    assert scored["anomaly_score"].between(0, 1).all()
    assert scored["is_validation_holdout"].any()
    assert set(scored["risk_level"].unique()).issubset({"Low", "Guarded", "Elevated", "High"})
    assert "first_aid_reports_trailing_z" in scored.columns
    forecasted, info = add_forecast(scored)
    assert "forecast_service_next_hour" in forecasted.columns
    assert info["training_rows"] > 0
    outlook, key_horizons = build_heat_risk_outlook(scored, horizons=(6, 24))
    assert len(outlook) == scored["area"].nunique() * 24
    assert set(key_horizons["horizon_hours"].unique()) == {6, 24}
    assert outlook["projected_heat_risk_index"].between(0, 1).all()
    assert outlook["outlook_confidence"].between(0, 1).all()
    assert outlook["projected_risk_tier"].isin(["Low", "Guarded", "Elevated", "High"]).all()
    sample_weather = pd.DataFrame({
        "forecast_timestamp": pd.date_range("2026-11-12", periods=24, freq="h"),
        "area": ["Abu Dhabi Island"] * 24,
        "temperature_c": [36.0] * 24,
        "relative_humidity_pct": [55.0] * 24,
        "wind_speed_kmh": [12.0] * 24,
        "wet_bulb_forecast_c": [28.0] * 24,
        "weather_source": ["test fixture"] * 24
    })
    live = score_live_weather_outlook(sample_weather, scored)
    assert len(live) == 24
    assert live["projected_heat_risk_index"].between(0, 1).all()
    assert live["projected_risk_tier"].isin(["Low", "Guarded", "Elevated", "High"]).all()
    # Hour-of-day profiles should distinguish the simulated daytime shift from night.
    daytime = live.loc[live["forecast_timestamp"].dt.hour.between(8, 16), "shift_exposure_index"].mean()
    nighttime = live.loc[live["forecast_timestamp"].dt.hour.isin([0, 1, 2, 22, 23]), "shift_exposure_index"].mean()
    assert daytime > nighttime
    assert live["operational_profile_samples"].gt(0).all()
    assert live["demo_operational_profile"].nunique() == 1
    predictive = build_predictive_intelligence(live, scored)
    assert len(predictive) == 1
    assert predictive["forecast_peak_index"].between(0, 1).all()
    assert predictive["forecast_peak_tier"].isin(["Low", "Guarded", "Elevated", "High"]).all()
    assert predictive["trajectory_direction"].isin(["Rising", "Falling", "Stable"]).all()
    assert predictive["trajectory_pattern"].isin([
        "Morning peak", "Afternoon peak", "Sustained elevated exposure",
        "Rising", "Falling", "Stable"
    ]).all()
    assert predictive[["guarded_hours", "elevated_hours", "high_hours"]].ge(0).all().all()
    assert predictive["environment_contribution_pct"].between(0, 100).all()
    assert predictive["workplace_contribution_pct"].between(0, 100).all()
    assert predictive["suggested_operational_action"].notna().all()

    # Geographic differentiation test: same weather must still show distinct
    # operational scenario profiles, and the profile labels must be explicit.
    all_weather = pd.concat([
        pd.DataFrame({
            "forecast_timestamp": pd.date_range("2026-11-12", periods=24, freq="h"),
            "area": [area] * 24,
            "temperature_c": [36.0] * 24,
            "relative_humidity_pct": [55.0] * 24,
            "wind_speed_kmh": [12.0] * 24,
            "wet_bulb_forecast_c": [28.0] * 24,
            "weather_source": ["test fixture"] * 24
        }) for area in scored["area"].drop_duplicates()
    ], ignore_index=True)
    all_live = score_live_weather_outlook(all_weather, scored)
    assert all_live.groupby("area")["demo_operational_profile"].first().nunique() == scored["area"].nunique()
    assert all_live.groupby("area")["projected_heat_risk_index"].mean().nunique() > 1
    all_predictive = build_predictive_intelligence(all_live, scored)
    assert len(all_predictive) == scored["area"].nunique()
    assert all_predictive["demo_operational_profile"].nunique() == scored["area"].nunique()

    # Deterministic shape test: a morning hump must not be labelled merely Stable.
    shape_fixture = live.copy()
    shape_fixture["forecast_timestamp"] = pd.date_range("2026-11-12 00:00", periods=24, freq="h")
    shape_fixture["projected_heat_risk_index"] = [
        0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.34, 0.33,
        0.31, 0.29, 0.27, 0.25, 0.23, 0.22, 0.21, 0.20,
        0.20, 0.19, 0.19, 0.18, 0.18, 0.19, 0.20, 0.20
    ]
    shape_result = build_predictive_intelligence(shape_fixture, scored)
    assert shape_result.iloc[0]["trajectory_pattern"] == "Morning peak"

if __name__ == "__main__":
    test_pipeline()
    print("Temporal ML pipeline tests passed.")
