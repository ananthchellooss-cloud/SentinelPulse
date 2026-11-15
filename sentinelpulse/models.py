from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "temperature_c", "relative_humidity_pct", "wind_speed_kmh",
    "wet_bulb_approx_c", "workload_index", "shift_exposure_index",
    "first_aid_reports", "heat_related_clinic_visits", "absence_reports"
]

SERVICE_COLUMNS = [
    "first_aid_reports", "heat_related_clinic_visits", "absence_reports"
]

def _safe_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clean = frame[columns].replace([np.inf, -np.inf], np.nan)
    return clean.fillna(clean.median()).fillna(0.0)

def _trailing_robust_deviation(group: pd.DataFrame, column: str,
                               window: int = 168, min_periods: int = 24) -> pd.Series:
    """
    Calculate a one-sided trailing robust z-score.
    Baseline uses only observations strictly before the current timestamp.
    """
    values = pd.to_numeric(group[column], errors="coerce").astype(float)
    prior = values.shift(1)
    rolling = prior.rolling(window=window, min_periods=min_periods)
    median = rolling.median()
    mad = (prior - median).abs().rolling(window=window, min_periods=min_periods).median()
    robust_scale = 1.4826 * mad

    # Use a trailing standard deviation only when MAD is zero/undefined.
    std = prior.rolling(window=window, min_periods=min_periods).std()
    scale = robust_scale.where(robust_scale > 0.05, std)
    z = (values - median) / scale.clip(lower=0.5)
    # At the start of a series, do not fabricate an anomaly from an unavailable baseline.
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8, 8)

def score_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-signal early-warning scoring:
    - one-sided rolling baselines for service deviations
    - Isolation Forest trained on an earlier temporal segment
    - transparent environmental/workplace components
    - short persistence smoothing to reduce one-hour alert spikes

    Scores are prioritization indices, not disease probabilities or clinical thresholds.
    """
    required = ["timestamp", "area", *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["area", "timestamp"]).reset_index(drop=True)
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek

    # Temporal split: the unsupervised detector learns its feature space on the
    # first 70% of unique timestamps, then scores the full series.
    unique_times = np.sort(out["timestamp"].dropna().unique())
    split_idx = max(1, int(len(unique_times) * 0.70))
    cutoff = unique_times[min(split_idx, len(unique_times)-1)]
    train_mask = out["timestamp"] < cutoff
    if train_mask.sum() < 50:
        train_mask = pd.Series(True, index=out.index)

    X = _safe_numeric_frame(out, FEATURES)
    scaler = StandardScaler()
    scaler.fit(X.loc[train_mask])
    X_scaled = scaler.transform(X)

    detector = IsolationForest(
        n_estimators=100,
        contamination="auto",
        max_samples="auto",
        random_state=42,
        # Streamlit Community Cloud runs in a constrained container; avoid
        # spawning every available CPU worker, which can stall startup.
        n_jobs=1
    )
    detector.fit(X_scaled[train_mask])
    raw_anomaly = -detector.score_samples(X_scaled)

    train_raw = raw_anomaly[np.asarray(train_mask)]
    low, high = np.quantile(train_raw, [0.05, 0.99])
    out["anomaly_score"] = np.clip(
        (raw_anomaly - low) / max(high - low, 1e-8), 0, 1
    )
    out["model_training_cutoff"] = pd.Timestamp(cutoff)

    # Environmental context index. Approximate wet-bulb is not WBGT and is
    # not used as a stand-alone occupational safety threshold.
    wet_bulb = pd.to_numeric(out["wet_bulb_approx_c"], errors="coerce").fillna(0)
    temperature = pd.to_numeric(out["temperature_c"], errors="coerce").fillna(0)
    humidity = pd.to_numeric(out["relative_humidity_pct"], errors="coerce").fillna(0)
    out["environment_score"] = (
        0.55 * np.clip((wet_bulb - 24.0) / 9.0, 0, 1)
        + 0.30 * np.clip((temperature - 32.0) / 12.0, 0, 1)
        + 0.15 * np.clip(humidity / 100.0, 0, 1)
    ).clip(0, 1)

    # One-sided area-specific service baselines: no future observations enter the baseline.
    service_z_cols = []
    for column in SERVICE_COLUMNS:
        z_name = f"{column}_trailing_z"
        out[z_name] = 0.0
        for _, idx in out.groupby("area", sort=False).groups.items():
            ordered = out.loc[idx].sort_values("timestamp")
            z = _trailing_robust_deviation(ordered, column)
            out.loc[ordered.index, z_name] = z.to_numpy()
        service_z_cols.append(z_name)

    z_frame = out[service_z_cols].clip(lower=0).div(4.0).clip(upper=1)
    out["service_anomaly_score"] = z_frame.mean(axis=1).clip(0, 1)

    workload = pd.to_numeric(out["workload_index"], errors="coerce").fillna(0).clip(0, 1)
    exposure = pd.to_numeric(out["shift_exposure_index"], errors="coerce").fillna(0).clip(0, 1)
    out["workplace_exposure_score"] = (0.55 * workload + 0.45 * exposure).clip(0, 1)

    # Weighted signal fusion; not a calibrated probability.
    out["fused_risk_score"] = (
        0.30 * out["environment_score"]
        + 0.30 * out["service_anomaly_score"]
        + 0.20 * out["workplace_exposure_score"]
        + 0.20 * out["anomaly_score"]
    ).clip(0, 1)

    # Three-hour smoothing makes persistent signals more prominent than isolated spikes.
    out["persistent_risk_score"] = (
        out.groupby("area", sort=False)["fused_risk_score"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    out["risk_level"] = pd.cut(
        out["persistent_risk_score"],
        bins=[-0.001, 0.30, 0.46, 0.62, 1.001],
        labels=["Low", "Guarded", "Elevated", "High"]
    ).astype(str)

    component_cols = [
        "environment_score", "service_anomaly_score",
        "workplace_exposure_score", "anomaly_score"
    ]
    contributor_labels = {
        "environment_score": "Environmental conditions",
        "service_anomaly_score": "Health-service deviations",
        "workplace_exposure_score": "Workplace exposure",
        "anomaly_score": "Combined multivariate anomaly"
    }
    out["top_contributor"] = (
        out[component_cols].idxmax(axis=1).map(contributor_labels)
    )
    out["data_completeness"] = (
        1.0 - out[FEATURES].isna().mean(axis=1)
    ).clip(0, 1)
    out["is_validation_holdout"] = out["timestamp"] >= pd.Timestamp(cutoff)
    return out

def add_forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    One-step-ahead service-volume forecast demonstration.
    Features use lagged service values and current environmental context.
    For rigorous forecasting evaluation, use a separate rolling-origin backtest.
    """
    out = df.copy()
    out["service_total"] = out["first_aid_reports"] + out["heat_related_clinic_visits"]
    out["hour"] = pd.to_datetime(out["timestamp"]).dt.hour
    out["dayofweek"] = pd.to_datetime(out["timestamp"]).dt.dayofweek
    out["lag1_service"] = out.groupby("area")["service_total"].shift(1)
    out["lag24_service"] = out.groupby("area")["service_total"].shift(24)
    out["lag1_wetbulb"] = out.groupby("area")["wet_bulb_approx_c"].shift(1)

    features = [
        "hour", "dayofweek", "temperature_c", "wet_bulb_approx_c",
        "workload_index", "shift_exposure_index",
        "lag1_service", "lag24_service", "lag1_wetbulb"
    ]
    train = out.dropna(subset=["lag1_service", "lag24_service", "lag1_wetbulb"]).copy()
    train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["service_total"])
    out["forecast_service_next_hour"] = np.nan

    model = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42
    )
    if len(train) >= 100:
        # Fit only on earlier timestamps; do not train on the temporal holdout.
        cutoff = out.loc[out.get("is_validation_holdout", pd.Series(False, index=out.index)),
                         "timestamp"].min()
        if pd.notna(cutoff):
            fit_rows = train["timestamp"] < cutoff
        else:
            fit_rows = pd.Series(True, index=train.index)
        fit_data = train.loc[fit_rows]
        if len(fit_data) < 100:
            fit_data = train
        model.fit(fit_data[features], fit_data["service_total"])
        valid_rows = train[features].notna().all(axis=1)
        predictions = np.maximum(0, model.predict(train.loc[valid_rows, features]))
        out.loc[train.loc[valid_rows].index, "forecast_service_next_hour"] = predictions
        training_rows = len(fit_data)
    else:
        training_rows = 0

    return out, {
        "model": "HistGradientBoostingRegressor",
        "training_rows": int(training_rows),
        "features": features,
        "forecast_target": "next-hour aggregate first-aid + clinic signal"
    }


def build_heat_risk_outlook(df: pd.DataFrame, horizons: tuple[int, ...] = (6, 24),
                            lookback_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a transparent 1–24 hour heat-exposure outlook from recent same-hour
    historical patterns. This is a seasonal-baseline projection, not a
    meteorological forecast or clinical risk probability.

    Forecasts temperature, humidity, workload and shift exposure using the
    median of the same hour-of-day over the lookback window. Risk is an
    operational index based only on projected environment and workplace context.
    """
    required = ["timestamp", "area", "temperature_c", "relative_humidity_pct",
                "wet_bulb_approx_c", "workload_index", "shift_exposure_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for heat outlook: {missing}")
    if not horizons or any((not isinstance(h, int) or h < 1 or h > 24) for h in horizons):
        raise ValueError("horizons must contain integer hours from 1 to 24")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    rows = []
    for area, group in data.groupby("area", sort=False):
        group = group.sort_values("timestamp")
        latest = group.iloc[-1]
        cutoff = latest["timestamp"] - pd.Timedelta(days=lookback_days)
        history = group[group["timestamp"] >= cutoff].copy()
        if history.empty:
            continue
        history["hour"] = history["timestamp"].dt.hour

        for horizon in range(1, max(horizons) + 1):
            target_time = latest["timestamp"] + pd.Timedelta(hours=horizon)
            same_hour = history[history["hour"] == target_time.hour]
            # Sparse history falls back to all recent observations rather than
            # inventing a forecast; confidence is downgraded accordingly.
            source = same_hour if len(same_hour) >= 3 else history
            values = {}
            for col in ["temperature_c", "relative_humidity_pct",
                        "wet_bulb_approx_c", "workload_index", "shift_exposure_index"]:
                values[col] = float(pd.to_numeric(source[col], errors="coerce").median())

            wet_bulb = values["wet_bulb_approx_c"]
            temp = values["temperature_c"]
            humidity = values["relative_humidity_pct"]
            env_score = (
                0.55 * np.clip((wet_bulb - 24.0) / 9.0, 0, 1)
                + 0.30 * np.clip((temp - 32.0) / 12.0, 0, 1)
                + 0.15 * np.clip(humidity / 100.0, 0, 1)
            )
            workplace_score = (
                0.55 * np.clip(values["workload_index"], 0, 1)
                + 0.45 * np.clip(values["shift_exposure_index"], 0, 1)
            )
            projected_score = float(np.clip(0.65 * env_score + 0.35 * workplace_score, 0, 1))
            # Confidence is a data-sufficiency/stability indicator, not a
            # statistical prediction interval.
            sample_factor = min(len(same_hour) / 7.0, 1.0)
            variability = float(pd.to_numeric(source["wet_bulb_approx_c"], errors="coerce").std() or 0)
            stability_factor = float(np.clip(1.0 - variability / 8.0, 0.25, 1.0))
            confidence = float(np.clip(sample_factor * stability_factor, 0, 1))
            tier = ("Low" if projected_score < 0.30 else
                    "Guarded" if projected_score < 0.46 else
                    "Elevated" if projected_score < 0.62 else "High")
            rows.append({
                "area": area, "forecast_timestamp": target_time,
                "horizon_hours": horizon,
                "projected_temperature_c": values["temperature_c"],
                "projected_relative_humidity_pct": values["relative_humidity_pct"],
                "projected_wet_bulb_approx_c": wet_bulb,
                "projected_workload_index": values["workload_index"],
                "projected_shift_exposure_index": values["shift_exposure_index"],
                "projected_environment_score": float(env_score),
                "projected_workplace_score": float(workplace_score),
                "projected_heat_risk_index": projected_score,
                "projected_risk_tier": tier,
                "baseline_samples_same_hour": int(len(same_hour)),
                "outlook_confidence": confidence
            })

    outlook = pd.DataFrame(rows)
    selected = outlook[outlook["horizon_hours"].isin(horizons)].copy() if not outlook.empty else outlook.copy()
    return outlook, selected


def score_live_weather_outlook(weather: pd.DataFrame, scored_history: pd.DataFrame) -> pd.DataFrame:
    """
    Score forecast weather with hour-of-day operational profiles.

    Workload and shift exposure are estimated from the historical median for
    the same area and local hour. This avoids carrying a night-shift value into
    the following daytime forecast. These operational profiles remain synthetic
    in the competition demo and must be replaced with authorized real schedules
    or operational data before deployment.
    """
    required_weather = ["forecast_timestamp", "area", "temperature_c",
                        "relative_humidity_pct", "wind_speed_kmh", "wet_bulb_forecast_c"]
    missing = [c for c in required_weather if c not in weather.columns]
    if missing:
        raise ValueError(f"Missing weather forecast fields: {missing}")
    required_history = ["timestamp", "area", "workload_index", "shift_exposure_index"]
    missing_history = [c for c in required_history if c not in scored_history.columns]
    if missing_history:
        raise ValueError(f"Missing operational context fields: {missing_history}")

    history = scored_history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history["hour_of_day"] = history["timestamp"].dt.hour
    # Per-area, per-hour medians create a representative operational profile.
    # A per-area overall median is used only if an hourly profile is absent.
    hourly_profile = (
        history.groupby(["area", "hour_of_day"], as_index=False)
        .agg(workload_index=("workload_index", "median"),
             shift_exposure_index=("shift_exposure_index", "median"),
             operational_profile_samples=("timestamp", "count"))
    )
    area_fallback = (
        history.groupby("area", as_index=False)
        .agg(workload_fallback=("workload_index", "median"),
             shift_fallback=("shift_exposure_index", "median"))
    )
    if "demo_operational_profile" in history.columns:
        profile_names = (history.groupby("area", as_index=False)
                         .agg(demo_operational_profile=("demo_operational_profile", "first")))
        area_fallback = area_fallback.merge(profile_names, on="area", how="left")
    else:
        area_fallback["demo_operational_profile"] = "Unspecified synthetic operational profile"

    out = weather.copy()
    out["forecast_timestamp"] = pd.to_datetime(out["forecast_timestamp"])
    out["hour_of_day"] = out["forecast_timestamp"].dt.hour
    out = out.merge(hourly_profile, on=["area", "hour_of_day"], how="left")
    out = out.merge(area_fallback, on="area", how="left")
    out["workload_index"] = pd.to_numeric(out["workload_index"], errors="coerce").fillna(out["workload_fallback"]).fillna(0.5)
    out["shift_exposure_index"] = pd.to_numeric(out["shift_exposure_index"], errors="coerce").fillna(out["shift_fallback"]).fillna(0.5)
    out["operational_profile_samples"] = out["operational_profile_samples"].fillna(0).astype(int)
    if "demo_operational_profile" not in out.columns:
        out["demo_operational_profile"] = out["demo_operational_profile_y"].fillna("Unspecified synthetic operational profile")
    elif "demo_operational_profile_y" in out.columns:
        out["demo_operational_profile"] = out["demo_operational_profile"].fillna(out["demo_operational_profile_y"])
    out = out.drop(columns=[c for c in ["workload_fallback", "shift_fallback", "demo_operational_profile_y"] if c in out.columns])

    wet = pd.to_numeric(out["wet_bulb_forecast_c"], errors="coerce")
    temp = pd.to_numeric(out["temperature_c"], errors="coerce")
    humidity = pd.to_numeric(out["relative_humidity_pct"], errors="coerce")
    out["forecast_environment_score"] = (
        0.55 * np.clip((wet - 24.0) / 9.0, 0, 1)
        + 0.30 * np.clip((temp - 32.0) / 12.0, 0, 1)
        + 0.15 * np.clip(humidity / 100.0, 0, 1)
    ).clip(0, 1)
    out["forecast_workplace_score"] = (
        0.55 * out["workload_index"].clip(0, 1)
        + 0.45 * out["shift_exposure_index"].clip(0, 1)
    )
    out["projected_heat_risk_index"] = (
        0.65 * out["forecast_environment_score"]
        + 0.35 * out["forecast_workplace_score"]
    ).clip(0, 1)
    out["projected_risk_tier"] = pd.cut(
        out["projected_heat_risk_index"],
        bins=[-0.001, 0.30, 0.46, 0.62, 1.001],
        labels=["Low", "Guarded", "Elevated", "High"]
    ).astype(str)
    out["outlook_type"] = "Live meteorological forecast + hour-of-day synthetic operational profile"
    out["operational_context_basis"] = "Historical median for matching area and local hour; synthetic demonstration data"
    return out


def build_predictive_intelligence(live_outlook: pd.DataFrame,
                                  scored_history: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize area-level forecast shape, peak timing, drivers and actions.

    Weather is sourced from the public forecast feed; operational exposure
    profiles are synthetic/hour-matched. This is not an individual or clinical
    prediction and is not a substitute for site measurements or safety rules.
    """
    required_live = [
        "area", "forecast_timestamp", "projected_heat_risk_index",
        "forecast_environment_score", "forecast_workplace_score",
        "operational_profile_samples"
    ]
    missing = [c for c in required_live if c not in live_outlook.columns]
    if missing:
        raise ValueError(f"Missing live outlook fields: {missing}")
    if live_outlook.empty:
        return pd.DataFrame()

    def longest_run(mask):
        best = current = 0
        for flag in mask:
            current = current + 1 if bool(flag) else 0
            best = max(best, current)
        return best

    rows = []
    for area, group in live_outlook.groupby("area", sort=False):
        g = group.sort_values("forecast_timestamp").copy()
        g["score"] = pd.to_numeric(g["projected_heat_risk_index"], errors="coerce")
        g = g.dropna(subset=["score"])
        if g.empty:
            continue

        scores = g["score"].astype(float).reset_index(drop=True)
        times = pd.to_datetime(g["forecast_timestamp"]).reset_index(drop=True)
        peak_pos = int(scores.idxmax())
        peak_score = float(scores.iloc[peak_pos])
        first_score, last_score = float(scores.iloc[0]), float(scores.iloc[-1])
        delta = last_score - first_score
        direction = ("Rising" if delta >= 0.03 else
                     "Falling" if delta <= -0.03 else "Stable")
        slope = float(np.polyfit(np.arange(len(scores)), scores.to_numpy(), 1)[0]) if len(scores) > 1 else 0.0

        # Threshold exposure is counted in hourly forecast slots, not inferred illness.
        guarded_mask = scores >= 0.30
        elevated_mask = scores >= 0.46
        high_mask = scores >= 0.62
        guarded_hours = int(guarded_mask.sum())
        elevated_hours = int(elevated_mask.sum())
        high_hours = int(high_mask.sum())
        sustained_elevated_run = longest_run(elevated_mask.tolist())

        peak_time = times.iloc[peak_pos]
        peak_hour = int(peak_time.hour)
        if sustained_elevated_run >= 3:
            pattern = "Sustained elevated exposure"
        elif 5 <= peak_hour <= 11 and peak_score >= max(first_score, last_score) + 0.025:
            pattern = "Morning peak"
        elif 12 <= peak_hour <= 17 and peak_score >= max(first_score, last_score) + 0.025:
            pattern = "Afternoon peak"
        elif direction == "Rising":
            pattern = "Rising"
        elif direction == "Falling":
            pattern = "Falling"
        else:
            pattern = "Stable"

        env_values = pd.to_numeric(g["forecast_environment_score"], errors="coerce")
        work_values = pd.to_numeric(g["forecast_workplace_score"], errors="coerce")
        mean_env = float((0.65 * env_values).mean())
        mean_workplace = float((0.35 * work_values).mean())
        contribution_total = max(mean_env + mean_workplace, 1e-9)
        env_pct = 100.0 * mean_env / contribution_total
        workplace_pct = 100.0 * mean_workplace / contribution_total
        driver = "Environmental conditions" if mean_env >= mean_workplace else "Workplace exposure profile"

        sample_counts = pd.to_numeric(g["operational_profile_samples"], errors="coerce").fillna(0)
        min_samples = int(sample_counts.min()) if len(sample_counts) else 0
        if min_samples >= 10:
            coverage = "Higher historical coverage"
        elif min_samples >= 4:
            coverage = "Moderate historical coverage"
        else:
            coverage = "Limited historical coverage"

        tier = ("Low" if peak_score < 0.30 else
                "Guarded" if peak_score < 0.46 else
                "Elevated" if peak_score < 0.62 else "High")
        if tier == "High":
            action = "Activate the site's established heat-safety protocol; supervisor review and prompt reassessment"
        elif tier == "Elevated":
            action = "Plan preventive controls: shaded rest, hydration access, cooling and supervisor checks"
        elif tier == "Guarded":
            action = "Increase monitoring and verify conditions, workload and planned breaks"
        else:
            action = "Continue routine monitoring and maintain access to water, shade and rest"

        rows.append({
            "area": area,
            "demo_operational_profile": (str(g["demo_operational_profile"].dropna().iloc[0])
                                         if "demo_operational_profile" in g.columns and g["demo_operational_profile"].notna().any()
                                         else "Unspecified synthetic operational profile"),
            "mean_projected_workload": float(pd.to_numeric(g.get("workload_index", pd.Series(dtype=float)), errors="coerce").mean()) if "workload_index" in g.columns else np.nan,
            "mean_projected_shift_exposure": float(pd.to_numeric(g.get("shift_exposure_index", pd.Series(dtype=float)), errors="coerce").mean()) if "shift_exposure_index" in g.columns else np.nan,
            "forecast_peak_index": peak_score,
            "forecast_peak_tier": tier,
            "forecast_peak_time": peak_time,
            "forecast_start_index": first_score,
            "forecast_end_index": last_score,
            "forecast_change": delta,
            "trajectory_direction": direction,
            "trend": pattern,  # retained for compatibility with existing UI
            "trajectory_pattern": pattern,
            "trend_slope_per_hour": slope,
            "guarded_hours": guarded_hours,
            "elevated_hours": elevated_hours,
            "high_hours": high_hours,
            "longest_elevated_run_hours": sustained_elevated_run,
            "leading_driver": driver,
            "environment_contribution_pct": env_pct,
            "workplace_contribution_pct": workplace_pct,
            "mean_environment_contribution": mean_env,
            "mean_workplace_contribution": mean_workplace,
            "minimum_hourly_profile_samples": min_samples,
            "context_coverage": coverage,
            "suggested_operational_action": action,
            "interpretation_boundary": "Area-level operational index; not an individual clinical risk or illness probability"
        })
    return pd.DataFrame(rows)

