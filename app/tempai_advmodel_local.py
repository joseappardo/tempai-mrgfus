import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
from scipy.optimize import curve_fit

# -----------------------
# Load models
# -----------------------
MODEL_PATH_NL = "../models/tempai_minimal_gbr.joblib"
MODEL_PATH_LIN = "../models/tempai_minimal_linreg.joblib"

model_nl = joblib.load(MODEL_PATH_NL)
model_lin = joblib.load(MODEL_PATH_LIN)

FEATURE_ORDER_MINIMAL = [
    "actualpower",
    "actualenergy",
    "actualduration",
    "elementson",
    "skullarea",
    "mean_sdr",
]

# -----------------------
# Helpers
# -----------------------
def log_fit(x, a, b):
    return a * np.log(x) + b

def calculate_energy_kj(power_w, duration_s):
    try:
        return (float(power_w) * float(duration_s)) / 1000.0
    except Exception:
        return 0.0

def _select_model(selected_model: str):
    return model_nl if selected_model == "Non-Linear Model" else model_lin

def on_fix_power(fix_power):
    # si activo Fix Power -> apaga Fix Duration
    if fix_power:
        return gr.update(value=False)
    return gr.update()

def on_fix_duration(fix_duration):
    # si activo Fix Duration -> apaga Fix Power
    if fix_duration:
        return gr.update(value=False)
    return gr.update()
   
def _set_interactive(fix_duration, fix_power):
    # Solo el fijo debería ser editable:
    # - fix_duration => duration editable, power no (calculated)
    # - fix_power    => power editable, duration no (calculated)
    if fix_power:
        return gr.update(interactive=False), gr.update(interactive=True)
    if fix_duration:
        return gr.update(interactive=True), gr.update(interactive=False)
    # ninguno marcado => tratamos como fix_duration
    return gr.update(interactive=True), gr.update(interactive=False)

# MAE por rango (según tu tabla)
# MAE por rango (según tus resultados)
MAE_BY_NLIN = {
    "<50": 1.893048,
    "50-55": 1.633445,
    "55-60": 1.771758,
    ">=60": 5.286237
}

MAE_BY_LIN = {
    "<50": 2.39,
    "50-55": 2.50,
    "55-60": 2.06,
    ">=60": 5.66
}

def target_planner(selected_model, skullarea, mean_sdr, elementson,
                   target_temp, duration_s, power_w, fix_duration, fix_power):
    model = _select_model(selected_model)

    def _to_float(x, default):
        try:
            if x is None:
                return float(default)
            s = str(x).strip()
            if s == "" or s.upper() == "NA":
                return float(default)
            return float(s)
        except Exception:
            return float(default)

    T = _to_float(target_temp, 55.0)
    D = _to_float(duration_s, 11.0)
    P = _to_float(power_w, 800.0)

    # exclusividad lógica
    if fix_power:
        fix_duration = False
    if (not fix_duration) and (not fix_power):
        fix_duration = True  # default

    # helper msg (HTML rojo)
    def _red(msg: str) -> str:
        if not msg:
            return ""
        return (
            "<div style='margin-top:6px; padding:8px 10px; border-radius:8px; "
            "background: rgba(255,0,0,0.10); border: 1px solid rgba(255,0,0,0.35); "
            "color:#ff4d4d; font-weight:700;'>"
            f"{msg}"
            "</div>"
        )

    # FIX DURATION -> solve POWER
    if fix_duration:
        p_req = _solve_power_for_target(model, T, D, skullarea, mean_sdr, elementson)
        if p_req is None:
            # duration se mantiene, power queda vacío, msg rojo
            return int(round(D)), None, _red("Power out of range!")
        return int(round(D)), int(round(p_req)), ""

    # FIX POWER -> solve DURATION
    d_req = _solve_duration_for_target(model, T, P, skullarea, mean_sdr, elementson)
    if d_req is None:
        # power se mantiene, duration queda vacío, msg rojo
        return None, int(round(P)), _red("Duration out of range!")
    return int(round(d_req)), int(round(P)), ""


def _predict_temp_for(model, power_w, duration_s, skullarea, mean_sdr, elementson):
    e_kj = calculate_energy_kj(power_w, duration_s)
    input_df = pd.DataFrame({
        "actualpower": [float(power_w)],
        "actualenergy": [float(e_kj)],
        "actualduration": [float(duration_s)],
        "skullarea": [float(skullarea)],
        "mean_sdr": [float(mean_sdr)],
        "elementson": [float(elementson)],
    })[FEATURE_ORDER_MINIMAL]
    return float(model.predict(input_df)[0]), float(e_kj)

def _solve_power_for_target(model, target_temp, duration_s, skullarea, mean_sdr, elementson,
                            p_min=100, p_max=2000, tol=0.05, max_iter=40):
    lo, hi = float(p_min), float(p_max)
    t_lo, _ = _predict_temp_for(model, lo, duration_s, skullarea, mean_sdr, elementson)
    t_hi, _ = _predict_temp_for(model, hi, duration_s, skullarea, mean_sdr, elementson)

    # Si ni con p_max llegas -> NA
    if np.isfinite(t_hi) and t_hi < target_temp:
        return None

    # Si ya con p_min superas target, devuelve p_min (criterio >= target como tu tabla)
    if np.isfinite(t_lo) and t_lo >= target_temp:
        return lo

    # bisección
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        t_mid, _ = _predict_temp_for(model, mid, duration_s, skullarea, mean_sdr, elementson)
        if abs(t_mid - target_temp) <= tol:
            return mid
        if t_mid < target_temp:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def _solve_duration_for_target(model, target_temp, power_w, skullarea, mean_sdr, elementson,
                               d_min=5, d_max=30, tol=0.05, max_iter=40):
    lo, hi = float(d_min), float(d_max)
    t_lo, _ = _predict_temp_for(model, power_w, lo, skullarea, mean_sdr, elementson)
    t_hi, _ = _predict_temp_for(model, power_w, hi, skullarea, mean_sdr, elementson)

    if np.isfinite(t_hi) and t_hi < target_temp:
        return None

    if np.isfinite(t_lo) and t_lo >= target_temp:
        return lo

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        t_mid, _ = _predict_temp_for(model, power_w, mid, skullarea, mean_sdr, elementson)
        if abs(t_mid - target_temp) <= tol:
            return mid
        if t_mid < target_temp:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def target_temp_inverse(selected_model, skullarea, mean_sdr, elementson, target_temp, duration_s, power_w):
    """
    Devuelve:
    - power requerido si duration fijada
    - duration requerida si power fijado
    - energies resultantes (kJ)
    """
    model = _select_model(selected_model)

    target_temp = float(target_temp)
    duration_s = float(duration_s) if duration_s is not None else 11.0

    # si power no está informado, usa el actual del UI si se lo pasas; si queda None, no calculamos duration
    power_given = None
    try:
        if power_w is not None and str(power_w) != "":
            power_given = float(power_w)
    except Exception:
        power_given = None

    # 1) power requerido para esa duración
    p_req = _solve_power_for_target(model, target_temp, duration_s, skullarea, mean_sdr, elementson)
    _, e_req = _predict_temp_for(model, p_req, duration_s, skullarea, mean_sdr, elementson)

    # 2) duration requerida para esa potencia (si hay potencia)
    if power_given is not None:
        d_req = _solve_duration_for_target(model, target_temp, power_given, skullarea, mean_sdr, elementson)
        _, e_req2 = _predict_temp_for(model, power_given, d_req, skullarea, mean_sdr, elementson)
    else:
        d_req = None
        e_req2 = None

    # salida bonita
    return (
        round(p_req, 1),
        round(duration_s, 1),
        round(e_req, 3),
        (round(d_req, 2) if d_req is not None else "NA"),
        (round(power_given, 1) if power_given is not None else "NA"),
        (round(e_req2, 3) if e_req2 is not None else "NA"),
    )

def _mae_for_pred(selected_model: str, temp_c: float) -> float:
    """
    Devuelve el MAE esperado según el modelo y el rango de temperatura predicha.
    """
    mae_dict = MAE_BY_NLIN if selected_model == "Non-Linear Model" else MAE_BY_LIN

    if temp_c < 50:
        return mae_dict["<50"]
    elif temp_c < 55:
        return mae_dict["50-55"]
    elif temp_c < 60:
        return mae_dict["55-60"]
    else:
        return mae_dict[">=60"]


def _param_card(p, d, e, sa, sdr, ne, selected_model):
    # “Vector” de parámetros en una tarjeta compacta
    return (
        "<div style='margin-top:10px; padding:10px; border-radius:10px; "
        "background:#1a1a1a; color:#fff; font-size:14px;'>"
        "<div style='font-weight:700; margin-bottom:6px;'>Parameters used</div>"
        "<div style='display:flex; flex-wrap:wrap; gap:10px; opacity:0.95;'>"
        f"<div><b>Model</b>: {selected_model}</div>"
        f"<div><b>Power</b>: {p:.1f} W</div>"
        f"<div><b>Duration</b>: {d:.1f} s</div>"
        f"<div><b>Energy</b>: {e:.3f} kJ</div>"
        f"<div><b>Skull Area</b>: {sa:.2f}</div>"
        f"<div><b>Mean SDR</b>: {sdr:.3f}</div>"
        f"<div><b>N Elements</b>: {ne:.0f}</div>"
        "</div>"
        "</div>"
    )

def predict_html_and_params(selected_model, actualpower, actualduration, skullarea, mean_sdr, elementson):
    """
    Predicción + tarjeta de parámetros.
    Energy se recalcula aquí.
    """
    model = _select_model(selected_model)

    p = float(actualpower)
    d = float(actualduration)
    sa = float(skullarea)
    sdr = float(mean_sdr)
    ne = float(elementson)
    e_kj = calculate_energy_kj(p, d)

    input_df = pd.DataFrame({
        "actualpower": [p],
        "actualenergy": [e_kj],
        "actualduration": [d],
        "skullarea": [sa],
        "mean_sdr": [sdr],
        "elementson": [ne],
    })[FEATURE_ORDER_MINIMAL]

    pred = float(model.predict(input_df)[0])
    mae = _mae_for_pred(selected_model,pred)

    pred_html = (
        "<div style='font-size: 26px; background-color: #111111; color: #ffffff; "
        "padding: 12px; border-radius: 10px; text-align: center;'>"
        f"<b>{pred:.1f} °C</b> <span style='font-size:18px;'>(± {mae:.1f} °C)</span><br>"
        f"<span style='font-size:14px; opacity:0.85;'>Energy={e_kj:.3f} kJ</span>"
        "</div>"
    )

    params_html = _param_card(p, d, e_kj, sa, sdr, ne, selected_model)
    return pred_html, params_html, pred

def _required_power_from_curve(power_range_w, temps, target_temp):
    power_range_w = np.asarray(power_range_w, dtype=float)
    temps = np.asarray(temps, dtype=float)

    if np.nanmax(temps) < target_temp:
        return "NA"

    idxs = np.where(temps >= target_temp)[0]
    if len(idxs) == 0:
        return "NA"
    idx = int(idxs[0])

    if idx == 0:
        return round(float(power_range_w[0]), 1)

    x0, x1 = power_range_w[idx - 1], power_range_w[idx]
    y0, y1 = temps[idx - 1], temps[idx]
    if y1 == y0:
        return round(float(x1), 1)

    p = x0 + (target_temp - y0) * (x1 - x0) / (y1 - y0)
    return round(float(p), 1)

def make_plot_and_table(
    selected_model,
    skullarea, mean_sdr, elementson,
    x_axis_mode,   # "Energy (kJ)" or "Power (W)"
    smooth_mode    # "Log-fit (smooth)" or "Raw model"
):
    model = _select_model(selected_model)

    durations = [9, 10, 12, 15, 20]
    power_range_w = np.linspace(200, 1400, 120)
    temp_targets = [50, 55, 60]

    table_rows = []
    fig, ax = plt.subplots(figsize=(12, 8))

    for duration_s in durations:
        energies_kj = (power_range_w * duration_s) / 1000.0

        input_df = pd.DataFrame({
            "actualpower": power_range_w.astype(float),
            "actualenergy": energies_kj.astype(float),
            "actualduration": np.full_like(power_range_w, float(duration_s), dtype=float),
            "skullarea": np.full_like(power_range_w, float(skullarea), dtype=float),
            "mean_sdr": np.full_like(power_range_w, float(mean_sdr), dtype=float),
            "elementson": np.full_like(power_range_w, float(elementson), dtype=float),
        })[FEATURE_ORDER_MINIMAL]

        temps = model.predict(input_df).astype(float)

        if smooth_mode == "Log-fit (smooth)":
            try:
                popt, _ = curve_fit(log_fit, energies_kj, temps, maxfev=20000)
                temps_plot = log_fit(energies_kj, *popt)
            except Exception:
                temps_plot = temps
        else:
            temps_plot = temps

        if x_axis_mode == "Energy (kJ)":
            x = energies_kj
            ax.set_xlabel("Energy [kJ]")
        else:
            x = power_range_w
            ax.set_xlabel("Power [W]")

        ax.plot(x, temps_plot, linewidth=3, label=f"Duration: {duration_s}s")

        req = []
        for tt in temp_targets:
            req.append(_required_power_from_curve(power_range_w, temps_plot, tt))
        table_rows.append([duration_s] + req)

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    ax.set_title("Predicted Temperature Trajectories", fontsize=16, color="white")
    ax.set_ylabel("Peak Average Temperature [°C]")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.set_ylim(40, 65)
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")

    leg = ax.legend(fontsize=12, loc="center left", bbox_to_anchor=(1, 0.5), frameon=True)
    leg.get_frame().set_facecolor("black")
    leg.get_frame().set_edgecolor("white")
    for text in leg.get_texts():
        text.set_color("white")

    fig.tight_layout()

    table_df = pd.DataFrame(
        table_rows,
        columns=["Duration (s)", "Power for 50°C (W)", "Power for 55°C (W)", "Power for 60°C (W)"]
    )
    return fig, table_df

def _empty_log_df():
    return pd.DataFrame(columns=[
        "SonicationID",
        "Time",
        "Power (W)",
        "Duration (s)",
        "Energy (kJ)",
        "Peak Avg Temp (Real, °C)",
        "Prediction Non-linear (°C)",
        "Error (Real - Pred, °C)",
    ])

def _sonication_plot(log_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 7.0))  # más grande

    # tamaños “fuertes”
    TITLE_FS = 22
    LABEL_FS = 18
    TICK_FS  = 14
    ANNO_FS  = 12
    LEG_FS   = 14
    POINT_S  = 90  # tamaño puntos

    if log_df is None or len(log_df) == 0:
        ax.set_title("Sonication log: Energy vs Temperature", fontsize=TITLE_FS)
        ax.set_xlabel("Energy [kJ]", fontsize=LABEL_FS)
        ax.set_ylabel("Peak Avg Temp (Real) [°C]", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig

    x = log_df["Energy (kJ)"].astype(float).values
    y = log_df["Peak Avg Temp (Real, °C)"].astype(float).values

    # puntos negros (más grandes)
    ax.scatter(x, y, s=POINT_S)

    # labels: "ID: power/duration"
    for _, r in log_df.iterrows():
        label = (
            f"ID {int(r['SonicationID'])}\n"
            f"P={float(r['Power (W)']):.0f} W\n"
            f"D={float(r['Duration (s)']):.0f} s"
        )
        ax.annotate(
            label,
            (float(r["Energy (kJ)"]), float(r["Peak Avg Temp (Real, °C)"])),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=ANNO_FS
        )

    # log-fit gris
    try:
        mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) >= 3:
            popt, _ = curve_fit(log_fit, x[mask], y[mask], maxfev=20000)
            x_fit = np.linspace(np.min(x[mask]), np.max(x[mask]), 250)
            y_fit = log_fit(x_fit, *popt)
            ax.plot(x_fit, y_fit, linewidth=4, alpha=0.6)
    except Exception:
        pass

    ax.set_title("Sonication log: Energy vs Temperature", fontsize=TITLE_FS)
    ax.set_xlabel("Energy [kJ]", fontsize=LABEL_FS)
    ax.set_ylabel("Peak Avg Temp (Real) [°C]", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def set_sonication(
    log_df,
    peak_real_temp,
    actualpower,
    actualduration,
    skullarea,
    mean_sdr,
    elementson
):
    """
    Registra una sonicación usando:
    - valores actuales (power/duration -> energy recalculada)
    - predicción NON-LINEAR (siempre) para la columna Prediction Non-linear
    """
    if log_df is None or not isinstance(log_df, pd.DataFrame):
        log_df = _empty_log_df()

    p = float(actualpower)
    d = float(actualduration)
    sa = float(skullarea)
    sdr = float(mean_sdr)
    ne = float(elementson)
    e_kj = calculate_energy_kj(p, d)
    real_t = float(peak_real_temp)

    # pred non-linear SIEMPRE
    input_df = pd.DataFrame({
        "actualpower": [p],
        "actualenergy": [e_kj],
        "actualduration": [d],
        "skullarea": [sa],
        "mean_sdr": [sdr],
        "elementson": [ne],
    })[FEATURE_ORDER_MINIMAL]
    pred_nl = float(model_nl.predict(input_df)[0])
    err = real_t - pred_nl

    son_id = 1 if len(log_df) == 0 else int(log_df["SonicationID"].max()) + 1
    now_str = datetime.now().strftime("%H:%M:%S")

    new_row = {
        "SonicationID": son_id,
        "Time": now_str,
        "Power (W)": p,
        "Duration (s)": d,
        "Energy (kJ)": e_kj,
        "Peak Avg Temp (Real, °C)": real_t,
        "Prediction Non-linear (°C)": pred_nl,
        "Error (Real - Pred, °C)": err,
    }

    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    fig = _sonication_plot(log_df)
    return log_df, fig

def exit_app(log_df):
    try:
        save_treatment_artifacts(log_df)
    except Exception as e:
        print(f"[WARN] Could not save treatment logs: {e}")
    os._exit(0)

def _ensure_log_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "treatmentlogs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_treatment_artifacts(log_df: pd.DataFrame):
    out_dir = _ensure_log_dir()
    ts = _timestamp()

    # CSV
    csv_path = os.path.join(out_dir, f"sonication_log_{ts}.csv")
    if log_df is None:
        log_df = _empty_log_df()
    log_df.to_csv(csv_path, index=False)

    # PNG (figura resumen)
    fig = _sonication_plot(log_df)
    png_path = os.path.join(out_dir, f"sonication_summary_{ts}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)  # importante para no acumular figuras

    return csv_path, png_path


# -----------------------
# Gradio Interface (Local)
# -----------------------
theme = gr.themes.Ocean()

with gr.Blocks(theme=theme, title="MRgFUS TempAI v0.1-beta") as iface:
    log_state = gr.State(_empty_log_df())

    with gr.Row():
        with gr.Column(scale=1, min_width=160):
            gr.Button("⛔ Save & Exit", variant="stop").click(fn=exit_app, inputs=[log_state], outputs=None)

        with gr.Column(scale=6):
            gr.Markdown("<h2 style='text-align: center; font-size: 30px;'>MRgFUS TempAI v0.3-beta</h2>")

    with gr.Row():
        with gr.Column(scale=1):
            selected_model = gr.Dropdown(
                choices=["Non-Linear Model", "Linear Model"],
                label="Model",
                value="Non-Linear Model"
            )

            x_axis_mode = gr.Radio(
                choices=["Energy (kJ)", "Power (W)"],
                value="Energy (kJ)",
                label="X-axis"
            )

            smooth_mode = gr.Radio(
                choices=["Log-fit (smooth)", "Raw model"],
                value="Log-fit (smooth)",
                label="Trajectories"
            )

            actualpower = gr.Number(label="Power (W)", precision=0, value=200)
            actualduration = gr.Number(label="Duration (s)", precision=0, value=10)

            skullarea = gr.Number(label="Skull Area", precision=0, value=336)
            mean_sdr = gr.Number(label="Mean SDR", precision=3, value=0.57)
            elementson = gr.Number(label="N Elements", precision=0, value=986)

            output_html = gr.HTML()
            params_html = gr.HTML()

            gr.Markdown("### Sonication")
            peak_real_temp = gr.Number(label="Peak Avg Temp (Real, °C)", precision=1, value=55.0)
            set_btn = gr.Button("Set Sonication", variant="primary")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Trajectory Plot")
            table_output = gr.Dataframe(
                label="Required Power Table",
                headers=["Duration (s)", "Power for 50°C (W)", "Power for 55°C (W)", "Power for 60°C (W)"]
            )
            with gr.Group():
                gr.Markdown("### Target (inverse planning)")
            
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            target_temp = gr.Number(label="Target Peak Avg Temp (°C)", precision=1, value=55.0)
                        with gr.Row():
                            planner_msg = gr.HTML(value="")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            fix_duration = gr.Checkbox(label="Fix duration", value=True)
                            fix_power = gr.Checkbox(label="Fix power", value=False)
            
                        with gr.Row():
                            target_duration = gr.Number(label="Duration (s)", precision=0, value=11)
                            target_power = gr.Number(label="Power (W)", precision=0, value=800)


    with gr.Row():
        with gr.Column(scale=2):
            son_table = gr.Dataframe(label="Sonication Log", interactive=False)
        with gr.Column(scale=1):
            son_plot = gr.Plot(label="Energy vs Temp (Real)")

    # -----------------------
    # Auto-update wiring
    # IMPORTANT: usamos trigger_mode="always_last" + queue(True) para evitar el bug de tecleo rápido
    # -----------------------
    pred_controls = [selected_model, actualpower, actualduration, skullarea, mean_sdr, elementson]
    for c in pred_controls:
        c.input(
            fn=predict_html_and_params,
            inputs=[selected_model, actualpower, actualduration, skullarea, mean_sdr, elementson],
            outputs=[output_html, params_html, peak_real_temp],
            trigger_mode="always_last"
        )

    plot_controls = [selected_model, skullarea, mean_sdr, elementson, x_axis_mode, smooth_mode]
    for c in plot_controls:
        c.input(
            fn=make_plot_and_table,
            inputs=[selected_model, skullarea, mean_sdr, elementson, x_axis_mode, smooth_mode],
            outputs=[plot_output, table_output],
            trigger_mode="always_last"
        )

    # 2) Activar/desactivar edición en power/duration según lock
    fix_power.change(
        fn=on_fix_power,
        inputs=[fix_power],
        outputs=[fix_duration],
        trigger_mode="always_last"
    )
    
    fix_duration.change(
        fn=on_fix_duration,
        inputs=[fix_duration],
        outputs=[fix_power],
        trigger_mode="always_last"
    )

    
    # 3) Planner: recalcula el parámetro libre
    planner_inputs = [selected_model, skullarea, mean_sdr, elementson,
                      target_temp, target_duration, target_power, fix_duration, fix_power]
    
    for c in [target_temp, target_duration, target_power, fix_duration, fix_power,
              selected_model, skullarea, mean_sdr, elementson]:
        c.input(
            fn=target_planner,
            inputs=planner_inputs,
            outputs=[target_duration, target_power, planner_msg],
            trigger_mode="always_last"
        )



    # Set sonication: usa valores actuales + temp real
    set_btn.click(
        fn=set_sonication,
        inputs=[log_state, peak_real_temp, actualpower, actualduration, skullarea, mean_sdr, elementson],
        outputs=[son_table, son_plot],
        trigger_mode="always_last"
    ).then(
        fn=lambda df: df,
        inputs=son_table,
        outputs=log_state,
        trigger_mode="always_last"
    )

    # Trigger once at load
    iface.load(
        fn=lambda sm, p, d, sa, sdr, ne, xmode, smode: (
            *predict_html_and_params(sm, p, d, sa, sdr, ne),
            *make_plot_and_table(sm, sa, sdr, ne, xmode, smode),
            _empty_log_df(),
            _sonication_plot(_empty_log_df()),
            *target_planner(sm, sa, sdr, ne, 55.0, 11, 800, True, False),
            True,  # fix_duration
            False  # fix_power
        ),
        inputs=[selected_model, actualpower, actualduration, skullarea, mean_sdr, elementson, x_axis_mode, smooth_mode],
        outputs=[
            output_html, params_html, peak_real_temp,
            plot_output, table_output,
            son_table, son_plot,
            target_duration, target_power, planner_msg, fix_duration, fix_power
        ]
    )

if __name__ == "__main__":
    # CLAVE: queue activada para que trigger_mode="always_last" funcione bien ante inputs rápidos
    iface.queue(max_size=32)
    iface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )
