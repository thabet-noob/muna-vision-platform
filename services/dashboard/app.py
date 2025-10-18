import os, time
import numpy as np
from dash import Dash, html, dcc, Output, Input, no_update
import plotly.graph_objects as go
import cv2

from src.common.video import VideoStream
from src.pipelines.inference.pipeline import detect_and_annotate, compute_happiness, to_jpeg

RTSP_URL = os.environ.get("RTSP_URL", "rtsp://USER:PASS@192.168.1.207:554/live2")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;8000000|reorder_queue_size;0|fflags;nobuffer"
CONF  = float(os.environ.get("CONF","0.35"))
IMGSZ = int(os.environ.get("IMGSZ","640"))
TICK_MS = int(os.environ.get("TICK_MS","500"))
DISP_W, DISP_H = 960, 540

stream = VideoStream(RTSP_URL).start()
_happy_score = 50.0
_tick = 0

def gauge_color(v: float) -> str:
    if v < 40: return "#ef4444"
    if v < 70: return "#f59e0b"
    return "#10b981"

def gauge_label(v: float) -> str:
    if v < 50: return "🙁 Low"
    if v < 60: return "😐 Neutral"
    return "🙂 Happy"

def gauge_figure(val: float, faces_seen: int) -> go.Figure:
    c = gauge_color(val)
    label = gauge_label(val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=max(0, min(100, val)),
        number={"suffix": "%", "font": {"size": 40, "color": "#111"}},
        delta={"reference": 50, "increasing": {"color": "#16a34a"}, "decreasing": {"color": "#dc2626"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar": {"color": c, "thickness": 0.35},
            "bgcolor": "#ffffff",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100],"color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": "#111", "width": 4}, "thickness": 0.75, "value": val},
        },
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Happiness • faces: {faces_seen}", "font": {"size": 16, "color": "#111"}},
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#ffffff",
        height=230,
        annotations=[
            dict(text=label, x=0.5, y=0.0, xref='paper', yref='paper', showarrow=False,
                 font=dict(size=16, color="#111"))
        ],
    )
    return fig

app = Dash(__name__)
app.title = "Muna School Academy — Students & People"
INK = "#fafafa"; BLACK = "#0b0b10"; WHITE = "#ffffff"; YELLOW = "#FDE047"; ORANGE = "#FB923C"; MUTED = "#9ca3af"

header = html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px","marginBottom": "12px"}, children=[
    html.Img(src=app.get_asset_url("muna.jpg"), style={
        "height": "56px", "width": "56px", "borderRadius": "10px",
        "objectFit": "cover", "boxShadow": "0 1px 2px rgba(0,0,0,.5)"}),
    html.Div([
        html.H2("Muna School Academy", style={"margin": 0, "color": INK}),
        html.Div("@Omar_chelligue • Year 1", style={"margin": 0, "color": MUTED, "fontSize": "14px"}),
    ]),
])

app.layout = html.Div(style={
        "minHeight":"100vh","display":"flex","flexDirection":"column","background":BLACK,"color":INK,
        "fontFamily":"-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Inter, sans-serif","padding":"20px"
    }, children=[
        header,
        html.Div(style={"display":"flex","gap":"16px","alignItems":"stretch"}, children=[
            html.Div(style={"flex":3,"position":"relative"}, children=[
                html.Div("LIVE", style={
                    "position":"absolute","top":"10px","left":"10px","background":"#ef4444","color":"white",
                    "padding":"4px 8px","borderRadius":"999px","fontSize":"12px","fontWeight":700,"zIndex":2}),
                html.Img(id="frame", style={"width":"100%","borderRadius":"16px","border":"1px solid rgba(255,255,255,.12)",
                        "boxShadow":"0 10px 20px rgba(0,0,0,.5)"}),
            ]),
            html.Div(style={"flex":1,"display":"flex","flexDirection":"column","gap":"16px"}, children=[
                html.Div([ dcc.Graph(id="happy-gauge", config={"displayModeBar": False}, style={"width":"100%"}) ], 
                    style={"background":WHITE,"borderRadius":"16px","padding":"4px 8px 0 8px",
                           "border":"1px solid rgba(0,0,0,.1)","boxShadow":"0 10px 20px rgba(0,0,0,.35)"}),
                html.Div([ html.Div("Persons", style={"fontSize":"16px","color":"#111"}),
                           html.Div(id="kpi-persons", style={ "fontSize":"120px","fontWeight":900,"lineHeight":1,
                           "letterSpacing":"-2px","color":"#111","textAlign":"center"})],
                    style={"background":WHITE,"borderRadius":"16px","padding":"16px 20px","border":"1px solid rgba(0,0,0,.1)",
                           "boxShadow":"0 10px 20px rgba(0,0,0,.35)"}),
                html.Div([ html.Div("Students", style={"fontSize":"16px","color":"#111"}),
                           html.Div(id="kpi-students", style={ "fontSize":"100px","fontWeight":800,"lineHeight":1,
                           "letterSpacing":"-1px","color":"#111","textAlign":"center"})],
                    style={"background":YELLOW,"borderRadius":"16px","padding":"16px 20px","border":"1px solid rgba(0,0,0,.12)",
                           "boxShadow":"0 10px 20px rgba(0,0,0,.35)"}),
                html.Div([ html.Div("Non‑students", style={"fontSize":"16px","color":"#111"}),
                           html.Div(id="kpi-nonstudents", style={ "fontSize":"100px","fontWeight":800,"lineHeight":1,
                           "letterSpacing":"-1px","color":"#111","textAlign":"center"})],
                    style={"background":"#FB923C","borderRadius":"16px","padding":"16px 20px","border":"1px solid rgba(0,0,0,.12)",
                           "boxShadow":"0 10px 20px rgba(0,0,0,.35)"}),
            ]),
        ]),
        dcc.Interval(id="tick", interval=TICK_MS, n_intervals=0),
    ])

@app.callback(
    Output("frame", "src"),
    Output("kpi-persons", "children"),
    Output("kpi-students", "children"),
    Output("kpi-nonstudents", "children"),
    Output("happy-gauge", "figure"),
    Input("tick", "n_intervals"),
)
def update(n):
    global _tick, _happy_score
    _tick += 1
    frame = stream.read()
    if frame is None:
        return no_update, no_update, no_update, no_update, no_update
    try:
        annotated, people, students, head_rois = detect_and_annotate(frame, imgsz=IMGSZ, conf=CONF)
        non_students = max(0, people - students)
        src = to_jpeg(annotated, size=(960,540))
        h = compute_happiness(frame, head_rois) if (_tick % 2 == 0) else _happy_score
        fig = gauge_figure(h, faces_seen=people)
        return src, str(people), str(students), str(non_students), fig
    except Exception as e:
        print("update error:", e)
        return no_update, no_update, no_update, no_update, no_update

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
    finally:
        stream.stop()
