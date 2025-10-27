import numpy as np
import plotly.graph_objects as go
import streamlit as st
from st_aggrid import AgGrid
import pandas as pd

st.set_page_config(page_title="GC-MS 2D Visualizer", layout="wide")

def gcms_visualizer(im, rt, masses):
    st.title("GC-MS 2D Visualizer")

    if 'generate' not in st.session_state:
        st.session_state.generate = False

    with st.sidebar:
        st.header("Control Panel")

        st.subheader("Retention Time Range")
        col1, col2 = st.columns(2)
        with col1:
            rt_min = st.number_input("Min RT", float(rt.min()), float(rt.max()), float(rt.min()), step=0.1)
        with col2:
            rt_max = st.number_input("Max RT", float(rt.min()), float(rt.max()), float(rt.max()), step=0.1)

        st.subheader("Intensity Threshold")
        im_max = im.max() if np.any(im) else 1
        log_threshold = st.slider(
            "Log10 Minimum Intensity",
            0,
            int(np.log10(im_max)),
            value=2,
            step=1
        )
        min_intensity = 10**log_threshold
        st.write(f"Actual threshold: {min_intensity:,.0f}")

        st.subheader("Mass Selection")
        selected_masses = []
        for m in masses:
            if st.checkbox(f"m/z {m}", key=f"mass_{m}"):
                selected_masses.append(m)

        if st.button("Generate Visualization", type="primary"):
            st.session_state.generate = True

    if st.session_state.generate:
        time_mask = (rt >= rt_min) & (rt <= rt_max)

        has_signal = []
        max_intensities = []

        st.subheader("2D Signal Plots")
        with st.container():
            for m in selected_masses:
                idx = np.where(masses == m)[0][0]
                intensity = im[time_mask, idx]

                if intensity.max() >= min_intensity:
                    has_signal.append(m)
                    max_intensities.append(intensity.max())

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rt[time_mask],
                        y=intensity,
                        mode='lines',
                        line=dict(width=2),
                        name=f"m/z {m}",
                        hovertemplate="RT: %{x:.2f}<br>Intensity: %{y:.0f}<extra></extra>"
                    ))
                    fig.update_layout(
                        title=f"m/z {m}",
                        xaxis_title="Retention Time",
                        yaxis_title="Intensity",
                        height=300,
                        margin=dict(t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.success(f"Displayed {len(has_signal)} masses with intensity ≥ {min_intensity:,.0f}")

        if st.checkbox("Show mass list with intensities"):
            df = pd.DataFrame({
                "m/z": has_signal,
                "Max Intensity": max_intensities,
                "Log10 Intensity": np.log10(max_intensities)
            })
            AgGrid(
                df.sort_values("Max Intensity", ascending=False),
                height=300,
                gridOptions={
                    'pagination': True,
                    'paginationPageSize': 10,
                    'enableRangeSelection': True
                }
            )

        if st.button("Reset Parameters"):
            st.session_state.generate = False
            st.experimental_rerun()



def get_matrix(ms):
    ms = ms.to_numpy()
    RT = ms[:,0]
    im = ms[:,2:]
    TIC = ms[:,1]
    return RT, im, TIC

# Example usage (replace with your actual data)
if __name__ == "__main__":
    file = pd.read_csv("C:\\Users\\vvelezpe\\OneDrive - Imperial College London\\Projects\\VAPOR\\vapor_untargeted_output\\Average.csv")

    RT,im,_ = get_matrix(file)
    mz = np.arange(501)
    
    gcms_visualizer(im, RT, mz)