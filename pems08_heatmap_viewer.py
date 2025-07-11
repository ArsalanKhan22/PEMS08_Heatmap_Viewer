import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io


class PEMS08HeatmapApp:
    """
    Streamlit-based interactive application for visualizing traffic heatmaps
    from the PEMS08 dataset. Supports user-defined slicing, feature selection,
    color tuning, and exporting the generated heatmap.
    """

    def __init__(self):
        """Initialize app configuration and constants."""
        self.feature_names = ["Traffic Flow", "Occupancy", "Speed"]
        self.dataset_shape = (17856, 170, 3)  # [time, sensors, features]
        self.data = None
        self.heatmap_data = None
        self.vmin = None
        self.vmax = None

    def load_data(self, uploaded_file):
        """
        Load the uploaded .dat file and reshape to [time, sensor, features].

        Parameters:
            uploaded_file (UploadedFile): The file uploaded through Streamlit.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            raw_bytes = uploaded_file.read()
            flat_data = np.frombuffer(raw_bytes, dtype=np.float32)
            self.data = flat_data.reshape(self.dataset_shape)
            st.success(f"`data.dat` loaded successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            st.error(f"❌ Failed to load or reshape file: {e}")
            return False

    def sidebar_controls(self):
        """
        Render Streamlit sidebar inputs for feature selection, slicing ranges,
        and color range sliders.

        Returns:
            dict: Dictionary containing all user input values.
        """
        st.sidebar.header("⚙️ Heatmap Settings")

        feature_index = st.sidebar.selectbox(
            "Select Feature",
            options=[0, 1, 2],
            format_func=lambda i: self.feature_names[i]
        )

        st.sidebar.markdown("### ⏱ Time Range")
        time_start = st.sidebar.number_input("Start Time Step", 0, self.data.shape[0] - 1, 0)
        time_steps = st.sidebar.number_input("Number of Time Steps", 1, self.data.shape[0] - time_start, 1000)

        st.sidebar.markdown("### 🛰 Sensor Range")
        sensor_start = st.sidebar.number_input("Start Sensor Index", 0, self.data.shape[1] - 1, 0)
        sensor_count = st.sidebar.number_input("Number of Sensors", 1, self.data.shape[1] - sensor_start, 50)

        return {
            "feature_index": feature_index,
            "time_start": time_start,
            "time_steps": time_steps,
            "sensor_start": sensor_start,
            "sensor_count": sensor_count
        }

    def compute_heatmap_data(self, feature_index, time_start, time_steps, sensor_start, sensor_count):
        """
        Slice and transpose the heatmap data based on user input.

        Parameters:
            feature_index (int): Index of the selected feature channel.
            time_start (int): Starting index for time.
            time_steps (int): Number of time steps to include.
            sensor_start (int): Starting sensor index.
            sensor_count (int): Number of sensors to include.
        """
        end_time = time_start + time_steps
        end_sensor = sensor_start + sensor_count
        sliced = self.data[time_start:end_time, sensor_start:end_sensor, feature_index]
        self.heatmap_data = sliced.T  # Transpose to [sensor, time]

        min_val = float(np.min(self.heatmap_data))
        max_val = float(np.max(self.heatmap_data))
        p1 = float(np.percentile(self.heatmap_data, 1))
        p99 = float(np.percentile(self.heatmap_data, 99))

        st.sidebar.markdown("### 🎨 Color Range")
        self.vmin = st.sidebar.slider("Min Color Value (vmin)", min_value=min_val, max_value=max_val, value=p1)
        self.vmax = st.sidebar.slider("Max Color Value (vmax)", min_value=min_val, max_value=max_val, value=p99)

    def render_heatmap(self, feature_index, time_start, sensor_start):
        """
        Generate and display the heatmap using Seaborn and Matplotlib.

        Parameters:
            feature_index (int): Selected feature index.
            time_start (int): Start time index (for axis tick labels).
            sensor_start (int): Start sensor index (for axis tick labels).
        """
        st.subheader(f"📊 Heatmap of Sensors Over Time (Channel {feature_index})")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            self.heatmap_data,
            cmap="inferno",
            ax=ax,
            vmin=self.vmin,
            vmax=self.vmax,
            cbar_kws={"label": self.feature_names[feature_index]}
        )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sensor ID")
        ax.set_title(f"Heatmap of Sensors Over Time (Channel {feature_index})")

        # Set tick labels (10 evenly spaced)
        ax.set_xticks(np.linspace(0, self.heatmap_data.shape[1], num=10, dtype=int))
        ax.set_xticklabels([f"{int(t + time_start)}" for t in np.linspace(0, self.heatmap_data.shape[1], num=10, dtype=int)])

        ax.set_yticks(np.linspace(0, self.heatmap_data.shape[0], num=10, dtype=int))
        ax.set_yticklabels([f"{int(s + sensor_start)}" for s in np.linspace(0, self.heatmap_data.shape[0], num=10, dtype=int)])

        st.pyplot(fig)

        return fig

    def download_heatmap_button(self, fig, feature_index):
        """
        Provide a Streamlit download button to save the heatmap figure as PNG.

        Parameters:
            fig (matplotlib.figure.Figure): The heatmap figure.
            feature_index (int): Feature channel index for file naming.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="💾 Download Heatmap as PNG",
            data=buf,
            file_name=f"heatmap_channel_{feature_index}.png",
            mime="image/png"
        )

    def run(self):
        """Main method to run the Streamlit app."""
        st.set_page_config(layout="wide")
        st.title("🚦 PEMS08 Traffic Heatmap Viewer (Modular, Enhanced)")

        uploaded_file = st.file_uploader("Upload `data.dat` (raw float32)", type=["dat"])
        if not uploaded_file:
            st.info("📂 Please upload the `data.dat` file to begin.")
            return

        if not self.load_data(uploaded_file):
            return

        inputs = self.sidebar_controls()
        self.compute_heatmap_data(**inputs)
        fig = self.render_heatmap(inputs["feature_index"], inputs["time_start"], inputs["sensor_start"])
        self.download_heatmap_button(fig, inputs["feature_index"])


if __name__ == "__main__":
    app = PEMS08HeatmapApp()
    app.run()
