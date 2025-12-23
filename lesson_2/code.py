import time
import threading
import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from CapsuleSDK.Capsule import Capsule
from CapsuleSDK.DeviceLocator import DeviceLocator
from CapsuleSDK.DeviceType import DeviceType
from CapsuleSDK.Device import Device
from CapsuleSDK.EEGTimedData import EEGTimedData
from CapsuleSDK.Resistances import Resistances

from eeg_utils import *

# Конфиг
PLATFORM = 'mac'
EEG_WINDOW_SECONDS = 4.0
CHANNELS = 4
BUFFER_LEN = int(SAMPLE_RATE * EEG_WINDOW_SECONDS)
TARGET_SERIAL = None

ALPHA_LOW, ALPHA_HIGH = 8.0, 12.0
BETA_LOW, BETA_HIGH = 13.0, 30.0

ALPHA_THRESHOLD = 5e-12
CALIBRATION_DURATION = 10.0
progress_value = 0
last_accum_time = None
progress_step_sec = 0.1

device = None
device_locator = None


class EventFiredState:
    def __init__(self): self._awake = False

    def is_awake(self): return self._awake

    def set_awake(self): self._awake = True

    def sleep(self): self._awake = False


device_list_event = EventFiredState()
device_conn_event = EventFiredState()
device_eeg_event = EventFiredState()

ring = RingBuffer(n_channels=CHANNELS, maxlen=BUFFER_LEN)
channel_names = []
resistances_values = [0.0] * CHANNELS

eeg_filter = RealTimeFilter(sfreq=SAMPLE_RATE, l_freq=1, h_freq=40, n_channels=CHANNELS)

def non_blocking_cond_wait(wake_event: EventFiredState, name: str, total_sleep_time: int):
    print(f"Waiting {name} up to {total_sleep_time}s...")
    steps = int(total_sleep_time * 50)
    for _ in range(steps):
        if device_locator is not None:
            try:
                device_locator.update()
            except:
                pass
        if wake_event.is_awake(): return True
        time.sleep(0.02)
    return False


def on_device_list(locator, info, fail_reason):
    global device
    chosen = None
    if len(info) == 0:
        print("No devices found.")
        return
    print(f"Found {len(info)} devices.")
    if TARGET_SERIAL is None:
        chosen = info[0]
    else:
        for dev in info:
            if dev.get_serial() == TARGET_SERIAL:
                chosen = dev
                break
    if chosen is None:
        print(f"Target device {TARGET_SERIAL} not found!")
        return
    print(f"\nConnecting to: {chosen.get_serial()} ({chosen.get_name()})")
    device = Device(locator, chosen.get_serial(), locator.get_lib())
    device_list_event.set_awake()


def on_connection_status_changed(d, status):
    global channel_names
    print("Connection status changed:", status)
    ch_obj = device.get_channel_names()
    channel_names = [ch_obj.get_name_by_index(i) for i in range(len(ch_obj))]
    device_conn_event.set_awake()


def on_resistances(resistances_obj: Resistances):
    global resistances_values
    resistances_values = [resistances_obj.get_value(i) / 1000 for i in range(len(resistances_obj))]


def on_eeg(d, eeg: EEGTimedData):
    global ring
    samples = eeg.get_samples_count()
    ch = eeg.get_channels_count()
    if samples <= 0: return

    block = np.zeros((ch, samples), dtype=float)
    for i in range(samples):
        for c in range(ch):
            block[c, i] = eeg.get_processed_value(c, i)

    filtered_block = eeg_filter.filter_block(block[:CHANNELS, :])

    ring.append_block(filtered_block)
    if not device_eeg_event.is_awake():
        device_eeg_event.set_awake()

fig, (ax_eeg, ax_psd, ax_bands, ax_progress) = plt.subplots(4, 1, figsize=(10, 12))

# EEG
lines_eeg = [ax_eeg.plot([], [], lw=1, label=f'Ch{i}')[0] for i in range(CHANNELS)]
ax_eeg.set_title("Filtered EEG (1-40 Hz)")
ax_eeg.set_ylabel("µV")
ax_eeg.legend(loc='upper right', fontsize='small')
ax_eeg.grid(True)

# PSD
lines_psd = [ax_psd.plot([], [], lw=1, label=f'Ch{i}')[0] for i in range(CHANNELS)]
ax_psd.set_title("PSD")
ax_psd.set_xlim(0, 45)
ax_psd.set_ylim(0, 1e-11)
ax_psd.set_ylabel("µV²/Hz")
ax_psd.grid(True)

# Alpha/Beta power
alpha_avg_history = []
beta_avg_history = []
time_history = []
line_alpha, = ax_bands.plot([], [], 'b-', lw=2, label='Avg Alpha (8-12Hz)')
line_beta, = ax_bands.plot([], [], 'r-', lw=2, label='Avg Beta (13-30Hz)')
thr_line = ax_bands.axhline(ALPHA_THRESHOLD, color='gray', linestyle='--', label='Alpha Threshold')
ax_bands.set_title("Rhythms Power")
ax_bands.set_ylabel("µV²/Hz")
ax_bands.legend(loc='upper right', fontsize='small')
ax_bands.grid(True)

# 4. progress bar
ax_progress.set_xlim(0, 100)
ax_progress.set_ylim(-0.5, 0.5)
ax_progress.set_title("BCI Progress")
bar_container = ax_progress.barh([0], [0], height=0.8, color='green', alpha=0.8)
progress_bar = bar_container[0]
ax_progress.set_yticks([])
txt_progress = ax_progress.text(50, 0, "0%", ha='center', va='center', fontweight='bold')

txt_imp = fig.text(0.02, 0.02, "Impedance: ...", fontsize=9, family='monospace')


def update_plot(_):
    global channel_names, alpha_avg_history, beta_avg_history, time_history
    global progress_value, last_accum_time

    buf = ring.get()
    if buf.shape[1] == 0: return lines_eeg + lines_psd + [line_alpha, line_beta, progress_bar]

    current_time = time.time()
    t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])

    for i in range(CHANNELS):
        lines_eeg[i].set_data(t, buf[i, :])
        if i < len(channel_names): lines_eeg[i].set_label(channel_names[i])

    all_eeg = buf.flatten()
    ymin, ymax = all_eeg.min(), all_eeg.max()
    if ymin == ymax: ymin -= 1e-6; ymax += 1e-6
    ax_eeg.set_ylim(ymin * 1.1, ymax * 1.1)
    ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0)

    try:
        freqs, psd = compute_psd_mne(buf, sfreq=SAMPLE_RATE, fmin=1.0, fmax=50.0)
        for i in range(min(psd.shape[0], CHANNELS)):
            lines_psd[i].set_data(freqs, psd[i, :])

        alpha_pows = integrate_band(freqs, psd, ALPHA_LOW, ALPHA_HIGH)
        beta_pows = integrate_band(freqs, psd, BETA_LOW, BETA_HIGH)

        avg_alpha = np.mean(alpha_pows)
        avg_beta = np.mean(beta_pows)

        time_history.append(current_time)
        alpha_avg_history.append(avg_alpha)
        beta_avg_history.append(avg_beta)

        MAX_HIST = 100
        if len(time_history) > MAX_HIST:
            time_history = time_history[-MAX_HIST:]
            alpha_avg_history = alpha_avg_history[-MAX_HIST:]
            beta_avg_history = beta_avg_history[-MAX_HIST:]

        if len(time_history) > 0:
            t_rel = [x - time_history[0] for x in time_history]
            line_alpha.set_data(t_rel, alpha_avg_history)
            line_beta.set_data(t_rel, beta_avg_history)
            ax_bands.set_xlim(t_rel[0], t_rel[-1])

            max_val = max(max(alpha_avg_history), max(beta_avg_history), ALPHA_THRESHOLD)
            ax_bands.set_ylim(0, max_val * 1.2)

        if len(time_history) > 0 and (current_time - time_history[0]) >= CALIBRATION_DURATION:
            now = time.time()
            if last_accum_time is None: last_accum_time = now
            if now - last_accum_time >= progress_step_sec:
                if avg_alpha > ALPHA_THRESHOLD:
                    progress_value = min(100, progress_value + 2)
                else:
                    progress_value = max(0, progress_value - 1)
                last_accum_time = now
            progress_bar.set_width(progress_value)
            txt_progress.set_text(f"{progress_value}%")
            # Color change based on progress
            progress_bar.set_color('red' if progress_value < 30 else 'yellow' if progress_value < 70 else 'green')

    except Exception as e:
        print(f"Update Error: {e}")

    imp_str = "Impedance (kOhm): " + " | ".join(
        [f"{channel_names[i] if i < len(channel_names) else i}: {resistances_values[i]:.1f}" for i in range(CHANNELS)])
    txt_imp.set_text(imp_str)

    return lines_eeg + lines_psd + [line_alpha, line_beta, progress_bar]


def main():
    global device_locator, device

    # Путь к библиотеке относительно файла скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if PLATFORM == 'win':
        lib_path = os.path.join(script_dir, 'CapsuleClient.dll')
    else:
        lib_path = os.path.join(script_dir, 'libCapsuleClient.dylib')

    capsuleLib = Capsule(lib_path)

    device_locator = DeviceLocator(capsuleLib.get_lib())
    device_locator.set_on_devices_list(on_device_list)
    device_locator.request_devices(device_type=DeviceType.Band, seconds_to_search=10)

    if not non_blocking_cond_wait(device_list_event, 'device list', 12): return

    device.set_on_connection_status_changed(on_connection_status_changed)
    device.set_on_eeg(on_eeg)
    device.set_on_resistances(lambda d, r: on_resistances(r))
    device.connect(bipolarChannels=False)

    if not non_blocking_cond_wait(device_conn_event, 'device connection', 20): return

    device.start()
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    running = True

    def updater():
        while running:
            try:
                device_locator.update()
            except:
                pass
            time.sleep(0.01)

    threading.Thread(target=updater, daemon=True).start()
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    running = False
    device.stop()
    device.disconnect()


if __name__ == '__main__':
    main()