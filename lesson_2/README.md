# Курс «Нейронауки и нейроинтерфейсы». Неделя 13


## 🧠 Проектное занятие №2


Добро пожаловать на второе проектное занятие по курсу Центрального университета «Нейронауки и нейроинтерфейсы» в 2025 году! Сегодня мы научимся применять библиотеку MNE для анализа, визуализируем ЭЭГ и PSD всех каналов в реальном времени, рассчитаем мощность альфа/бета-ритмов и напишем простой классификатор для отправки 0/1 на устройство.


---


### Цель занятия


Научиться визуализировать ЭЭГ, PSD в реальном времени, анализировать их с помощью MNE и визуализировать, включая разработку простого классификатора на основе мощности альфа/бета-ритмов.


---


### По итогам занятия вы сможете:


1) отрисовывать графики EEG, PSD и альфа-мощности в реальном времени;
2) использовать MNE для анализа потоковых данных ЭЭГ;
3) реализовывать простой классификатор на основе порога альфа-мощности.


---


## Требования перед стартом


- Python 3.11+
- `CapsuleClient.dll` (Windows) или `libCapsuleClient.dylib` (macOS) в папке с кодом
- Подключённое устройство Neiry HeadBand по Bluetooth
- Установленные зависимости
```bash
pip install numpy matplotlib mne
```


### 1. Введение в eeg_utils.py


Файл `eeg_utils.py` содержит полезные инструменты для работы с потоковыми данными ЭЭГ.


`SAMPLE_RATE`: константа, определяющая частоту дискретизации (предполагается 250 Гц).
`RingBuffer`: класс кольцевого буфера для хранения последних N сэмплов данных с нескольких каналов. Полезен для визуализации и анализа текущего «окна» данных.
`compute_psd_mne`: функция для вычисления спектральной плотности мощности (PSD) с использованием MNE.
`integrate_band`: функция для вычисления интегральной мощности в заданной полосе частот (например, альфа-ритм 8–12 Гц) из данных PSD.


### 2. Отрисовка ЭЭГ для всех каналов в реальном времени


Используется `matplotlib` и `FuncAnimation` для создания прокручивающегося графика ЭЭГ.


Данные ЭЭГ накапливаются в `RingBuffer`. `update_plot` функция вызывается регулярно, извлекает данные из буфера и обновляет линии графика.


---


#### Импорты библиотек, инициализации переменных
```python
from eeg_utils import RingBuffer, SAMPLE_RATE


# config
PLATFORM = 'mac'
EEG_WINDOW_SECONDS = 4.0
CHANNELS = 4 #4 if Bipolar=False, 2 otherwise
BUFFER_LEN = int(SAMPLE_RATE * EEG_WINDOW_SECONDS)
MAX_PLOT_CHANNELS = CHANNELS


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


# Инициализация буфера (n_channels, buffer_len)
ring = RingBuffer(n_channels=CHANNELS, maxlen=BUFFER_LEN)
# Инициализация channel_names как пустого списка, будет заполнено позже
channel_names = []
```
#### Функция ожидания события с таймаутом, обновляющая состояние локатора
```python
def non_blocking_cond_wait(wake_event: EventFiredState, name: str, total_sleep_time: int):
   print(f"Waiting {name} up to {total_sleep_time}s...")
   steps = int(total_sleep_time * 50)
   for _ in range(steps):
       if device_locator is not None:
           try:
               device_locator.update()
           except Exception:
               pass
       if wake_event.is_awake():
           return True
       time.sleep(0.02)
   return False


def on_device_list(locator, info, fail_reason):
   global device
   if len(info) == 0:
       print("No devices found in this scan.")
       return
   print(f"Found {len(info)} device(s). Using first device:")
   info0 = info[0]
   print("Serial:", info0.get_serial())
   print("Name:  ", info0.get_name())
   print("FW:    ", info0.get_firmware())
   print("Type:  ", info0.get_type())
   device = Device(locator, info0.get_serial(), locator.get_lib())
   device_list_event.set_awake()


def on_connection_status_changed(d, status):
   global channel_names
   print("Connection status changed:", status)
   ch_obj = device.get_channel_names()
   channel_names.extend([ch_obj.get_name_by_index(i) for i in range(len(ch_obj))])
   print(f"Channel names: {channel_names}")
   device_conn_event.set_awake()
```
#### Визуализация
```python
# Отрисовка на одном графике
fig, ax = plt.subplots(figsize=(10, 6))
lines = []
for i in range(CHANNELS):
   ln, = ax.plot([], [], label=f'Ch{i}', lw=1)
   lines.append(ln)


ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (µV)")
ax.set_title("EEG Channels")
ax.legend(loc='upper right')
ax.grid(True)


def update_plot(_):
   global channel_names
   buf = ring.get()
   if buf.shape[1] == 0:
       return lines
   t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
   # Обновляем данные
   for i in range(CHANNELS):
       lines[i].set_data(t, buf[i, :])
       # Обновляем имена каналов, если доступны
       if len(channel_names) > i:
            lines[i].set_label(channel_names[i])
       else:
            lines[i].set_label(f'Ch{i}')


   # Динамическое масштабирование по Y
   all_data = buf.flatten()
   ymin, ymax = all_data.min(), all_data.max()
   if ymin == ymax:
       ymin -= 1e-6; ymax += 1e-6
   pad = 0.1*(ymax - ymin)
   ax.set_ylim(ymin-pad, ymax+pad)
   ax.set_xlim(-EEG_WINDOW_SECONDS, 0)
   # Перерисовываем легенду
   ax.legend(loc='upper right')
   return lines
```
#### Основная функция подключения и получения информации
```python
def main():
   global device_locator, device
   if PLATFORM == 'win':
       capsuleLib = Capsule('./CapsuleClient.dll')
   else:
       capsuleLib = Capsule('./libCapsuleClient.dylib')


   device_locator = DeviceLocator(capsuleLib.get_lib())
   device_locator.set_on_devices_list(on_device_list)
   device_locator.request_devices(device_type=DeviceType.Band, seconds_to_search=10)


   if not non_blocking_cond_wait(device_list_event, 'device list', 12):
       print("No device found. Exiting.")
       return


   device.set_on_connection_status_changed(on_connection_status_changed)
   device.set_on_eeg(on_eeg)
   device.connect(bipolarChannels=False)
   if not non_blocking_cond_wait(device_conn_event, 'device connection', 20):
       print("Failed to connect.")
       return


   device.start()
   print("Device started. Opening plot...")


   ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)


   running = True
   def updater():
       while running:
           try:
               device_locator.update()
           except Exception:
               pass
           time.sleep(0.01)


   t = threading.Thread(target=updater, daemon=True)
   t.start()
   plt.tight_layout()
   plt.show()


   running = False
   device.stop()
   device.disconnect()
   print("Stopped.")


if __name__ == '__main__':
   main()
```
---


### 3. Отрисовка PSD для всех каналов в реальном времени
---


#### Импорт класса для работы с данными PSD и определение глобальных переменных и обработчика
```python
from CapsuleSDK.PSDData import PSDData


psd_freqs_global = None
psd_vals_global = None


def on_psd_data(dev: Device, psd: PSDData):
   global psd_freqs_global, psd_vals_global
   # Получаем частоты и значения PSD из объекта psd
   # (предполагаемые методы, уточните имена из SDK)
   psd_freqs_global = [psd.get_frequency(i) for i in range(psd.get_frequencies_count())]
   # Предполагаем, что psd.get_psd(channel_index, freq_index) возвращает значение
   # и что psd.get_channels_count() возвращает количество каналов
   psd_vals_global = [[psd.get_psd(ch, f) for f in range(psd.get_frequencies_count())] for ch in range(psd.get_channels_count())]
```


#### Визуализация
```python
# Отрисовка ЭЭГ и PSD на двух графиках
fig, (ax_eeg, ax_psd) = plt.subplots(2, 1, figsize=(10, 8), sharex=False) # Создаём два подграфика


lines_eeg = []
for i in range(CHANNELS):
   ln, = ax_eeg.plot([], [], label=f'Ch{i}', lw=1)
   lines_eeg.append(ln)
ax_eeg.set_ylabel("Amplitude (µV)")
ax_eeg.set_title("EEG Channels")
ax_eeg.legend(loc='upper right')
ax_eeg.grid(True)


lines_psd = []
for i in range(CHANNELS):
   ln, = ax_psd.plot([], [], label=f'PSD Ch{i}', lw=1)
   lines_psd.append(ln)
ax_psd.set_xlabel("Frequency (Hz)") # Ось X для PSD
ax_psd.set_ylabel("PSD (µV²/Hz)")
ax_psd.set_title("PSD Channels")
ax_psd.legend(loc='upper right')
ax_psd.grid(True)


def update_plot(_):
   global channel_names, psd_freqs_global, psd_vals_global # Добавляем psd переменные


   # --- Обновление EEG ---
   buf = ring.get()
   if buf.shape[1] != 0:
       t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
       for i in range(CHANNELS):
           lines_eeg[i].set_data(t, buf[i, :])
           if len(channel_names) > i:
               lines_eeg[i].set_label(channel_names[i])
           else:
               lines_eeg[i].set_label(f'Ch{i}')


       all_data_eeg = buf.flatten()
       ymin, ymax = all_data_eeg.min(), all_data_eeg.max()
       if ymin == ymax:
           ymin -= 1e-6; ymax += 1e-6
       pad = 0.1*(ymax - ymin)
       ax_eeg.set_ylim(ymin-pad, ymax+pad)
       ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0) # У EEG своя ось X
       ax_eeg.legend(loc='upper right')


   if psd_freqs_global is not None and psd_vals_global is not None:
       freqs = np.array(psd_freqs_global)
       psd_vals = np.array(psd_vals_global)
       # Убедимся, что количество каналов PSD соответствует ожидаемому
       num_psd_chs = min(psd_vals.shape[0], CHANNELS)
       for i in range(num_psd_chs):
           # Предполагаем, что psd_vals_global имеет форму (n_channels, n_freqs)
           lines_psd[i].set_data(freqs, psd_vals[i, :])
           # Возможно, стоит обновить и ярлыки для PSD, если нужно
           lines_psd[i].set_label(f'{channel_names[i] if i < len(channel_names) else f"Ch{i}"}')


       # Обновляем масштаб для PSD
       all_data_psd = psd_vals.flatten()
       ymin_psd, ymax_psd = all_data_psd.min(), all_data_psd.max()
       if ymin_psd == ymax_psd:
           ymin_psd -= 1e-12; ymax_psd += 1e-12 # Маленькое значение для масштаба
       pad_psd = 0.1*(ymax_psd - ymin_psd)
       ax_psd.set_ylim(ymin_psd-pad_psd, ymax_psd+pad_psd)
       ax_psd.set_xlim(0, 40) # У PSD своя ось X
       ax_psd.legend(loc='upper right')
   else:
       for ln in lines_psd:
           ln.set_data([], [])


   return lines_eeg + lines_psd
```


#### Регистрация обработчика данных PSD
```python
device.set_on_psd(on_psd_data)
```
---
### 4. Отрисовка альфа-мощности для всех каналов в реальном времени
---
#### Импорт инструмента для интеграции по полосе частот
```python
from eeg_utils import RingBuffer, SAMPLE_RATE, integrate_band # Добавляем integrate_band
```
#### Создание подграфиков для ЭЭГ, PSD и альфа-мощности
```python
fig, (ax_eeg, ax_psd, ax_alpha) = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
```
#### Обновление функции `update_plot` для отображения альфа-мощности
```python
def update_plot(_): 143 def update_plot(_):
   global channel_names, psd_freqs_global, psd_vals_global, alpha_history, time_history # Добавляем alpha переменные
  
   # ...


   # --- Обновление Alpha Power ---
   # Вычисляем альфа-мощность на основе текущих данных PSD
   if psd_freqs_global is not None and psd_vals_global is not None:
       freqs = np.array(psd_freqs_global)
       psd_vals = np.array(psd_vals_global)
       # Интегрируем мощность в диапазоне альфа для каждого канала
       current_alpha_powers = []
       for ch_idx in range(min(psd_vals.shape[0], CHANNELS)):
           alpha_pow_ch = integrate_band(freqs, psd_vals[ch_idx, :], ALPHA_LOW, ALPHA_HIGH)
           current_alpha_powers.append(alpha_pow_ch)


       # Добавляем текущие значения в историю
       t_now = time.time()
       time_history.append(t_now)
       for ch_idx, alpha_pow_val in enumerate(current_alpha_powers):
           alpha_history[ch_idx].append(alpha_pow_val)


       # Ограничиваем длину истории (например, последние 120 значений, как в примере)
       maxlen = 120
       for ch_idx in range(CHANNELS):
           if len(alpha_history[ch_idx]) > maxlen:
               alpha_history[ch_idx] = alpha_history[ch_idx][-maxlen:]
       if len(time_history) > maxlen:
           time_history[:] = time_history[-maxlen:]


       # Преобразуем времена в относительные (от начала отображения)
       t0 = time_history[0] if time_history else t_now
       t_rel = [(x - t0) for x in time_history]


       # Обновляем данные линий альфа-мощности
       for ch_idx, ln_alpha in enumerate(lines_alpha):
           ln_alpha.set_data(t_rel, alpha_history[ch_idx])
           ln_alpha.set_label(f'Alpha {channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"}')


       # Масштабируем ось Y для альфа-графика
       all_alpha_data = np.concatenate([alpha_history[i] for i in range(CHANNELS) if alpha_history[i]])
       if len(all_alpha_data) > 0:
           ymin_alpha, ymax_alpha = all_alpha_data.min(), all_alpha_data.max()
           if ymin_alpha == ymax_alpha:
               ymin_alpha -= 1e-12; ymax_alpha += 1e-12
           pad_alpha = 0.1 * (ymax_alpha - ymin_alpha)
           ax_alpha.set_ylim(ymin_alpha - pad_alpha, ymax_alpha + pad_alpha)


       # Масштабируем ось X для альфа-графика (используем относительное время)
       if len(t_rel) > 0:
           ax_alpha.set_xlim(t_rel[0], t_rel[-1])


       # Обновляем легенду для альфа-графика
       ax_alpha.legend(loc='upper right')
   else:
       # Если данные PSD ещё не получены, очищаем линии альфа-мощности
       for ln_alpha in lines_alpha:
           ln_alpha.set_data([], [])


   return lines_eeg + lines_psd + lines_alpha
```
---
#### Задание 4.1
Попробуйте отрисовать мощность бета-ритма вместо альфа.


---
### 5. Базовый классификатор для всех каналов (через порог)
---
#### Инициализация переменных для порога альфа-мощности и подсчёта прогресса по каналам
```python
alpha_baseline = [2e-11] * CHANNELS
accumulated_counts = [0] * CHANNELS
total_counts = [0] * CHANNELS
progress_percent = [0.0] * CHANNELS
```
#### Создание подграфиков для ЭЭГ, PSD, альфа-мощности и прогресс-бара по каналам
```python
fig, (ax_eeg, ax_psd, ax_alpha, ax_progress) = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
```
#### Создание подграфиков для ЭЭГ, PSD, альфа-мощности и прогресс-бара по каналам
```python
ax_progress.set_xlim(0, 100)
ax_progress.set_ylim(-0.5, CHANNELS - 0.5) # Настройка под количество каналов
ax_progress.set_xlabel("Progress %")
ax_progress.set_ylabel("Channel")
ax_progress.set_title("Alpha Power Threshold Crossings")
ax_progress.grid(True, axis='x')


bars = []
for i in range(CHANNELS):
   # Используем barh: (y_pos, width, height)
   bar = ax_progress.barh(i, width=progress_percent[i], height=0.8, color='green', alpha=0.7, label=f'Ch{i} Progress')
   bars.append(bar[0]) # barh возвращает список, берём первый элемент
# Добавляем сетку и лимиты для наглядности
ax_progress.set_yticks(range(CHANNELS))
ax_progress.set_yticklabels([f'Ch{i}' for i in range(CHANNELS)])
```
#### Обновление прогресса для каждого канала в функции `update_plot`
```python
if psd_freqs_global is not None and psd_vals_global is not None:
       # ... (ваш код вычисления current_alpha_powers из секции Alpha Power) ...
       # Предположим, current_alpha_powers уже вычислены как список/массив
       if 'current_alpha_powers' in locals() and len(current_alpha_powers) > 0: # Проверяем, что переменная существует и не пуста
           for ch_idx, alpha_pow_val in enumerate(current_alpha_powers):
               # Увеличиваем общий счётчик
               total_counts[ch_idx] += 1
               # Проверяем, пересекает ли текущая мощность порог
               if alpha_pow_val > alpha_baseline[ch_idx]:
                   accumulated_counts[ch_idx] += 1
               # Рассчитываем процент
               progress_percent[ch_idx] = (accumulated_counts[ch_idx] / total_counts[ch_idx]) * 100 if total_counts[ch_idx] > 0 else 0.0
               # Обновляем ширину полосы прогресса для канала ch_idx
               bars[ch_idx].set_width(progress_percent[ch_idx])


   return lines_eeg + lines_psd + lines_alpha + bars
```
---
### 6. Улучшаем отрисовку работы классификатора:


Давайте упростим прогресс-бар до одного, основанного на средней альфа-мощности всех каналов, и добавим параметры для регулировки скорости.


---
#### Определение переменных для общего прогресса и скорости изменения
```python
alpha_baseline_per_channel = [2e-11] * CHANNELS # Оставим отдельные пороги, если нужно, но используем для вычисления средней
ALPHA_BASELINE_OVERALL = 2e-11 # Порог для средней альфа-мощности
PROGRESS_PER_SUCCESS = 1.0  # Скорость увеличения
PROGRESS_PER_FAILURE = 0.5  # Скорость уменьшения (может быть 0, если не нужно уменьшать)
overall_progress_percent = 0.0
overall_total_counts = 0
```


#### Настройка осей и создание одного общего прогресс-бара
```python
ax_progress.set_ylim(-0.5, 0.5) # Только одна строка для одного бара
ax_progress.set_xlabel("Overall Progress %")
ax_progress.set_ylabel("All Channels Avg.")
ax_progress.set_title("Progress: Average Alpha Power Above Threshold")


bar = ax_progress.barh(0, width=overall_progress_percent, height=0.8, color='green', alpha=0.7)[0]
```


#### Обновление одного общего прогресса на основе средней альфа-мощности
```python
# --- Изменить блок обновления прогресса ---
if psd_freqs_global is not None and psd_vals_global is not None:
   # ... (ваш код вычисления current_alpha_powers из секции Alpha Power) ...
   if 'current_alpha_powers' in locals() and len(current_alpha_powers) > 0:
       # --- Добавить эту строку ---
       global overall_progress_percent, overall_total_counts
       # -----------------------


       overall_total_counts += 1 # Увеличиваем общий счётчик измерений


       # Вычисляем среднюю альфа-мощность по всем каналам
       mean_alpha_power = np.mean(current_alpha_powers)


       # Проверяем, пересекает ли СРЕДНЯЯ мощность общий порог
       if mean_alpha_power > ALPHA_BASELINE_OVERALL:
           overall_progress_percent += PROGRESS_PER_SUCCESS
       else:
           overall_progress_percent -= PROGRESS_PER_FAILURE


       # Ограничиваем процент от 0 до 100%
       overall_progress_percent = max(0.0, min(overall_progress_percent, 100.0))


       # Обновляем ширину ОДНОГО полосы прогресса
       bar.set_width(overall_progress_percent)


return lines_eeg + lines_psd + lines_alpha + [bar]
```
---


#### Задание 6.1


### Итоги


- Освоили визуализацию ЭЭГ и PSD в реальном времени с помощью `matplotlib`.
- Применили функции из MNE для вычисления мощности в определённых диапазонах (альфа, бета).
- Реализовали простой «классификатор» на основе порога альфа-мощности.


### Что будет на следующем занятии


- Знакомство с ESP32
- Управление машинкой путём отправки данных
- Интеграция Neiry HeadBand с ESP32


### Попробовать дома


1. Добавить отображение импеданса в графическом интерфейсе.
2. Запрограммировать считывание мощностей альфа/бета-ритмов.
3. Реализовать алгоритм для управления устройством (алгоритм выдаёт 0/1 или go / no go)
4. Изучить заранее специфику работы ESP32.
5. Выбрать оператора для управления устройством.
