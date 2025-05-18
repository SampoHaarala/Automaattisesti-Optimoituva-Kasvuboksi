import serial              # Arduinon kanssa kommunikointiin
import numpy as np         # Lineaarialgebra LinUCB:lle
import pandas as pd        # (valinnainen) lokitukseen ja datan seurantaan
import time                # Ajan mittaamiseen
import serial
import serial.tools.list_ports
import LinUCB
import os

# 🔍 Automattinen porttihaku
def find_arduino_port(baudrate=9600, timeout=15):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if "bluetooth" in desc or "wireless" in desc:
            print(f"Ohitetaan Bluetooth-portti: {port.device} - {port.description}")
            continue

        print(f"Testataan portti: {port.device} - {port.description}")
        try:
            ser = serial.Serial(port.device, baudrate=baudrate, timeout=timeout)
            time.sleep(2)
            ser.reset_input_buffer()

            ser.write(b'PING\n')

            # 🔁 Odota enintään 3 sekuntia vastausta
            start = time.time()
            response = ""
            while time.time() - start < 3:
                if ser.in_waiting:
                    response = ser.readline().decode('utf-8').strip()
                    break
                time.sleep(0.1)

            if response:
                print(f"Vastaus: {response}")
                if "ARDUINO" in response.upper():
                    print(f"Yhdistetty porttiin {port.device}")
                    return ser
                else:
                    print("Väärä vastaus, ei ARDUINO.")
            else:
                print("Ei vastausta.")

        except Exception:
            continue
    raise RuntimeError("Arduinoa ei löytynyt yhdistetyistä porteista.")

def load_training_data(filename="kasvuloki.csv"):
    try:
        df = pd.read_csv(filename)

        X_log = df[["temp", "moisture", "red", "deepRed", "blue", "green_before", "resistor_average_before"]].to_numpy()
        A_log = df["action"].to_numpy().astype(int)
        R_log = df["reward"].to_numpy().astype(float)

        print(f"Loaded {len(X_log)} rows from {filename}.")
        return X_log, A_log, R_log

    except FileNotFoundError:
        print("Log file not found. Proceeding with manual data only.")
        return np.empty((0, 7)), np.array([], dtype=int), np.array([], dtype=float)

# Luo tyhjä lokidataframe
log_file = "kasvuloki.csv"

if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    print(f"Loaded existing log with {len(log_df)} rows.")
else:
    log_df = pd.DataFrame(columns=["timestamp", "temp", "moisture", "red", "deepRed",
                                   "blue", "green_before", "resistor_average_before", "action", 
                                   "green_after", "resistor_average_after", "reward"])
    
# 📌 Initialisoi LinUCB
# 3 toimintoa: 0 = DO_NOTHING, 1 = ADD_WATER, 2 = TURN_LIGHT_OFF
n_actions = 3
model = LinUCB.LinUCB(n_arms=4, n_features=7, alpha=1.0)  # alpha = tutkimus vs hyödyntäminen

# 🌡️ Simuloitu opetusdata (ensimmäinen kylmä alustus)
# Voit korvata tämän aidolla sensoridatalla ajan myötä
X_init = np.array([
    # temp, hum, red, deepRed, blue, green, ldr
    [24, 40, 180, 250, 70, 280, 130],   # dry, dim → ADD_WATER
    [25, 42, 190, 260, 75, 290, 120],
    [26, 45, 195, 270, 80, 300, 140],
    
    [27, 65, 210, 280, 90, 320, 180],   # too bright → TURN_LIGHT_OFF
    [28, 70, 220, 290, 95, 325, 190],
    [26, 60, 210, 275, 90, 310, 180],

    [24, 55, 170, 230, 65, 270, 80],    # dim + low green → TURN_LIGHT_ON
    [23, 60, 160, 220, 60, 260, 70],
    [22, 58, 155, 210, 55, 255, 60],

    [25, 65, 200, 270, 85, 310, 130],   # balanced → DO_NOTHING
    [26, 62, 205, 275, 88, 315, 120],
    [27, 68, 210, 280, 90, 320, 125],

    [25, 48, 185, 255, 72, 295, 140],   # dry side → ADD_WATER
    [29, 72, 225, 300, 100, 330, 185],  # hot + bright → TURN_LIGHT_OFF
    [28, 69, 195, 265, 85, 280, 90],    # humid, low green → DO_NOTHING

    [23, 63, 160, 225, 65, 265, 60],    # dim → TURN_LIGHT_ON
    [24, 50, 180, 245, 70, 285, 110],   # dry/moderate → ADD_WATER
    [26, 55, 190, 260, 75, 275, 100],   # balanced → DO_NOTHING
    [27, 58, 200, 270, 82, 280, 170],   # bright → TURN_LIGHT_OFF
    [25, 66, 205, 275, 85, 305, 130],   # optimal → DO_NOTHING
])


# Vastaavat toimenpiteet
A_init = np.array([
    1, 1, 1,      # dry → ADD_WATER
    2, 2, 2,      # bright → TURN_LIGHT_OFF
    3, 3, 3,      # dim + low green → TURN_LIGHT_ON
    0, 0, 0,      # balanced → DO_NOTHING
    1, 2, 0,      # moderate + dry/bright → mixed
    3, 1, 0, 2, 0 # context-based
])


 # Simuloitu reward (vihreän valon ∆)
R_init = np.array([
    0.42, 0.40, 0.38,   # watering helped
    0.34, 0.33, 0.31,   # turning light off lowered stress
    0.36, 0.38, 0.37,   # turning light on improved photosynthesis
    0.29, 0.30, 0.31,   # doing nothing maintained balance
    0.35, 0.32, 0.27,   # moderate benefit
    0.36, 0.39, 0.28, 0.33, 0.30
])


# Load past log and combine
X_log, A_log, R_log = load_training_data()

X_combined = np.vstack((X_init, X_log))
A_combined = np.concatenate((A_init, A_log))
R_combined = np.concatenate((R_init, R_log))

# Train model
for x, a, r in zip(X_combined, A_combined, R_combined):
    model.update(a, x, r)

# 📟 Yhdistä Arduinoon
try:
    ser = find_arduino_port()
except RuntimeError as e:
    print(e)
    exit(1)  # Pysäyttää ohjelman

# Valo päälle simulaation alkuun.
ser.write(b'TURN_LIGHT_ON\n')

# 🔁 Pääsilmukka
loopNumber = 0
while True:
    print(f"Loop number: {loopNumber}")
    ser.reset_input_buffer()
    ser.write(b'REQUEST_DATA\n')
    time.sleep(1)
    if ser.in_waiting:
        try:
            # 📥 Sensoridata muotoa: "25.3,61.0,302,512"
            # (temp, moisture, AS7341_vihrea, resistorAverage)
            line = ser.readline().decode('utf-8').strip()
            temp, moist, red, deepRed, blue, green, resistorAverage = map(float, line.split(','))
            ser.write(b'REQUEST_GREEN\n')
            time.sleep(1)
            line = ser.readline().decode('utf-8').strip()
            green, resistorAverage = map(float, line.split(','))
            context = np.array([[temp, moist, red, deepRed, blue, green, resistorAverage]])

            action = model.select_arm(context.flatten())
            print(f"Taking action: {action}")

            # Lähetä toimenpide Arduinolle
            if action == 1:
                ser.write(b'ADD_WATER\n')
            elif action == 2:
                ser.write(b'TURN_LIGHT_OFF\n')
            elif action == 3:
                ser.write(b'TURN_LIGHT_ON\n')
            else:
                ser.write(b'DO_NOTHING\n')

            # ⏳ Odotetaan vaste
            time.sleep(300)

            # 📥 Uusi mittaus: sama formaatti
            ser.write(b'REQUEST_GREEN\n')
            time.sleep(1)
            line2 = ser.readline().decode('utf-8').strip()
            green2, resistorAverage2 = list(map(float, line2.split(',')))

            # 🎯 Lasketaan yhdistetty reward (painotettu)
            reward = 0.9 * (green2 - green) + 0.2 * (resistorAverage2 - resistorAverage)

            # Päivitä malli
            model.update(action, context.flatten(), reward)

            # 📊 Lisää lokiin
            log_df.loc[len(log_df)] = {
                "timestamp": pd.Timestamp.now(),
                "temp": temp,
                "moisture": moist,
                "red": red,
                "deepRed": deepRed,
                "blue": blue,
                "green_before": green,
                "resistor_average_before": resistorAverage,
                "action": action,
                "green_after": green2,
                "resistor_average_after": resistorAverage2,
                "reward": reward
            }

            # 💾 Tallenna CSV:hen jokaisen kierroksen jälkeen (varmuuden vuoksi)
            log_df.to_csv("kasvuloki.csv", mode='a', header=not os.path.exists("kasvuloki.csv"), index=False)

            # Ilmoita prosessin etenemisestä.
            print(f"ACTION: {action}, REWARD: {reward:.3f}")
            loopNumber += 1

        except Exception as e:
            print("Virhe:", e)

    time.sleep(1)
