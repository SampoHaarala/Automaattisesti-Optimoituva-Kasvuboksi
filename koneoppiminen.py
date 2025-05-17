import serial              # Arduinon kanssa kommunikointiin
import numpy as np         # Lineaarialgebra LinUCB:lle
import pandas as pd        # (valinnainen) lokitukseen ja datan seurantaan
import time                # Ajan mittaamiseen
import serial
import serial.tools.list_ports
import LinUCB

# 🔍 Automattinen porttihaku
def find_arduino_port(baudrate=9600, timeout=15):
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        try:
            ser = serial.Serial(port.device, baudrate=baudrate, timeout=timeout)
            time.sleep(2)
            ser.write(b'PING\n')  # Vapaaehtoinen testikommunikaatio
            if ser.in_waiting:
                response = ser.readline().decode('utf-8').strip()
                if "ARDUINO" in response.upper():
                    print(f"Yhdistetty: {port.device}")
                    return ser
            else:
                return ser  # Palauta, vaikka ei vastaa testiin
        except Exception:
            continue
    raise RuntimeError("Arduinoa ei löytynyt yhdistetyistä porteista.")

# Luo tyhjä lokidataframe
log_df = pd.DataFrame(columns=["timestamp", "temp", "moisture", "green_before", "action", "green_after", "reward"])

# 📌 Initialisoi LinUCB
# 3 toimintoa: 0 = DO_NOTHING, 1 = ADD_WATER, 2 = TURN_LIGHT_OFF
n_actions = 3
model = LinUCB.LinUCB(n_arms=4, n_features=4, alpha=1.0)  # alpha = tutkimus vs hyödyntäminen

# 🌡️ Simuloitu opetusdata (ensimmäinen kylmä alustus)
# Voit korvata tämän aidolla sensoridatalla ajan myötä
X_init = np.array([
    [25, 60, 300, 300],   # [lämpötila °C, ilmankosteus %, vihreän valon arvo, LDR keskiarvo]
    [26, 55, 290, 100],
    [24, 70, 310, 100],
    [27, 50, 280, 120]
])
A_init = np.array([1, 2, 0, 3])  # Manuaaliset toimenpiteet
R_init = np.array([0.3, 0.1, 0.4, 0.2])  # Simuloitu reward (vihreän valon ∆)

# Mallin opetus
for x, a, r in zip(X_init, A_init, R_init):
    model.update(a, x, r)

# 📟 Yhdistä Arduinoon
try:
    ser = find_arduino_port()
except RuntimeError as e:
    print(e)
    exit(1)  # Pysäyttää ohjelman

# 🔁 Pääsilmukka
while True:
    if ser.in_waiting:
        try:
            # 📥 Sensoridata muotoa: "25.3,61.0,302,512"
            # (temp, moisture, AS7341_vihrea, resistorAverage)
            line = ser.readline().decode('utf-8').strip()
            temp, moist, green, resistorAverage = map(float, line.split(','))
            context = np.array([[temp, moist, green, resistorAverage]])

            action = model.select_arm(context.flatten())

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
            time.sleep(600)

            # 📥 Uusi mittaus: sama formaatti
            ser.write(b'REQUEST_GREEN\n')
            line2 = ser.readline().decode('utf-8').strip()
            green2, resistorAverage2 = list(map(float, line2.split(',')))

            # 🎯 Lasketaan yhdistetty reward (painotettu)
            reward = 0.7 * (green2 - green) + 0.3 * (resistorAverage2 - resistorAverage)

            # Päivitä malli
            model.update(action, context.flatten(), reward)

            # 📊 Lisää lokiin
            log_df.loc[len(log_df)] = {
                "timestamp": pd.Timestamp.now(),
                "pre_temp": temp,
                "pre_moisture": moist,
                "pre_green": green,
                "pre_resistor_average": resistorAverage,
                "action": action,
                "post_green": green2,
                "post_resistor_average": resistorAverage2,
                "reward": reward
            }

            # 💾 Tallenna CSV:hen jokaisen kierroksen jälkeen (varmuuden vuoksi)
            log_df.to_csv("kasvuloki.csv", index=False)

            # Ilmoita prosessin etenemisestä.
            print(f"ACTION: {action}, REWARD: {reward:.3f}")

        except Exception as e:
            print("Virhe:", e)

    time.sleep(1)
