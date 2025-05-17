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
        print(f"Testataan portti: {port.device} - {port.description}")
        try:
            ser = serial.Serial(port.device, baudrate=baudrate, timeout=timeout)
            time.sleep(3)
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

ser.write(b'REQUEST_GREEN\n')
print("REQUEST_GREEN")
time.sleep(3)

ser.write(b'TURN_LIGHT_ON\n')
print("TURN_LIGHT_ON")
time.sleep(3)

ser.write(b'TURN_LIGHT_OFF\n')
print("TURN_LIGHT_OFF")
time.sleep(3)

print("Testi läpi")