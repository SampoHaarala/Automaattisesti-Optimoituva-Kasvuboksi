import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

# CSV-tiedoston nimi
csv_file = "kasvuloki.csv"

# Aikarajat graafeihin ja animaatioon (muotoa "YYYY-MM-DD HH:MM")
START_TIME = "2025-05-19 02:30"
END_TIME = "2025-05-19 07:00"
ANIMATED_VAR = "reward"  # Muuttuja, josta tehdään animaatio
ANIMATION_INTERVAL = 300  # ms per frame
ANIMATION_FILENAME = "animated_reward.gif"

# Tarkista, että tiedosto on olemassa
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Tiedostoa '{csv_file}' ei löytynyt.")

# Ladataan CSV-tiedosto DataFrameen
df = pd.read_csv(csv_file)

# Asetetaan aikaleima indeksiksi, jos sarake on olemassa
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)  # Järjestetään aikajärjestykseen
    df.set_index("timestamp", inplace=True)

    # Poistetaan rivit, joista puuttuu dataa kiinnostavista sarakkeista
    variables = [
        "temp", "moisture", "red", "deepRed", "blue",
        "green_before", "green_after",
        "resistor_average_before", "resistor_average_after",
        "reward"
    ]
    df.dropna(subset=variables, inplace=True)

    # Poistetaan rivit, joissa aikaväli kahden mittauksen välillä ylittää 10 minuuttia
    df = df[df.index.to_series().diff().fillna(pd.Timedelta(seconds=0)) <= pd.Timedelta(minutes=10)]

    # Suodatetaan aikarajan mukaan
    start_dt = pd.to_datetime(START_TIME)
    end_dt = pd.to_datetime(END_TIME)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    # Poistetaan duplikaattipäivät: jätetään vain uusin per päivä
    df = df[~df.index.duplicated(keep='last')]

# Piirretään kuvaajat
for var in variables:
    if var in df.columns:
        plt.figure()
        plt.plot(df.index, df[var], marker='o')  # Käytetään aikaleima-akselia
        plt.title(var.capitalize())
        plt.xlabel("Aika", fontsize=8)
        plt.ylabel(var)
        plt.xticks(fontsize=6, rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plot_{var}.png")  # Tallennetaan kuva tiedostoksi
        plt.close()

# Luodaan animaatio valitusta muuttujasta
if ANIMATED_VAR in df.columns:
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    line, = ax.plot([], [], marker='o')

    ax.set_title(f"{ANIMATED_VAR.capitalize()} over Time")
    ax.set_xlabel("Aika")
    ax.set_ylabel(ANIMATED_VAR)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_ylim(df[ANIMATED_VAR].min() * 0.95, df[ANIMATED_VAR].max() * 1.05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(fontsize=6, rotation=45)
    ax.grid(True)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        x_data.append(df.index[frame])
        y_data.append(df[ANIMATED_VAR].iloc[frame])
        line.set_data(x_data, y_data)
        return line,

    ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, interval=ANIMATION_INTERVAL)
    ani.save(ANIMATION_FILENAME, writer='pillow')
    plt.close()

print("Kuvaajat ja animaatio luotu. Tarkista .png- ja .gif-tiedostot.")
