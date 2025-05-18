import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

# CSV-tiedoston nimi
csv_file = "kasvuloki.csv"

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

    # Poistetaan rivit ennen klo 13.00
    df = df[df.index.time >= pd.to_datetime("13:00").time()]

    # Poistetaan duplikaattipäivät: jätetään vain uusin per päivä
    df = df[~df.index.duplicated(keep='last')]

# Piirretään kuvaajat
for var in variables:
    if var in df.columns:
        plt.figure()
        plt.plot(df.index, df[var], marker='o')  # Käytetään aikaleima-akselia
        plt.title(var.capitalize())
        plt.xlabel("Kellonaika", fontsize=8)
        plt.ylabel(var)
        plt.xticks(fontsize=6, rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plot_{var}.png")  # Tallennetaan kuva tiedostoksi
        plt.close()

print("Kuvaajat luotu tiedostoista. Tarkista .png-tiedostot.")