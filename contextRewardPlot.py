import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 🎯 Aseta graafin alku- ja loppuaika (tai jätä None jos haluat koko datasetin)
GRAPH_START = "2025-05-18 13:00:00"  # tai esim. "2025-05-18"
GRAPH_END   = "2025-05-19 00:00:00"  # tai None

# Muunna datetime-objekteiksi jos määritetty
start_time = pd.to_datetime(GRAPH_START) if GRAPH_START else None
end_time = pd.to_datetime(GRAPH_END) if GRAPH_END else None

# CSV-tiedoston nimi
csv_file = "kasvuloki.csv"

# Tarkista, onko tiedosto olemassa
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Tiedostoa '{csv_file}' ei löytynyt.")

# Ladataan tiedosto DataFrameen
df = pd.read_csv(csv_file)

# Asetetaan aikaleima indeksiksi, jos sarake on olemassa
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

if start_time:
    df = df[df.index >= start_time]
if end_time:
    df = df[df.index <= end_time]

# Valitaan kontekstidatat ja reward
context_vars = ["temp", "moisture", "red", "deepRed", "blue", "green_before", "resistor_average_before"]
reward_var = "reward"
all_vars = context_vars + [reward_var]

# Poistetaan puuttuvat arvot
df.dropna(subset=all_vars, inplace=True)

# Piirretään kaikki yhdessä kuvaajassa
plt.figure(figsize=(12, 6))
for var in all_vars:
    plt.plot(df.index, df[var], label=var)

plt.title("Context Variables and Reward Over Time")
plt.xlabel("Aika")
plt.ylabel("Arvo")
plt.xticks(rotation=45, fontsize=8)
plt.legend(loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_context_and_reward_combined.png")
plt.close()

print("Yhdistetty kuvaaja luotu. Tarkista 'plot_context_and_reward_combined.png'.")