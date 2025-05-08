import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

merged_df = pd.read_csv("merged_nypd_data.csv", parse_dates=['DATE'], low_memory=False)

races = [
    'AMERICAN INDIAN/\nALASKAN NATIVE',
    'ASIAN /\nPACIFIC ISLANDER',
    'BLACK',
    'BLACK HISPANIC',
    'WHITE',
    'WHITE HISPANIC',
    'UNKNOWN'
]

arrests = merged_df.dropna(subset=['PERP_RACE_x']).groupby('PERP_RACE_x').size().reindex(races).fillna(0)
shootings = merged_df.dropna(subset=['PERP_RACE_y']).groupby('PERP_RACE_y').size().reindex(races).fillna(0)

x = np.arange(len(races))

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_facecolor('#f0f8ff')
ax.grid(color='#add8e6', linestyle='-', linewidth=0.7)

ax.plot(x, arrests.values, color='#ff8c00', linewidth=2, marker='x', markersize=8, markeredgewidth=2, label='Arrests')
ax.plot(x, shootings.values, color='#006400', linewidth=2, marker='x', markersize=8, markeredgewidth=2, label='Shootings')

ax.annotate('', xy=(1.02, 0), xytext=(0, 0), xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', linewidth=1.5))
ax.annotate('', xy=(0, 1.02), xytext=(0, 0), xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', linewidth=1.5))

d = .015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (0.25-d, 0.25+d), **kwargs)
ax.plot((-d, +d), (1-d, 1+d), **kwargs)

ax.set_xticks(x)
ax.set_xticklabels(races, rotation=30, ha='right', fontsize=6)
ax.set_xlabel('Race', labelpad=10)
ax.set_ylabel('Number of Incidents', labelpad=10)
ax.set_title('Incidents by Race (Arrests vs. Shootings)', pad=5)

for side in ['top','right']:
    ax.spines[side].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.legend(frameon=False)

plt.subplots_adjust(left=0.1, bottom=0.25)

plt.show()
