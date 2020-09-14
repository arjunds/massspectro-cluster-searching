import pandas as pd
import numpy as np
import math as mt
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import re

# Six colors used for plotting different spectra
plt_colors = ["red", "blue", "green", "#FFA500", "magenta", "cyan"]

filetype = "pdf" # "png" or "pdf"

figdpi = 200 #int: DPI of the image output file

# corrects DPI of figure if using pdf
if filetype.lower() == "pdf":
    figdpi = 72

'''Graph setup below'''

#Sets x-axis bounds
min_xaxis = 50
max_xaxis = 1300

fig = plt.figure("MassSpectra Plot")

ax = fig.add_subplot()

# set x axes range
if min_xaxis is not None and max_xaxis is not None:
    ax.set_xlim([int(min_xaxis), int(max_xaxis)])


# remove top and right axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# label axes
ax.set_xlabel("Bin Lower Bounds [m/z]")
ax.set_ylabel(r"$Intensity\,[\%]$")

# set x labels
plt.xticks(rotation='75')
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end + 1, 75))

# set y labels
ax.set_ylim(0, 100)
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end + 1, 10))

# set grid
plt.grid(True, axis="y", color='black', linestyle=':', linewidth=0.1)
plt.tight_layout()

# Reads in the .csv file containing the spectra data
def readSpectraData(input_filename):
    input_data = pd.read_csv(input_filename)
    input_data = input_data.drop(input_data.columns[0], axis=1) #Trims the data so the first column isn't included
    return input_data

# Plots one spectra given the bin lower bounds and the intensities of the spectra
# Line color and Spectra Number are used for differentiating between other plots
def generate_massspectra_plot(bin_lower_bounds, intensities, line_color, spectra_number):
    spectra = "Spectra " + str(spectra_number)
    spectra_plt, = ax.plot(bin_lower_bounds, intensities, color=line_color, linewidth=.95, label=spectra)

# Takes in spectra data and the spectra to plot and plots them, ouputing the result to a file if filename is provided
def generate_multiple_massspectra_plots(spectra_data, spectra_numbers, output_filename=None):
    bin_lower_bounds = []
    # Loops through each column header in the .csv file to get the lower bound for plotting
    for column in spectra_data.columns:
        bound = re.findall(r"[-+]?\d*\.\d+|\d+", column) # Parses the float bound from the column header
        bin_lower_bounds.append(float(bound[0]))

    color_iterator = 0
    # Loops through all spectra and plots it
    for spectra in spectra_numbers:
        m_to_z = spectra_data.loc[spectra, :]
        intensities = m_to_z/np.max(m_to_z) * 100 # Normalizes the spectra values to intensities based on the max value for that spectra
        generate_massspectra_plot(bin_lower_bounds, intensities, plt_colors[color_iterator], spectra)
        color_iterator += 1 

    legend = plt.legend() # Includes the legend in the upper right that shows the color corresponding to each spectra plot

    if(output_filename != None):
        plt.savefig(output_filename + "." + filetype, dpi=fig.dpi, format=filetype)
        print("Plot saved to " + output_filename + "." + filetype)
    plt.show()


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description='Input data and specific spectra to plot')
    parser.add_argument('-data', '--data', nargs=1, type=str, metavar='', required=True, help='.csv file containing massspectra data')
    parser.add_argument('-spectra', '--spectra', nargs='+', type=int, metavar='', required=True, help='enter spectra (rows) from data file')
    parser.add_argument('-output', '--output', type=str, metavar='', help='filename (without extension) to output plot to')

    args = parser.parse_args()
    input_file = args.data[0]
    input_spectra = args.spectra
    output_filename = args.output

    # Only 6 colors so only 6 spectra can be plotted - more spectra can be plotted by adding more colors to the array at the top of file
    max_spectra_amount = len(plt_colors)
    if(len(input_spectra) > max_spectra_amount):
        print("Program can only support plotting " + str(max_spectra_amount) + " spectra at a time. Input has been trimmed to first " + str(max_spectra_amount) + " spectra entered")
        input_spectra = input_spectra[:max_spectra_amount]

    spectra_data = readSpectraData(input_file)
    generate_multiple_massspectra_plots(spectra_data, input_spectra, output_filename)
