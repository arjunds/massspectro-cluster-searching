import pandas as pd
import numpy as np
import math as mt
import argparse
import matplotlib.pyplot as plt
import re
import nimfa

# Six colors used for plotting different spectra
plt_colors = ["red", "blue", "green", "#FFA500", "magenta", "cyan"]
filetype = "pdf" # "png" or "pdf"
figdpi = 72 #int: DPI of the PDF output file

'''Configures a graph according to the given settingsAssumes that x_lim and y_lim are
2D vectors with an upper and lower limit such as [1,2]'''
def graphSetup(title, x_label, y_label, x_lim, y_lim):
    fig = plt.figure(title)
    ax = fig.add_subplot()

    # remove top and right axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # label axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # set x-axis
    plt.xticks(rotation='75')
    start, end = x_lim
    
    # Percentage is 10% percent of the difference between the min and max, rounded. 
    # This adds onto the end for extra space to make the GUI nice
    # It's also used to set the tick mark distance so its evenly spaced and scales based on the axis size
    percentage = round((end-start)*.05)
    percentage = percentage if percentage > 0 else 0.5
    end = end + percentage
    ax.set_xlim(start, end)
    ax.xaxis.set_ticks(np.arange(start, end, percentage))

    # set y-axis
    start, end = y_lim

    # Percentage is 10% percent of the difference between the min and max, rounded. 
    # This adds onto the end for extra space to make the GUI nicer
    # It's also used to set the tick mark distance so its evenly spaced and scales based on the axis size
    percentage = round((end-start)*.05)
    percentage = percentage if percentage > 0 else 0.5
    end = end + percentage
    ax.set_ylim(start, end)
    ax.yaxis.set_ticks(np.arange(start, end, percentage))

    # set grid
    plt.grid(True, axis="y", color='black', linestyle=':', linewidth=0.1)
    plt.tight_layout()

    #Object used for plotting
    return ax

#Saves the plot to a PDF file with the given name 
def savePlot(output_filename):
    if(output_filename != None):
        plt.savefig(output_filename + "." + filetype, dpi=figdpi, format=filetype)
        print("Plot saved to " + output_filename + "." + filetype)

#Takes a matrix X, and randomly sorts each column individually
def permuteColumns(x):
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]

# Reads in a .csv file containing spectra data
def readSpectraData(input_filename):
    input_data = pd.read_csv(input_filename)
    input_data = input_data.drop(input_data.columns[0], axis=1) #Trims the data so the first column isn't included
    return input_data

# Plots one spectra given the bin lower bounds and the intensities of the spectra
# Line color and Spectra Number are used for differentiating between other plots
def generateMassspectraPlot(bin_lower_bounds, intensities, line_color, spectra_number, ax):
    spectra = "Spectra " + str(spectra_number)
    spectra_plt = ax.bar(bin_lower_bounds, intensities, color=line_color, label=spectra)

# Takes in spectra data and the spectra #s to plot and plots them, ouputing the result to a file if filename is provided
def generateMultipleMassspectraPlots(spectra_data, spectra_numbers, output_filename=None):
    bin_lower_bounds = []
    # Loops through each column header in the .csv file to get the lower bound for plotting
    for column in spectra_data.columns:
        bound = re.findall(r"[-+]?\d*\.\d+|\d+", column) # Parses the float bound from the column header
        bin_lower_bounds.append(float(bound[0]))

    ax = graphSetup("MassSpectra Plot", "Bin Lower Bounds [m/z]", r"$Intensity\,[\%]$", [np.min(bin_lower_bounds), np.max(bin_lower_bounds)], [0,100])
    color_iterator = 0
    # Loops through all spectra and plots it
    for spectra in spectra_numbers:
        m_to_z = spectra_data.loc[spectra, :]
        intensities = m_to_z
        if(np.max(m_to_z) != 0):
            intensities = m_to_z/np.max(m_to_z) * 100 # Normalizes the spectra values to intensities based on the max value for that spectra
        generateMassspectraPlot(bin_lower_bounds, intensities, plt_colors[color_iterator], spectra, ax)
        color_iterator += 1 

    legend = plt.legend() # Includes the legend in the upper right that shows the color corresponding to each spectra plot
    savePlot(output_filename)
    plt.show()

''' Creates a NMF model of the entire spectra data and plots the basis vectors for this model
Plotted by normalizing the basis vector for each bin number to get intensity and plotting against the bin # '''
def generateSpectraPlot(spectra_data, output_filename=None):
    bin_lower_bounds = []

    # Loops through each column header in the .csv file to get the lower bound for plotting
    for column in spectra_data.columns:
        bound = re.findall(r"[-+]?\d*\.\d+|\d+", column) # Parses the float bound from the column header
        bin_lower_bounds.append(float(bound[0]))
    
    ax = graphSetup("MassSpectra NMF Basis Vector Plot", "Bin Lower Bounds [m/z]", r"$Intensity\,[\%]$", [np.min(bin_lower_bounds), np.max(bin_lower_bounds)], [0,100])
    # Convert to np array and transpose it so that the bin numbers are the rows and it's vectors of spectra intensity
    data = np.transpose(spectra_data.values)
    nmf_model = nimfa.Nmf(data)
    basis = nmf_model().basis()
    intensities = []

    for vector in basis:
        print(np.linalg.norm(vector))
        intensities.append(np.linalg.norm(vector)) # Adds the magnitude of the intensity vector to the array for graphing

    intensities = intensities/np.max(intensities) * 100
    spectra_plt = ax.bar(bin_lower_bounds, intensities)

    savePlot(output_filename)
    plt.show()

# Creates a plot that compares the rss values between the NMF model of the original data matrix and a randomly created one
def generateRssPlot(spectra_data, output_filename=None):
    data = np.transpose(spectra_data.values)
    #Reorganizes each column of the data matrix
    permutated_data = permuteColumns(data)
    
    #Range of numbers from 30-100 with interval of 2 - change this to change the k-values to test
    k = np.arange(30, 100, 2)
    rss = []

    #Loops through each k value and creates an NMF model for both the permutated data and original data
    #Adds the absolute value difference between the two rss values to an array
    for x in k:
        nmf_model = nimfa.Nmf(data, rank=x)
        nmf_model()
        data_rss = nmf_model.rss()
        permutation_model = nimfa.Nmf(permutated_data, rank=x)
        permutation_model()
        permutation_rss = permutation_model.rss()
        rss.append(abs(permutation_rss-data_rss))

    print(k)
    print(rss)

    ax = graphSetup("MassSpectra RSS Plot", "K Value (# of Basis Vectors)", "Difference in RSS Value", [np.min(k), np.max(k)], [int(np.min(rss)), mt.ceil(np.max(rss))])
   
    rss_plt = ax.plot(k, rss)

    savePlot(output_filename)
    plt.show()

# Sample Command to run:
# python3 mass_spectra_plot.py --data data/agp3k_data.csv --spectra 1 2 3 4 5 6 -output plot
if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description='Input data and specific spectra to plot')
    parser.add_argument('-data', '--data', nargs=1, type=str, metavar='', required=True, help='.csv file containing massspectra data')
    parser.add_argument('-spectra', '--spectra', nargs='+', type=int, metavar='', required=False, help='enter spectra (rows) from data file')
    parser.add_argument('-output', '--output', type=str, metavar='', help='filename (without extension) to output plot to')

    args = parser.parse_args()
    input_file = args.data[0]
    input_spectra = args.spectra
    output_filename = args.output

    # Only 6 colors so only 6 spectra can be plotted - more spectra can be plotted by adding more colors to the array at the top of file
    max_spectra_amount = len(plt_colors)
    """if(len(input_spectra) > max_spectra_amount):
        print("Program can only support plotting " + str(max_spectra_amount) + " spectra at a time. Input has been trimmed to first " + str(max_spectra_amount) + " spectra entered")
        input_spectra = input_spectra[:max_spectra_amount]
"""
    spectra_data = readSpectraData(input_file)
    generateRssPlot(spectra_data, output_filename)
#    generateSpectraPlot(spectra_data, output_filename)
#    generateMultipleMassspectraPlots(spectra_data, input_spectra, output_filename)
