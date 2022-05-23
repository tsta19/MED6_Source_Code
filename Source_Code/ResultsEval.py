import os
import numpy as np
from scipy import stats as st

def getDistribution(data):
    # The three-sigma-rule
    # 68.27% of the data should be within 1*stdev of the mean
    withinMean = []
    # 95.45% of the data should be withing 2*stdev of the mean
    withinTwoMean = []
    # 99.73 of the data should be within 3*stdev of the mean
    withinThreeMean = []

    mean = np.mean(data)
    stdev = np.std(data)
    for i in range(len(data)):
        if data[i] > (mean - stdev) and data[i] < (mean + stdev):
            withinMean.append(data[i])

        if data[i] > (mean - 2*stdev) and data[i] < (mean + 2*stdev):
            withinTwoMean.append(data[i])

        if data[i] > (mean - 3*stdev) and data[i] < (mean + 3*stdev):
            withinThreeMean.append(data[i])

    prcntWithinMean = round((len(withinMean)/len(data)) * 100, 3)
    prcntWithinTwoMean = round((len(withinTwoMean)/len(data)) * 100, 3)
    prcntWithinThreeMean = round((len(withinThreeMean)/len(data)) * 100)
    print(f'Percentage of data within 1*stdev of mean: {prcntWithinMean}')
    print(f'Percentage of data within 2*stdev of mean: {prcntWithinTwoMean}')
    print(f'Percentage of data within 3*stdev of mean: {prcntWithinThreeMean}')
    dist, p = st.shapiro(data)
    print(f'Distribution: {dist}')
    print(f'p-value: {p}')
    return dist

def ttest(data1, data2):

    pass

if __name__ == "__main__":
    path = 'ResultsEVAL'
    directory = os.listdir('ResultsEVAL')
    print(directory)

    landSwapFile = open(path + '//' + directory[0], 'r')
    landSwapRes = landSwapFile.read()
    landSwapRes = [abs(float(i)) for i in landSwapRes.split()]

    landOldFile = open(path + '//' + directory[1], 'r')
    landOldRes = landOldFile.read()
    landOldRes = [abs(float(i)) for i in landOldRes.split()]

    landNewFile = open(path + '//' + directory[2], 'r')
    landNewRes = landNewFile.read()
    landNewRes = [abs(float(i)) for i in landNewRes.split()]

    portNewFile = open(path + '//' + directory[5], 'r')
    portNewRes = portNewFile.read()
    portNewRes = [abs(float(i)) for i in portNewRes.split()]

    portSwapFile = open(path + '//' + directory[3], 'r')
    portSwapRes = portSwapFile.read()
    portSwapRes = [abs(float(i)) for i in portSwapRes.split()]

    portOldFile = open(path + '//' + directory[4], 'r')
    portOldRes = portOldFile.read()
    portOldRes = [abs(float(i)) for i in portOldRes.split()]

    stDevLandSwap = np.std(landSwapRes)
    stDevLandOld = np.std(landOldRes)
    stDevLandNew = np.std(landNewRes)

    stDevPortNew = np.std(portNewRes)
    stDevPortOld = np.std(portOldRes)
    stDevPortSwap = np.std(portSwapRes)

    print('--- Mean Absolute Difference ---')
    print(f'Landscape Swapped: {np.mean(landSwapRes)}')
    print(f'Landscape Old: {np.mean(landOldRes)}')
    print(f'Landscape New: {np.mean(landNewRes)}')
    print('---------------------------')
    print(f'Portraits Swapped: {np.mean(portSwapRes)}')
    print(f'Portraits Old: {np.mean(portOldRes)}')
    print(f'Portraits New: {np.mean(portNewRes)}')
    print('---------------------------')
    print('--- Standard Deviation ---')
    print(f'Landscape Swapped: {stDevLandSwap}')
    print(f'Landscape Old: {stDevLandOld}')
    print(f'Landscape New: {stDevLandNew}')
    print('---------------------------')
    print(f'Portrait Swapped: {stDevPortSwap}')
    print(f'Portrait Old: {stDevPortOld}')
    print(f'Portrait New: {stDevPortNew}')
    print('---------------------------')
    print('----- LandSwap -----')
    getDistribution(landSwapRes)
    print(f'Levene PORT: {st.levene(portNewRes, portOldRes)}')
    print(f'Levene LAND: {st.levene(landNewRes, landOldRes)}')
    print('----- LandOld -----')
    getDistribution(landOldRes)
    print('----- LandNew -----')
    getDistribution(landNewRes)
    print('----- PortSwap -----')
    getDistribution(portSwapRes)
    print('----- PortOld -----')
    getDistribution(portOldRes)
    print('----- PortNew -----')
    getDistribution(portNewRes)
    print(f"Is new vs old signicantly different?: WILCOXON")
    print("len ", len(portNewRes))
    print("len ", len(portOldRes))
    portOldRes = portOldRes[0:len(portNewRes)]
    statPort, pPort = st.wilcoxon(portNewRes, portOldRes)
    statLand, pLand = st.wilcoxon(landNewRes, landOldRes)
    print(f'Portrait old vs new:  stat: {statPort}, p: {pPort}')
    print(f'Landscape old vs new: stat: {statLand}, p: {pLand}')

    print(f"Is new vs old signicantly different?: MANNWHITENY")
    print("len ", len(portNewRes))
    print("len ", len(portOldRes))
    portOldRes = portOldRes[0:len(portNewRes)]
    statPort, pPort = st.mannwhitneyu(portNewRes, portOldRes)
    statLand, pLand = st.mannwhitneyu(landNewRes, landOldRes)
    print(f'Portrait old vs new:  stat: {statPort}, p: {pPort}')
    print(f'Landscape old vs new: stat: {statLand}, p: {pLand}')