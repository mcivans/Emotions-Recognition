import os
import sys
import glob

def getAllFilesOfDirectory(pathtofolder):
    searchmask = pathtofolder + '/*'
    listfiles = glob.glob(searchmask)
    return listfiles

def estimateConfusion(pathtofolder, emotionscatter):
    listfiles = getAllFilesOfDirectory(pathtofolder)
    for filename in listfiles:
        if 'AMAZED' in filename:
            emotionscatter['AMAZED'] += 1
        elif 'LOOKING' in filename:
            emotionscatter['LOOKING'] += 1
        elif 'DISGUISTING' in filename:
            emotionscatter['DISGUISTING'] += 1
        elif 'YAWNING' in filename:
            emotionscatter['YAWNING'] += 1
        elif 'CALM' in filename:
            emotionscatter['CALM'] += 1
        elif 'SMILE' in filename:
            emotionscatter['SMILE'] += 1
        else: # for not marked files, files without emotions or for directories
            emotionscatter['Other'] += 1

def makeCSVSelection(emotionscatterNS, emotionscatterS):
    f = open('confusion_matrix.csv', 'w')
    f.write(';Smile;NotSmile;All\n') # cells in csv-file of database are selectes by semis
    total = 0
    totalS = 0
    totalNS = 0
    for key in emotionscatterNS.keys():
        countNS = emotionscatterNS[key]
        countS = emotionscatterS[key]
        totalNS += countNS
        totalS += countS
        count = countNS + countS
        total += count
        if (countNS > 0 or countS > 0): # in file it's printed only cases with at least 1 appropriated file
            f.write(str(key) + ';' + str(countS) + ';' + str(countNS) + ';' + str(count) + '\n')
    f.write('Total;' + str(totalS) + ';' + str(totalNS) + ';' + str(total))
    f.close()

def makeCSV(emotionscatter):
    f = open('confusion_matrix.csv', 'w')
    f.write(';All\n')
    total = 0
    for key in emotionscatter.keys():
        count = emotionscatter[key]
        total += count
        if (count > 0):
            f.write(str(key)+';'+str(count)+'\n')
    f.write('Total;'+str(total))
    f.close()

print('Please, enter the mode:')
print('     1 - for classified selection')
print('     2 - for any directory')
mode = input('mode = ')
if not str(mode).isnumeric(): # validation for number of mode
    print('Wrong mode!')
    sys.exit(0)
mode = int(mode) % 2
path = input('Please, enter the path to folder: ')
if not os.path.isdir(path): # check the rightness of path to folder
    print('Directory doesn\'t exist. Please, check the path to folder!')
    sys.exit(0)
if mode: # for classified selection
    emotionscatterNS = {'Other': 0, 'AMAZED': 0, 'LOOKING': 0, 'DISGUISTING': 0, 'YAWNING': 0, 'CALM': 0, 'SMILE': 0}
    emotionscatterS = {'Other': 0, 'AMAZED': 0, 'LOOKING': 0, 'DISGUISTING': 0, 'YAWNING': 0, 'CALM': 0, 'SMILE': 0}
    pathtofolderNS = path + '/NotSmile'
    pathtofolderS = path + '/Smile'
    if not os.path.isdir(pathtofolderNS) or not os.path.isdir(pathtofolderS): # check the structure of selection folder
        print('This directory isn\'t Selection!')
        sys.exit(0)
    estimateConfusion(pathtofolderNS, emotionscatterNS)
    estimateConfusion(pathtofolderS, emotionscatterS)
    makeCSVSelection(emotionscatterNS, emotionscatterS)
else: # for any direction
    emotionscatter = {'AMAZED': 0, 'LOOKING': 0, 'DISGUISTING': 0, 'YAWNING': 0, 'CALM': 0, 'SMILE': 0, 'Other': 0}
    estimateConfusion(path, emotionscatter)
    makeCSV(emotionscatter)