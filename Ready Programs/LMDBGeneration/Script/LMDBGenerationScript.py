from PIL import Image
import numpy as np
import lmdb
import caffe
import sys
import argparse
import glob

def getAllFilesOfDirectory(pathtofolder):
    searchmask = pathtofolder + '/*'
    listfiles = glob.glob(searchmask)
    return listfiles

def makeListImagesAndLabels(pathtofolder):
    listimages = getAllFilesOfDirectory(pathtofolder)
    labels = []
    fimages = open('images.txt', 'w')
    length = len(listimages)
    for idx in range(0, length):
        elem = listimages[idx]
        fimages.write(str(elem))
        if 'SMILE' in elem:
            labels.append(0)
        else:
            labels.append(1)
        if (idx != length-1):
            fimages.write('\n')
    fimages.close()
    np.save('labels.npy', labels)
    return listimages, labels

def fillLmdb(images_file, labels_file, images, labels):
    images_db = lmdb.open(images_file, map_size=int(1e12), map_async=True, writemap=True)
    labels_db = lmdb.open(labels_file, map_size=int(1e12))
    images_txn = images_db.begin(write=True)
    labels_txn = labels_db.begin(write=True)
    examples = zip(images, labels)
    for in_idx, (image, label) in enumerate(examples):
        try:
            #save image
            im = Image.open(image)
            im = np.array(im) # or load whatever ndarray you need
            im = im[:,:,::-1]
            im = im.transpose((2,0,1))
            im_dat = caffe.io.array_to_datum(im)
            images_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            #save label
            label = np.array(label).astype(float).reshape(1,1,len(label))
            label_dat = caffe.io.array_to_datum(label)
            labels_txn.put('{:0>10d}'.format(in_idx), label_dat.SerializeToString())
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            print("Skipped image and label with id {0}".format(in_idx))
        if in_idx%500 == 0:
            string_ = str(in_idx+1) + ' / ' + str(len(images))
            sys.stdout.write("\r%s" % string_)
            sys.stdout.flush()
    images_txn.commit()
    labels_txn.commit()
    images_db.close()
    labels_db.close()
    print("\nFilling lmdb completed")

parser = argparse.ArgumentParser(description='Create LMDB for Caffe')
parser.add_argument('--pathToSelection', type=str, help='Path to folder with images of selection', required=True)
parser.add_argument('--imagesOut', type=str, help='Images LMDB file', required=True)
parser.add_argument('--labelsOut', type=str, help='Labels LMDB file', required=True)
args = parser.parse_args()
images, labels = makeListImagesAndLabels(args.pathToSelection)
print("Creating LMDB files")
fillLmdb(images_file='images.txt', labels_file='labels.npy', images=images, labels=labels)
print("Completed!")