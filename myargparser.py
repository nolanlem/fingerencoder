import argparse 


parser = argparse.ArgumentParser(description='autoenc arg parser')
parser.add_argument('-path', '--path', action='store', 
dest='myimgs')
results = parser.parse_args()
print 'the path is',results.myimgs
