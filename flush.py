import time 
import sys
i=0 

while(True):
	sys.stdout.write('%r \r'%(i)) 
	sys.stdout.flush()
	i += 1
	time.sleep(0.01)
	
