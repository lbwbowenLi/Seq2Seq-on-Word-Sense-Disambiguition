try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import glob
import sys

dir_names = glob.glob('/home/yi/Documents/rnn/word_sense_disambigation_corpora-master/semcor/*.xml')
dir_names.sort()

for name in dir_names:
	tree = ET.ElementTree(file = name)
	roots = tree.getroot()
	with open('all_word.txt','a') as tw:
		with open('all_sense.txt','a') as ts:
			for root in roots:
				if root.attrib['text'] == '.':
					tw.write(root.attrib['text'] + '\n')
					ts.write(root.attrib['text'] + '\n')
					continue
				else:
					tw.write(root.attrib['text'] + ' ')
					if 'sense' in root.attrib:
						ts.write(root.attrib['sense'].split('m_en_us')[-1].replace('.','') + ' ')
					else:
						ts.write('1111111111' + ' ')	
