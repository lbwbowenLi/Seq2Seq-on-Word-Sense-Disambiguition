import json
import sys
# with open('sem_train.json', 'r') as file:
# 	datas = file.readlines()
# 	for data in datas:
# 		d = json.loads(data)
# 		input = d['word'] + '\t' + d['sense'] + '\n'
# 		f = open('sem_train.txt','a')
# 		f.write(input)
# f.close()
# max1= 0
# max2 =0
# with open('sem_train.json', 'r') as file:
# 	datas = file.readlines()
# 	for data in datas:
# 		d = json.loads(data)
# 		max1 = max(max1, len(d['word'].split('#')))
# print(max1)
with open('sem_train_v2.txt','a') as txt:
	with open('/home/yi/Documents/rnn/semcor-json/brown1/tagfiles/br-a01.json', 'r') as file:
		dic = json.load(file)
		sentence_num = len(dic)
		for i in range(2):
			word_num = len(dic[i]['deps'])
			print(word_num)
			input1 = ''
			input2 = ''
			for j in range(word_num):
				input1 += dic[i]['deps'][j]['source'] + ' '
			for j in range(word_num):
				if 'key' in dic[i]['utt'][j]:
					input2 += dic[i]['utt'][j]['key'] + ' '
				else:
 					input2 += dic[i]['utt'][j]['lemma'] + ' '
			input_line = input1 + '\t' + input2
			print(input_line)





