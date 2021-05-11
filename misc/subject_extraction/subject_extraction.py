from trigram_tagger import SubjectTrigramTagger
from bs4 import BeautifulSoup
import requests
import re
import pickle
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
import json
import spacy

# Noun Part of Speech Tags used by NLTK
# More can be found here
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
nlp = spacy.load('en_core_web_sm')

def download_document(url):
	"""Downloads document using BeautifulSoup, extracts the subject and all
	text stored in paragraph tags
	"""
	r = requests.get(url)
	soup = BeautifulSoup(r.text, 'html.parser')
	title = soup.find('title').get_text()
	document = ' '.join([p.get_text() for p in soup.find_all('p')])
	return document

def clean_document(document):
	"""Remove enronious characters. Extra whitespace and stop words"""
	document = re.sub('[^A-Za-z .-]+', ' ', document)
	document = ' '.join(document.split())
	document = ' '.join([i for i in document.split() if i not in stop])
	return document

def tokenize_sentences(document):
	sentences = nltk.sent_tokenize(document)
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	return sentences

def get_entities(document):
	"""Returns Named Entities using NLTK Chunking"""
	entities = []
	sentences = tokenize_sentences(document)

	# Part of Speech Tagging
	sentences = [nltk.pos_tag(sent) for sent in sentences]
	for tagged_sentence in sentences:
		for chunk in nltk.ne_chunk(tagged_sentence):
			if type(chunk) == nltk.tree.Tree:
				entities.append(' '.join([c[0] for c in chunk]).lower())
	return entities

def word_freq_dist(document):
	"""Returns a word count frequency distribution"""
	words = nltk.tokenize.word_tokenize(document)
	words = [word.lower() for word in words if word not in stop]
	fdist = nltk.FreqDist(words)
	return fdist


def extract_subject(document, nlp):
	# Get most frequent Nouns
	tmp = nlp(document)
	sub_toks = [tok for tok in tmp if tok.dep_=='nsubj']
	if len(sub_toks) > 0:
		sub_freq = {}
		
		for tok in sub_toks:
			sub_freq[tok.text] = sub_freq.get(tok.text, 0) + 1
		sorted_sub = sorted(sub_freq.items(), key=lambda value: value[1], reverse=True)

		selected_sub = [x[0] for x in sorted_sub[0:2]]
		
		return selected_sub[0]
	else:
		subj = [tok for tok in tmp if tok.tag_ == 'NN']
		if len(subj) > 0:
			return subj[0].text
		else:
			return None


	'''	
	fdist = word_freq_dist(document)
	most_freq_nouns = [w for w, c in fdist.most_common(10)
					   if nltk.pos_tag([w])[0][1] in NOUNS]

	
	# Get Top 10 entities
	entities = get_entities(document)
	
	top_10_entities = [w for w, c in nltk.FreqDist(entities).most_common(10)]

	# Get the subject noun by looking at the intersection of top 10 entities
	# and most frequent nouns. It takes the first element in the list
	subject_nouns = [entity for entity in top_10_entities
					if entity.split()[0] in most_freq_nouns]
	print(subject_nouns)
	
	return subject_nouns[0]
	'''

def trained_tagger(existing=False):
	"""Returns a trained trigram tagger

	existing : set to True if already trained tagger has been pickled
	"""
	if existing:
		trigram_tagger = pickle.load(open('./subject_extraction/trained_tagger.pkl', 'rb'))
		return trigram_tagger

	# Aggregate trained sentences for N-Gram Taggers
	train_sents = nltk.corpus.brown.tagged_sents()
	train_sents += nltk.corpus.conll2000.tagged_sents()
	train_sents += nltk.corpus.treebank.tagged_sents()

	# Create instance of SubjectTrigramTagger and persist instance of it
	trigram_tagger = SubjectTrigramTagger(train_sents)
	pickle.dump(trigram_tagger, open('./subject_extraction/trained_tagger.pkl', 'wb'))

	return trigram_tagger

def tag_sentences(subject, cleaned_document):
	"""Returns tagged sentences using POS tagging"""
	trigram_tagger = trained_tagger(existing=True)

	# Tokenize Sentences and words
	sent = tokenize_sentences(cleaned_document)
	#merge_multi_word_subject(sentences, subject)

	# Tag each sentence
	tagged_sent = trigram_tagger.tag(sent[0])
	return tagged_sent

def merge_multi_word_subject(sentences, subject):
	"""Merges multi word subjects into one single token
	ex. [('steve', 'NN', ('jobs', 'NN')] -> [('steve jobs', 'NN')]
	"""
	if len(subject.split()) == 1:
		return sentences
	subject_lst = subject.split()
	sentences_lower = [[word.lower() for word in sentence]
						for sentence in sentences]
	for i, sent in enumerate(sentences_lower):
		if subject_lst[0] in sent:
			for j, token in enumerate(sent):
				start = subject_lst[0] == token
				exists = subject_lst == sent[j:j+len(subject_lst)]
				if start and exists:
					del sentences[i][j+1:j+len(subject_lst)]
					sentences[i][j] = subject
	return sentences

def get_svo(sentence, subject, ori_sent):
	"""Returns a dictionary containing:

	subject : the subject determined earlier
	action : the action verb of particular related to the subject
	object : the object the action is referring to
	phrase : list of token, tag pairs for that lie within the indexes of
				the variables above
	"""
	subject_idx = next((i for i, v in enumerate(sentence)
					if v[0].lower() in subject), None)
	data = {'subject': sentence[subject_idx][0]}
	for i in range(subject_idx, len(sentence)):
		found_action = False
		for j, (token, tag) in enumerate(sentence[i+1:]):
			if tag in VERBS:
				data['action'] = token
				found_action = True
			if tag in NOUNS and found_action == True:
				data['object'] = token
				return data # <subj, verb, obj>
	if found_action:
		data['object'] = '<EOS>'
		return data	# <subj, verb, eos>

	#else, using spacy
	tmp = nlp(ori_sent)
	found_action = False
	for word in tmp:
		if word.dep_ == 'ROOT' and word.tag_ in VERBS:
			data['action'] = word.text
			found_action = True 
		if word.dep_ == 'dobj' and word.tag_ in NOUNS and found_action:
			data['object'] = word.text
			return data # <subj, verb, obj>
	if found_action:
		data['object'] = '<EOS>'
		return data
	
	return {}

if __name__ == '__main__':
	'''	
	url = 'http://www.nytimes.com/2016/06/13/us/politics/bernie-sanders-campaign.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news'
	document = download_document(url)
	# document = pickle.load(open('document.pkl', 'rb'))
	print(document)
	document = clean_document(document)
	subject = extract_subject(document)
	tagged_sents = tag_sentences(subject, document)
	svos = [get_svo(sentence, subject)
						for sentence in tagged_sents]
	for svo in svos:
		if svo:
			print(svo)
	'''
	caption_file = 'msvd_anno/sents_all.json'
	with open(caption_file, 'r') as fd:
		captions = json.load(fd)

	cap_svo_all = {}
	cnt = 0
	for key, val in captions.items():
		print(cnt)
		cnt += 1
		document = '. '.join(val)
		cleaned_document = clean_document(document)
		subject = extract_subject(document, nlp)

		tagged_sents, ori_sents = tag_sentences(subject, cleaned_document, val)

		keep_sents = []
		keep_svos = []
		for ori_sent, tag_sent in zip(ori_sents, tagged_sents):
			svo = get_svo(tag_sent, subject, ori_sent)
			if svo:
				keep_sents.append(ori_sent)
				keep_svos.append(svo)
		cap_svo_all[key] = {}
		cap_svo_all[key]['cap'] = keep_sents
		cap_svo_all[key]['svo'] = keep_svos

		print('{} \t {}'.format(len(val), len(keep_svos)))
		print('{} \n {}'.format(keep_sents, keep_svos))

	cap_svo_file = 'msvd_anno/cap_svo_all.pkl'
	with open(cap_svo_file, 'wb') as fd:
		pickle.dump(cap_svo_all, fd)
