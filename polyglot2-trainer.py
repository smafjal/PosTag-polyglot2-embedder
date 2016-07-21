#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Embeddings trainer."""

from argparse import ArgumentParser
import logging
import sys
import os
from io import open
from os import path
from time import time
from glob import glob


LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

from polyglot2.polyglot2 import Polyglot
import numpy as np
from random import shuffle

class LineSentence(object):
	def __init__(self, source):
		self.sources = glob(source)
		shuffle(self.sources)
	def __iter__(self):
		try:
			self.source.seek(0)
			for line in self.source:
				lines_list=line.split()
				lines_list.insert(0,"<PAD>")
				lines_list.insert(len(lines_list),"</PAD>")
				yield lines_list

		except AttributeError:
			for source in self.sources:
				for line in open(source):
					l = line.strip()
					if not l: continue
					if l.startswith('[['): continue
					if l.endswith('[['): continue
					lines_list=l.split()
					lines_list.insert(0,"<PAD>")
					lines_list.insert(len(lines_list),"</PAD>")
					yield lines_list

def main(args,path):
  	sentences = LineSentence(args.files)
  	cnt=1
  	for x in sentences:
  		print x
  		cnt=cnt+1
  		if cnt>5:
  			break

  	# sentences=MySentences(path)
  	model = Polyglot(sentences=sentences, vocab_file=args.vocab, size=args.size, alpha=args.alpha,hidden=args.hidden, window=args.window, batch_size=args.batch_size,
          min_count=args.min_count, workers=args.workers)
  	model.save_word2vec_format(fname=args.output+'.model')

if __name__ == "__main__":
	data_path="corpus"
	parser = ArgumentParser()
	parser.add_argument("--files", dest="files", help="Corpus file[s]")
	parser.add_argument("--vocab", dest="vocab", help="Vocab that is list of"
	          " words with the their frequencies")
	parser.add_argument("--output", dest="output", help="Model text file output")
	parser.add_argument("--size", dest="size", help="Embedding size", type=int,
	          default=100)
	parser.add_argument("--alpha", dest="alpha", help="Initial learning rate", type=float,
	          default=0.025)
	parser.add_argument("--hidden", dest="hidden", help="Hidden layer size", type=int,
	          default=32)
	parser.add_argument("--window", dest="window", help="Context window size", type=int,
	          default=4)
	parser.add_argument("--batch", dest="batch_size", help="Batch size", type=int,
	          default=16)
	parser.add_argument("--min-count", dest="min_count", help="Minimum Count", type=int,
	          default=1)
	parser.add_argument("--workers", dest="workers", help="Number of Workers", type=int, default=8)
	parser.add_argument("-l", "--log", dest="log", help="log verbosity level", default="INFO")
	args = parser.parse_args()
  	if args.log == 'DEBUG':
  		sys.excepthook = debug
  	numeric_level = getattr(logging, args.log.upper(), None)
  	logging.basicConfig(level=numeric_level, format=LOGFORMAT)
  	main(args,path)
