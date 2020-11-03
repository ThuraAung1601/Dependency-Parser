from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        # outputs to action, then label!
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            features = self.extractor.get_input_representation(words, pos, state).reshape(1,6)
            possible_actions = self.model.predict(features)
            p = possible_actions[0]
            actions = dict()

            for i in range(len(p)):
                actions[i] = p[i]
            a = sorted(actions.items(), key=lambda x: x[1])
            a.reverse()
            
            for check_action in a:
                c, l = self.output_labels[check_action[0]]
                violation = False 
                if(len(state.stack) == 0 and (c == 'left_arc' or c == 'right_arc')):
                    violation = True 
                if(len(state.buffer) == 1 and c == 'shift' and len(state.stack) != 0):
                    violation = True
                if(len(state.stack) == 1 and c == 'left_arc'):
                    violation = True
                if not violation: 
                    if c == 'right_arc': state.right_arc(l)
                    if c == 'left_arc': state.left_arc(l)
                    if c == 'shift': state.shift()
                    break 

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
