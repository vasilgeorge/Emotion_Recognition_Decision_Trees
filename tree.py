
import numpy as np



class Tree:

    #op = None #values 1-45

    #kids = [] #1x2 cell array with values 0 or 1

    #class_ = None #empty for internal node ------- 0 or 1 for leaf node

    def __init__(self,examples,attributes,binary_targets):

        self.op = None #values 1-45

        self.kids = [] #1x2 cell array with values 0 or 1
        self.info_gain = None
        self.class_ = None #empty for internal node ------- 0 or 1 for leaf node

        self.attributes = attributes
        self.examples = examples
        self.binary_targets = binary_targets

    def majority_value(self):

        majority = 1

        count_0 = self.binary_targets.count(0)
        count_1 = self.binary_targets.count(1)

        #count_0 = np.count_nonzero(self.binary_targets == 0)
        #count_1 = np.count_nonzero(self.binary_targets == 1)

        if count_0 >= count_1: #check equality
            majority = 0

        return majority

    # returns the number of positive and negative examples of training data
    # attribute should be between 1-45, value either 0 or 1 and emotion between 1-6
    def split_p_n(self, attribute, value):

        p = 0
        n = 0
        # the whole set of training data for a certain target (emotion)
        if(attribute == -1):
            for i in range(len(self.binary_targets)):
                if(self.binary_targets[i] == 1):
                    p += 1
                else:
                    n += 1

        # the subset for a specific attribute
        else:
            # parameter attribute is between 1-45 while the array's indexes are 0-44
            y = attribute
            y = int(attribute)
            #print "attr"
            #print value
            #print len(self.examples)
            for i in range(len(self.examples)):
                # attribute has the value (0 or 1)
                if(self.examples[i][y] == value):
                    if(self.binary_targets[i] == 1):
                        p += 1
                    else:
                        n += 1

        return p, n


    def compare_targets(self):

        same_value = False
        #print self.binary_targets
        count_0 = 0
        count_1 = 0
        for i in range(len(self.binary_targets)):
            if self.binary_targets[i]==0:
                count_0+=1
            elif self.binary_targets[i]==1:
                count_1+=1
            else:
                print error

        if len(set(self.binary_targets)) == 1:
            same_value=True

        elif count_0>=0.95*len(self.binary_targets) or count_1>=0.95*len(self.binary_targets):
            same_value=True


        return same_value

    def attr_is_empty(self):

        is_empty = False
        if len(set(self.attributes)) == 1:
            if self.attributes[0] == 0:
                is_empty = True

            return is_empty


        # TO INVESTIGATE: it is not necessarily true that each time an attribute
        # is chosen it should not be existing in the table (by entering 0)
    def choose_best_attribute(self):

        max_pos = -1
        max_gain = -1

        for i in range(len(self.attributes)):
                # attributes[i] will be 1 if this remains unused
                # this is executed only the first time an unused
                # attribute is found
            if (max_pos == -1) and (self.attributes[i] == 1):
                max_pos = i
                max_gain = self.gain(i)


            # for all the other times
            if self.attributes[i] == 1:
                #print self.gain(self.attributes[i])
                #print self.gain(self.attributes[max_pos])
                if(self.gain(i) > self.gain(max_pos)):
                    #print true
                    max_pos = i
                    max_gain = self.gain(i)

        return max_pos, max_gain


    def _I(self,p, n):
        #print p,n
        if p==0 and n==0:
            result = 0
        else:
            first_part = float(p) / (p + n)
            second_part = float(n) / (p + n)
            #print first_part, second_part
            if first_part!=0:
                result_1 = -first_part * np.log2(first_part)
            else:
                result_1 = 0

            if second_part!=0:
                result_2 = -second_part * np.log2(second_part)
            else:
                result_2 = 0

            result = result_1 + result_2
            #print result

        return result


    def remainder(self,attribute):
                # attribute parameter = -1 since we want the values
                # (positive and negative) of the whole set of training data
        p, n = self.split_p_n(-1, 1)

        p0, n0 = self.split_p_n(attribute, 0)
        p1, n1 = self.split_p_n(attribute, 1)

        result_1 = (float(p0 + n0) / (p + n)) * self._I(p0, n0)
        result_2 = (float(p1 + n1) / (p + n)) * self._I(p1, n1)

        result = result_1 + result_2

        return result


    def gain(self,attribute):
        # attribute parameter = -1 since we want the values
        # (positive and negative) of the whole set of training data
        p, n = self.split_p_n(-1, 1)
        result = self._I(p, n) - self.remainder(attribute)

        return result
