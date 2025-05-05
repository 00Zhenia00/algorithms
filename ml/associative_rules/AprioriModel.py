from collections import defaultdict
from itertools import combinations

class AprioriModel:
    def __init__(self, min_support=0.5, min_confidence=0.5):
        """
        Initialize the Apriori algorithm with minimum support and confidence thresholds.
        
        Parameters:
        -----------
        min_support : float
            The minimum support threshold (between 0 and 1).
        min_confidence : float
            The minimum confidence threshold for association rules (between 0 and 1).
        """
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._frequent_itemsets = {}
        self._transaction_count = 0
        self._association_rules = []
    
    def _calculate_support_count(self, candidates, transactions):
        """
        Calculate the support count for each candidate itemset.
        
        Parameters:
        -----------
        candidates : list
            List of candidate itemsets
        transactions : list
            List of transactions
            
        Returns:
        --------
        dict : Support count for each candidate
        """
        counts = defaultdict(int)
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if set(candidate).issubset(transaction_set):
                    counts[candidate] += 1
        return counts
    
    def _generate_candidates(self, itemsets, target_size):
        """
        Generate candidate itemsets of size target_size from frequent itemsets of size target_size-1.
        
        Parameters:
        -----------
        itemsets : list
            List of frequent itemsets of size target_size-1
        target_size : int
            Size of the candidate itemsets to generate
            
        Returns:
        --------
        list : Candidate itemsets of size target_size
        """
        candidates = []
        itemsets_list = [frozenset(item) for item in itemsets]

        for i in range(len(itemsets_list)):
            for j in range(i+1, len(itemsets_list)):

                candidate = itemsets_list[i] | itemsets_list[j]
                if (len(candidate) == target_size) and (candidate not in candidates):
                    all_subsets_frequent = True
                    for subset in combinations(candidate, target_size-1):
                        if frozenset(subset) not in itemsets_list:
                            all_subsets_frequent = False
                            break
                    
                    if all_subsets_frequent:
                        candidates.append(candidate)
        
        return candidates
    
    def fit(self, transactions):
        """
        Find frequent itemsets using the Apriori algorithm.
        
        Parameters:
        -----------
        transactions : list
            List of transactions, where each transaction is a list of items
            
        Returns:
        --------
        self : object
        """
        self._transaction_count = len(transactions)
        min_support_count = self._min_support * self._transaction_count
        
        # Find large singleton itemsets (L1)
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1
        
        itemsets_len1 = {item: count for item, count in item_counts.items() if count >= min_support_count}
        self._frequent_itemsets[1] = itemsets_len1
        
        k = 2
        while self._frequent_itemsets.get(k-1, set()):
            # Generate candidates
            candidate_k = self._generate_candidates(list(self._frequent_itemsets[k-1].keys()), k)
            
            # Count support for candidates
            counts = self._calculate_support_count(candidate_k, transactions)
            
            # Filter candidates based on minimum support
            itemsets_lenk = {item: count for item, count in counts.items() if count >= min_support_count}
            
            # Add to frequent itemsets if not empty
            if itemsets_lenk:
                self._frequent_itemsets[k] = itemsets_lenk
            
            k += 1
        
        # Remove the empty last level
        if k-1 in self._frequent_itemsets and not self._frequent_itemsets[k-1]:
            del self._frequent_itemsets[k-1]
        
        # Generate association rules
        self._generate_rules()
        
        return self
    
    def _generate_rules(self):
        """
        Generate association rules from frequent itemsets.
        """
        self._association_rules = []
        
        # For each frequent itemset of size >= 2
        for k in range(2, len(self._frequent_itemsets) + 1):
            for itemset, support_count in self._frequent_itemsets[k].items():
                # Calculate the support of the itemset
                support = support_count / self._transaction_count
                
                # Generate all possible subsets to create rules
                for i in range(1, k):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # Calculate confidence
                        antecedent_support = self._frequent_itemsets[len(antecedent)][antecedent] / self._transaction_count
                        confidence = support / antecedent_support
                        
                        # Add rule if confidence meets threshold
                        if confidence >= self._min_confidence:
                            # Calculate lift
                            consequent_support = self._frequent_itemsets[len(consequent)][consequent] / self._transaction_count
                            lift = confidence / consequent_support
                            
                            self._association_rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
    
    def get_frequent_itemsets(self):
        """
        Return the frequent itemsets found.
        
        Returns:
        --------
        dict : Dictionary mapping itemset size to a dictionary of itemsets and their support counts
        """
        result = {}
        for k, itemsets in self._frequent_itemsets.items():
            result[k] = {frozenset(itemset): count / self._transaction_count for itemset, count in itemsets.items()}
        return result
    
    def get_association_rules(self):
        """
        Return the association rules found.
        
        Returns:
        --------
        list : List of association rules with their support, confidence, and lift
        """
        return self._association_rules
