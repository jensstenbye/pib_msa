import numpy as np
from itertools import combinations
from Bio import SeqIO


def parse_score_matrix(path):
    ''' Parses file containing score matrix into a score dictionary'''

    score_dict = {}
    with open(path, 'r') as score_file:
        # Parse gap score and build datastructure
        base_list = next(score_file).split()
        score_dict['gap'] = int(base_list.pop(0))
        for base in base_list:
            score_dict[base] = {}

        # Fill datastructure with scores
        for line in score_file:
            score_list = line.split()
            base = score_list.pop(0)
            for matched_base, score in zip(base_list, score_list):
                score_dict[base][matched_base] = int(score)

    return score_dict


def pairalign_score_matrix(seq_1: str, seq_2: str, score_dict: dict):
    ''' Returns scoring matrix of a pairwise global alignment of two sequences
    '''

    # Create score matrix and fill zero'th column and row with gap scores
    score_matrix = np.zeros((len(seq_1) + 1, len(seq_2) + 1))
    score_matrix[:, 0] = np.arange(0, len(seq_1) + 1) * score_dict['gap']
    score_matrix[0, :] = np.arange(0, len(seq_2) + 1) * score_dict['gap']

    # Fill in rest of score matrix
    for i, base_1 in enumerate(seq_1, 1):
        for j, base_2 in enumerate(seq_2, 1):
            from_left = score_matrix[i, (j-1)] + score_dict['gap']
            from_above = score_matrix[(i-1), j] + score_dict['gap']
            from_across = (score_matrix[(i-1), (j-1)]
                           + score_dict[base_1][base_2])

            score_matrix[i, j] = min(from_left, from_above, from_across)

    return score_matrix


def pairalign_backtracking(seq_1: str, seq_2: str, score_dict: dict):
    ''' Returns aligned sequences from pairwise alignment'''
    score_matrix = pairalign_score_matrix(seq_1, seq_2, score_dict)

    # Start backtracking at bottom right in the scoring matrix
    i, j = score_matrix.shape
    i, j = i - 1, j - 1

    # Build aligned sequences from behind
    alignment_1 = []
    alignment_2 = []
    while (i != 0) and (j != 0):
        if (score_matrix[i, j] == (score_matrix[i-1, j-1]
                                   + score_dict[seq_1[i-1]][seq_2[j-1]])):
            # Coming from across
            i -= 1
            j -= 1
            alignment_1.append(seq_1[i])
            alignment_2.append(seq_2[j])

        elif (score_matrix[i, j] == score_matrix[i-1, j] + score_dict['gap']):
            # Coming from right
            i -= 1
            alignment_1.append(seq_1[i])
            alignment_2.append('-')

        elif (score_matrix[i, j] == score_matrix[i, j-1] + score_dict['gap']):
            # Coming from above
            j -= 1
            alignment_1.append('-')
            alignment_2.append(seq_2[j])

        else:
            raise Exception(f'No backtrack path, exciting at i:{i}, j:{j}')

    # If edge reached, fill with gaps
    while i > 0:
        i -= 1
        alignment_1.append(seq_1[i])
        alignment_2.append('-')
    while j > 0:
        j -= 1
        alignment_1.append('-')
        alignment_2.append(seq_2[j])

    alignment_1 = ''.join(alignment_1[::-1])
    alignment_2 = ''.join(alignment_2[::-1])
    return alignment_1, alignment_2


def pairwise_distance_matrix(sequence_list: list, score_dict: dict):
    ''' Creates matrix of pairwise alignment scores between sequences in list
    '''
    pairwise_score_matrix = np.empty((len(sequence_list), len(sequence_list)))
    pairwise_score_matrix.fill(np.nan)

    pairwise_iterator = combinations(enumerate(sequence_list), 2)
    for (idx_seq1, seq_1), (idx_seq2, seq_2) in pairwise_iterator:
        pairalign_score = pairalign_score_matrix(seq_1, seq_2, score_dict)[-1, -1]

        pairwise_score_matrix[idx_seq1, idx_seq2] = pairalign_score
        pairwise_score_matrix[idx_seq2, idx_seq1] = pairalign_score

    return pairwise_score_matrix


def find_center_seq_idx(sequence_list: list, score_dict: dict):
    ''' Finds the key of the sequence with the lowest sum of pairs score'''

    pairwise_score_matrix = pairwise_distance_matrix(sequence_list, score_dict)
    sp_score = np.nansum(pairwise_score_matrix, axis=0)
    center_string_idx = np.argmin(sp_score)

    return center_string_idx


def gusfield_msa(sequence_dict, score_dict, output_tracker = False):
    ''' Conducts the 2sp Gusfield multiple sequence alignment'''

    seq_names = list(sequence_dict.keys())
    seq_values = list(sequence_dict.values())
    seq_index = list(range(len(seq_values)))
    msa_node_idx = [None] * len(seq_values)
    align_tracker = []

    # Find and add the center string to the msa
    center_seq_idx = find_center_seq_idx(seq_values, score_dict)
    center_seq = seq_values[center_seq_idx]
    msa = [list(seq_values.pop(center_seq_idx))]
    msa_node_idx[seq_index.pop(center_seq_idx)] = 0


    for node_idx, seq in zip(seq_index, seq_values):
        # Align each sequence with the center string, and extend the alignment
        align_center, align_new = pairalign_backtracking(center_seq, seq, score_dict)
        extend_multiple_alignment(msa, 0,
                                  align_center, align_new)

        msa_node_idx[node_idx] = len(msa) - 1 
        align_tracker.append(seq_names[0], seq_names[node_idx])     

    msa_dict = construct_msa_dict(msa, msa_node_idx, seq_names)

    if output_tracker:
        return msa_dict, align_tracker
    return msa_dict


def extend_multiple_alignment(multiple_alignment, msa_seq_idx,
                              align_msa_seq, align_new_seq):
    ''' Extends a multiple alignment with an alignment between
        the center string and a new string
    '''
    # Track position in string of current multiple alignment
    # and of the same string in the new pairwise alignment
    pointer_multiple = 0
    pointer_pairwise = 0
    extended_sequence = []

    # Extend multiple alignment handling substitution, 
    # insertion and deletion cases
    while pointer_pairwise <= (len(align_msa_seq)-1):
        if align_msa_seq[pointer_pairwise] == '-':
            # Insertion (i.e. gap in msa-sequence)
            extended_sequence.append(align_new_seq[pointer_pairwise])
            for sequence in multiple_alignment:
                # Correct alignments already present in multiple alignment
                sequence.insert(pointer_multiple, '-')

            pointer_multiple += 1
            pointer_pairwise += 1

        elif multiple_alignment[msa_seq_idx][pointer_multiple] != '-':
            # Deletion or substitution
            extended_sequence.append(align_new_seq[pointer_pairwise])
            pointer_multiple += 1
            pointer_pairwise += 1

        else:
            # Gaps in 'multiple center string' not in 'pairwise center string'
            extended_sequence.append('-')
            pointer_multiple += 1

    while pointer_multiple < (len(multiple_alignment[msa_seq_idx])):
        # Extend trailing gaps
        extended_sequence.append('-')
        pointer_multiple += 1

    multiple_alignment.append(extended_sequence)
    return


def get_sum_of_pairs_score(multiple_alignment_dict, score_dict):
    ''' Calculates the sum of pairs score for a multiple aligneent'''
    score = 0
    pairwise_iterator = combinations(multiple_alignment_dict.values(), 2)
    for (seq_1, seq_2) in pairwise_iterator:
        for base_1, base_2 in zip(seq_1, seq_2):
            if (base_1 != '-') and (base_2 != '-'):
                # Base in both sequences
                score += score_dict[base_1][base_2]
            elif (base_1 == '-') and (base_2 == '-'):
                # Gaps in both sequences
                continue
            else:
                # Gap in one sequence
                score += score_dict['gap']

    return score


def fasta_to_dict(path):
    ''' Parses fasta file into dict with name/sequence as key/value pair'''
    sequence_dict = {}
    for record in SeqIO.parse(path, "fasta"):
        sequence_dict[record.name] = str(record.seq)

    return sequence_dict

def dict_to_fasta(path, sequence_dict):
    ''' Writes sequence_dict to a fasta output file'''
    with open(path, 'w') as outfile:
        for name in sequence_dict:
            outfile.write(f'>{name}\n')
            outfile.write(f'{sequence_dict[name]}\n\n')
    return


def prim_msa(sequence_dict, score_dict, output_tracker = False):
    ''' Conducts multiple sequence alignment following the minimum
        spanning tree created by Prim's alogrithm'''
    seq_names = list(sequence_dict.keys())
    seq_values = list(sequence_dict.values())
    seq_index = list(range(len(seq_values)))
    align_tracker = []

    # Adjacency matrix pairwise contains pairwise distance between all seqs
    adjacency_matrix = pairwise_distance_matrix(seq_values, score_dict)

    # Prims A holds the distances between the tree and unlinked vertices
    # and the node in the tree with this distance
    prims_A = [(0, np.inf)] * len(seq_values)
    for idx in seq_index:
        if not np.isnan(adjacency_matrix[0, idx]):
            prims_A[idx] = ((0, adjacency_matrix[0, idx]))

    # Add root to msa (chosen to be 0)
    # msa_node_idx maps from index in seq_values to index in msa
    msa = [list(seq_values[0])]
    msa_node_idx = [None] * len(seq_values)
    msa_node_idx[0] = 0
    seq_index.remove(0)

    while seq_index:
        # Find the node closest to the tree
        node_idx, (tree_idx, adj) = min(enumerate(prims_A),
                                        key = lambda x: x[1][1])

        # Extend MSA with this node
        align_tree, align_node = pairalign_backtracking(seq_values[tree_idx],
                                                        seq_values[node_idx],
                                                        score_dict)
        extend_multiple_alignment(msa, msa_node_idx[tree_idx],
                                  align_tree, align_node)   
        msa_node_idx[node_idx] = len(msa) - 1
        align_tracker.append((seq_names[tree_idx], seq_names[node_idx]))

        # Update prims A
        prims_A[node_idx] = (node_idx, np.inf)
        seq_index.remove(node_idx)
        for idx in seq_index:
            if adjacency_matrix[node_idx, idx] < prims_A[idx][1]:                
                prims_A[idx] = ((node_idx, adjacency_matrix[node_idx, idx]))    

    msa_dict = construct_msa_dict(msa, msa_node_idx, seq_names)
    if output_tracker:
        return msa_dict, align_tracker
    return msa_dict


def construct_msa_dict(msa: list, msa_node_idx: list, seq_names:list):
    ''' Constructs a dictionary with the aligned sequences from msa matrix,
        sequence names and a mapping between the two'''
    msa = [''.join(sequence) for sequence in msa]

    msa_dict = {}
    for name_idx, sequence in zip(msa_node_idx, msa):
        msa_dict[seq_names[name_idx]] = sequence
    return msa_dict


def inspect_msa_pairwise_correctness(msa, alignment_tracker, sequence_dict):
    return


if __name__ == "__main__":
    sequence_dict_read = fasta_to_dict('C:/Users/jenss/Desktop/pib/sequences.fasta')
    score_dict_read = parse_score_matrix('C:/Users/jenss/Desktop/pib/score_matrix.txt')
    gusfield_alignment = gusfield_msa(sequence_dict_read, score_dict_read)
    sequence_dict_read = fasta_to_dict('C:/Users/jenss/Desktop/pib/sequences.fasta')
    prim_alignment = prim_msa(sequence_dict_read, score_dict_read)
    dict_to_fasta('alignment_prim.fasta', prim_alignment)
    dict_to_fasta('alignment_gus.fasta', gusfield_alignment)
    print(get_sum_of_pairs_score(gusfield_alignment, score_dict_read))
    print(get_sum_of_pairs_score(prim_alignment, score_dict_read))
    #print(get_sum_of_pairs_score(alignment, score_dict))