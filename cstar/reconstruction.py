from nw import NeedlemanWunsch
from itertools import combinations
from multiprocessing import Pool
import argparse
import random
import time
import Levenshtein
import numpy as np
from collections import Counter


# --- Utility and generation functions ---
def get_random_different_char(original_char, alphabet):
    new_char = random.choice(alphabet)
    while new_char == original_char:
        new_char = random.choice(alphabet)
    return new_char

def generate_true_sequence(length, alphabet='ACGT'):
    """Generates a random true sequence."""
    return "".join(random.choice(alphabet) for _ in range(length))

def generate_noisy_copy(true_sequence, r, alphabet='ACGT'):
    """Generates a noisy copy of the sequence based on the error model."""
    if not (0 <= 3 * r <= 1.0):
        raise ValueError("Error rate 'r' is too high (3*r must be <= 1.0)")

    noisy_list = []
    for char in true_sequence:
        action_rand = random.random()
        if action_rand < r:  # Deletion (Prob r)
            pass  # Do nothing, char is deleted
        elif action_rand < 2 * r:  # Insertion before char (Prob r)
            noisy_list.append(random.choice(alphabet))
            noisy_list.append(char)
        elif action_rand < 3 * r:  # Substitution (Prob r)
            noisy_list.append(get_random_different_char(char, alphabet))
        else:  # Keep (Prob 1-3r)
            noisy_list.append(char)
            
    return "".join(noisy_list)

def align_similar(s1, s2):
    change1, change2 = list(), list()
    i = 0
    while s1 != s2:
        if i > len(s1) - 1:
            s1 += s2[i:]
            change1.extend(range(i, i + len(s2[i:])))
            continue
        if i > len(s2) - 1:
            s2 += s1[i:]
            change2.extend(range(i, i + len(s1[i:])))
            continue
        if s1[i] != s2[i]:
            if s1[i] == '-':
                s2 = s2[0:i] + '-' + s2[i:]
                change2.append(i)
            else:
                s1 = s1[0:i] + '-' + s1[i:]
                change1.append(i)
        i += 1
    return sorted(change1), sorted(change2)

def adjust(string_list, indices):
    for i, string in enumerate(string_list):
        for index in indices:
            string = string[:index] + '-' + string[index:]
        string_list[i] = string

def worker(it):
    ((i, string_i), (j, string_j)), scores = it
    model = NeedlemanWunsch(string_i, string_j, scores).nw(True)
    (string_ai, string_aj), score = model['nw'][0], model['score']
    return (i, string_ai), (j, string_aj), score


class CenterStar:

    def __init__(self, scores, strings):
        self.scores = scores
        self.strings = strings
        self.dp = [[0] * (len(strings) + 1) for _ in range(len(strings))]

    def msa(self):
        msa_result = []
        max_row, max_value = 0, 0
        len_strings = len(self.strings)
        
        print(f"  Calculating {len_strings * (len_strings - 1) // 2} pairwise alignments...")
        
        tasks = tuple(combinations(zip(range(len_strings), self.strings), 2))
        tasks = zip(tasks, (self.scores for _ in range(len(tasks))))

        with Pool() as pool:
            result = pool.map(worker, tasks)
            for elem in result:
                (i, string_i), (j, string_j), score = elem
                ''' (0, 1, 2) => 0 is the first aligned string
                                 1 is the second aligned string
                                 2 is the score
                '''
                self.dp[i][j] = (string_i, string_j, score)
                self.dp[j][i] = (string_j, string_i, score)
                self.dp[i][-1] += score
                self.dp[j][-1] += score

                if self.dp[j][-1] > max_value:
                    max_row = j
                    max_value = self.dp[j][-1]
                if self.dp[i][-1] > max_value:
                    max_row = i
                    max_value = self.dp[i][-1]

            print(f"  Finding center sequence... (Index: {max_row}, Score: {max_value})")
            print(f"  Center sequence length: {len(self.strings[max_row])}")
            print(f"  > Edit distances of center to true sequence: {Levenshtein.distance(self.strings[max_row], self.strings[0])}")
            print(f"  Center sequence: {self.strings[max_row][:50]}...")
            
            print("  Aligning all sequences to the center sequence...")
            for i in range(len_strings):
                if i == max_row:
                    continue
                if not msa_result:
                    msa_result.extend(self.dp[max_row][i][0: 2])
                    continue

                new = list(self.dp[max_row][i][0: 2])
                ch_index1, ch_index2 = align_similar(msa_result[0], new[0])

                adjust(msa_result, ch_index1)
                adjust(new, ch_index2)
                msa_result.extend(new[1:])

        print(f"  MSA complete. Aligned {len(msa_result)} sequences.")
        return msa_result

# --- Evaluation functions ---
def run_evaluation(true_length, r, num_samples, num_runs):
    """Runs the reconstruction multiple times and evaluates performance."""
    print("--- Starting Evaluation ---")
    print(f"Parameters: True Length={true_length}, r={r}, Num Samples={num_samples}, Num Runs={num_runs}")
    
    # Default scoring for DNA alignment (match=1, mismatch=-1, gap=-1)
    scores = ["1", "-1", "-1"] 
    
    success_count = 0
    total_edit_distance = 0
    reconstruction_times = []

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        # 1. Generate true sequence and samples
        true_seq = generate_true_sequence(true_length)
        print(f"  True Sequence (first 50 chars): {true_seq[:50]}...")
        samples = [generate_noisy_copy(true_seq, r) for _ in range(num_samples)]
        print(f"  Generated {num_samples} noisy samples.")

        # 2. Run center-star MSA
        start_time = time.time()
        msa_result = CenterStar(scores, samples).msa()
        
        # 3. Generate consensus from MSA
        print("  Generating consensus sequence from MSA...")
        msa_len = len(msa_result[0])
        consensus = []
        for i in range(msa_len):
            column = [seq[i] for seq in msa_result]
            counts = Counter(column)
            # Ignore gaps when creating consensus
            if '-' in counts and counts['-'] >= len(column) / 2:
                continue
            most_common = counts.most_common(2)
            most_common = [item for item in most_common if item[0] != '-']
            if most_common:
                consensus.append(most_common[0][0])
        
        estimated_seq = "".join(consensus)
        end_time = time.time()
        run_time = end_time - start_time
        reconstruction_times.append(run_time)
        
        print(f"  Reconstruction Time: {run_time:.2f} seconds")
        print(f"  Estimated Sequence Length: {len(estimated_seq)}")
        print(f"  Estimated Sequence (first 50 chars): {estimated_seq[:50]}...")

        # 4. Compare result
        edit_distance = Levenshtein.distance(estimated_seq, true_seq)
        if edit_distance == 0:
            success_count += 1
            print("  Result: SUCCESS")
        else:
            total_edit_distance += edit_distance
            print(f"  Result: FAILURE (Edit Distance: {edit_distance})")

    # --- 5. Summarize Evaluation ---
    print("\n--- Evaluation Summary ---")
    print(f"Parameters: True Length={true_length}, r={r}, Num Samples={num_samples}, Num Runs={num_runs}")

    success_rate = (success_count / num_runs) * 100
    average_distance_on_fail = total_edit_distance / (num_runs - success_count) if (num_runs - success_count) > 0 else 0
    average_time = sum(reconstruction_times) / num_runs if num_runs > 0 else 0

    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{num_runs})")
    print(f"Average Edit Distance (when failed): {average_distance_on_fail:.2f}")
    print(f"Average Reconstruction Time: {average_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple sequence alignment')
    parser.add_argument("--inputfile", help="input file location")
    parser.add_argument("--outputfile", help="output file location")
    parser.add_argument("--generate", action="store_true", help="auto-generate sequences instead of reading from file")
    parser.add_argument("--true_length", type=int, default=100, help="length of true sequence for generation")
    parser.add_argument("--error_rate", type=float, default=0.005, help="error rate for sequence generation")
    parser.add_argument("--num_samples", type=int, default=5, help="number of sequences to generate")
    parser.add_argument("--num_runs", type=int, default=1, help="number of evaluation runs")
    args = parser.parse_args()

    if args.generate:
        # Auto-generate mode
        if not (0 <= 3 * args.error_rate <= 1.0):
            print(f"ERROR: Invalid value for error_rate={args.error_rate}. Need 0 <= 3*error_rate <= 1.0")
            exit(1)
            
        run_evaluation(args.true_length, args.error_rate, args.num_samples, args.num_runs)
    else:
        # Original file-based mode
        if not args.inputfile or not args.outputfile:
            print("Error: inputfile and outputfile required when not in generate mode")
            parser.print_help()
            exit(1)
            
        with open(args.inputfile, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            scores = lines.pop(0).split(',')
            
            print(f"Reading {len(lines)} sequences from {args.inputfile}")
            msa = CenterStar(scores, lines).msa()

            print(f"Writing aligned sequences to {args.outputfile}")
            with open(args.outputfile, 'w') as out:
                out.writelines(map(lambda x: x + '\n', msa))