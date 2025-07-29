file_path = '../torch_dna/cifar10_candidate_permut.out'  # Your file path
max_passes=5

dict_list = []  # Stores all dictionaries
used_keys_per_dict = []  # Tracks used keys for each dictionary
used_values_per_dict = []  # Tracks used values for each dictionary
for _ in range(max_passes):
    with open(file_path, 'r') as file:
        for line in file:
            first, second = map(int, line.strip().split())
            placed = False

            # Try to place in existing dictionaries
            for i in range(len(dict_list)):
                if (second not in used_keys_per_dict[i] and
                        first not in used_values_per_dict[i]):
                    dict_list[i][second] = first
                    used_keys_per_dict[i].add(second)
                    used_values_per_dict[i].add(first)
                    placed = True
                    break

            # If couldn't place, create new dictionary
            if not placed:
                new_dict = {}
                new_used_keys = set()
                new_used_values = set()

                new_dict[second] = first
                new_used_keys.add(second)
                new_used_values.add(first)

                dict_list.append(new_dict)
                used_keys_per_dict.append(new_used_keys)
                used_values_per_dict.append(new_used_values)

ordered_backdoor_lists = []
for i, d in enumerate(dict_list, 1):
    if len(d) == 10:
        ordered_backdoor_lists.append([])
        for j in range(10) :
            ordered_backdoor_lists[-1].append(d[j])
        commastr = ",".join(map(str, ordered_backdoor_lists[-1]))
        print(commastr)