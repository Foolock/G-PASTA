def read_numbers_from_file(filename, lines):
    numbers = []
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if line_number in lines:
                parts = line.split(':')
                if len(parts) == 2:
                    number = parts[1].strip().rstrip('%')
                    try:
                        number = float(number)
                        numbers.append(number)
                    except ValueError:
                        print(f"Invalid number found on line {line_number}: {number}")
    results = 0
    for i in range(len(numbers)):
        results += numbers[i]
    return results / len(numbers)

# Usage example
rebuild_time = [3, 7, 11]  # Specify the line numbers you want to read here
run_time = [4, 8, 12]

# files 
exp_path = './des_perf_exp/'

file_path1 = exp_path + 'partition_120000_btask_only_partition_cost_not_reverse.txt'
file_path2 = exp_path + 'partition_140000_btask_only_partition_cost_not_reverse.txt'
file_path3 = exp_path + 'partition_160000_btask_only_partition_cost_not_reverse.txt'
file_path4 = exp_path + 'partition_180000_btask_only_partition_cost_not_reverse.txt'
file_path5 = exp_path + 'partition_200000_btask_only_partition_cost_not_reverse.txt'
file_path6 = exp_path + 'partition_220000_btask_only_partition_cost_not_reverse.txt'
file_path7 = exp_path + 'partition_240000_btask_only_partition_cost_not_reverse.txt'
file_path8 = exp_path + 'partition_260000_btask_only_partition_cost_not_reverse.txt'
file_path9 = exp_path + 'partition_280000_btask_only_partition_cost_not_reverse.txt'
file_path10 = exp_path + 'partition_300000_btask_only_partition_cost_not_reverse.txt'

rebuild_time1 = read_numbers_from_file(file_path1, rebuild_time)/1000
rebuild_time2 = read_numbers_from_file(file_path2, rebuild_time)/1000
rebuild_time3 = read_numbers_from_file(file_path3, rebuild_time)/1000
rebuild_time4 = read_numbers_from_file(file_path4, rebuild_time)/1000
rebuild_time5 = read_numbers_from_file(file_path5, rebuild_time)/1000
rebuild_time6 = read_numbers_from_file(file_path6, rebuild_time)/1000
rebuild_time7 = read_numbers_from_file(file_path7, rebuild_time)/1000
rebuild_time8 = read_numbers_from_file(file_path8, rebuild_time)/1000
rebuild_time9 = read_numbers_from_file(file_path9, rebuild_time)/1000
rebuild_time10 = read_numbers_from_file(file_path10, rebuild_time)/1000


run_time1 = read_numbers_from_file(file_path1, run_time)/1000
run_time2 = read_numbers_from_file(file_path2, run_time)/1000
run_time3 = read_numbers_from_file(file_path3, run_time)/1000
run_time4 = read_numbers_from_file(file_path4, run_time)/1000
run_time5 = read_numbers_from_file(file_path5, run_time)/1000
run_time6 = read_numbers_from_file(file_path6, run_time)/1000
run_time7 = read_numbers_from_file(file_path7, run_time)/1000
run_time8 = read_numbers_from_file(file_path8, run_time)/1000
run_time9 = read_numbers_from_file(file_path9, run_time)/1000
run_time10 = read_numbers_from_file(file_path10, run_time)/1000

print("rebuild_time = [", end = '')
print(rebuild_time1, end = ', ')
print(rebuild_time2, end = ', ')
print(rebuild_time3, end = ', ')
print(rebuild_time4, end = ', ')
print(rebuild_time5, end = ', ')
print(rebuild_time6, end = ', ')
print(rebuild_time7, end = ', ')
print(rebuild_time8, end = ', ')
print(rebuild_time9, end = ', ')
print(rebuild_time10, end = ']\n')

print("run_time = [", end = '')
print(run_time1, end = ', ')
print(run_time2, end = ', ')
print(run_time3, end = ', ')
print(run_time4, end = ', ')
print(run_time5, end = ', ')
print(run_time6, end = ', ')
print(run_time7, end = ', ')
print(run_time8, end = ', ')
print(run_time9, end = ', ')
print(run_time10, end = ']')






























