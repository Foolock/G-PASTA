import re
import sys
import os
import threading

# circuit with sdc : c17, c1355, ac97_ctrl, aes_core, c17_slack, c1908, c2670, c3540, c3_slack, c432, c499, c5315, c6288, c7552
#                    c7552_slack, c880, des_perf, s1196, s1494, s27, s27_spef, s344, s349, s386, s400, s510, s526, vga_lcd, 


def read_file(file1):

  with open(file1, "r") as f1:
    file1_lines = f1.readlines()

  return file1_lines



def process_input(line, ite, f):
  prefix = "A" + str(ite) 
  line = re.sub(r"^input\s+(\w+)", rf"input {prefix}_\1", line)
  f.write(line)

def process_output(line, ite, f):
  prefix = "A" + str(ite)
  line = re.sub(r"^output\s+(\w+)", rf"output {prefix}_\1", line)
  f.write(line)

def process_wire(line, ite, f):
  prefix = "A" + str(ite)
  line = re.sub(r"^wire\s+(\w+)", rf"wire {prefix}_\1", line)
  f.write(line)

def process_cell(line, ite, f):
  prefix = "A" + str(ite) 
  line = re.sub(r"inst(\w+)", rf"{prefix}_inst\1", line)
  line = re.sub(r'\.([^(]*)\(([^)]*)\)', rf'.\1({prefix}_\2)', line)
  f.write(line)



def process_file_v(file1, file1_lines, output_file_path, iterations):

  all_cells = []

  copy_file1_lines = file1_lines
  
  ite = 0

  with open(output_file_path, "a") as f:

    for ite in range(iterations):
      for line1 in copy_file1_lines:

        if "input" in line1:
          process_input(line1, ite, f)
  
        elif "output" in line1:
          process_output(line1, ite, f)
        
        elif "wire" in line1:
          if ";" in line1:
            process_wire(line1, ite, f)
  
        elif "inst" in line1:
          process_cell(line1, ite, f) 
  
        elif "module" in line1 and "end" not in line1:
          pass
  
        elif "\n" == line1:
          f.write(line1)
        
        elif "endmodule" in line1:
          if ite == int(iterations)-1: 
            f.write(line1)
  
        elif "Start" in line1:
          pass
  
        else:
          if ite == 0:   
            if ");" in line1:
              all_cells.append(line1[:-3] + ",\n")
            else:
              all_cells.append(line1)

      copy_file1_lines = file1_lines


  all_cells_string = "module " + file1 + "_" + str(iterations) + " (\n"
  for ite in range(iterations):
    for cell in all_cells:
      prefix = "A" + str(ite) + "_"
      all_cells_string = all_cells_string+(prefix+cell)

  with open(output_file_path, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(all_cells_string[:-2]+");\n" + content)


if __name__ == "__main__":

  file1_v_path = sys.argv[1] + '.v'
  iteration = int(sys.argv[2])
  output_v_path    = sys.argv[1] + "_" + str(iteration) + ".v"

  if os.path.exists(output_v_path):
    os.remove(output_v_path)
  

  file1_lines = read_file(file1_v_path)

  process_file_v(sys.argv[1], file1_lines, output_v_path, iteration)


