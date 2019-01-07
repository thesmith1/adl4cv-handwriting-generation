import os

src_label_file = "/home/paulstpr/Desktop/out_labels.txt"
src_folder = "/home/paulstpr/Desktop/processed/"

if __name__ == '__main__':

    with open(src_label_file, "r") as f:
        new_lines = []
        for line in f.readlines():
            new_line = "_" + line
            new_lines.append(new_line)

    with open(src_label_file, "w") as f:
        for new_line in new_lines:
            f.write(new_line)

    for filename in os.listdir(src_folder):       os.rename(os.path.join(src_folder, filename), os.path.join(src_folder, "_" + filename))
